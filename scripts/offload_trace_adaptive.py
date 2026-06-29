"""Trace-getriebener Offloading-Simulator mit Adaptive Stopping.

Kombiniert in einem einzigen Forward-Pass:
  - Block-Zugriffs-Traces  (aux["iters"][r]["topk_idx"])
  - Logits pro Iteration   (logits_all)

Daraus werden r_stop-Werte fuer alle Stopping-Configs berechnet und
Cache-Simulationen unter vier Policies durchgefuehrt.

Vergleich Stopping-Configs:
  fixed_R2 / R3 / R4 / R6 (Baselines)
  kl_div_minR3_t0.005      (Best Candidate aus adaptive_stopping.py)

Cache-Policies:
  lru                 LRU-Eviction (Baseline)
  lfu                 Least-Frequently-Used Eviction
  hot_block_pinned    Top-K/2 heisseste Bloecke dauerhaft resident, LRU fuer Rest
  transition_prefetch Naechsten wahrscheinlichsten Core prefetchen

Hauptfrage:
  Spart adaptive stopping Compute, ohne die Offload-Geometrie zu verschlechtern?
  Gibt es stabile Leiterbahn-Strukturen (Hot-Blocks, Hot-Cores, Transitions)?

Nutzung:
  python scripts/offload_trace_adaptive.py --device cuda --seeds 0 1
  python scripts/offload_trace_adaptive.py --device cuda --seeds 0 1 2
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import OrderedDict, Counter
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rblm.heteromini import HeteroMiniData, DATA_ROOT
from rblm import model_io

HELDOUT_DATA_ROOT = DATA_ROOT + "_heldout"

RESULTS = os.path.join(ROOT, "results")

CKPTS = {
    0: "results/BASELINE_srcore_b32_k8_R6_15k.pt",
    1: "results/hm_cont_hm_srcore_b32_k8_R6_s1.pt",
    2: "results/hm_cont_hm_srcore_b32_k8_R6_s2.pt",
}

STOPPING_CONFIGS = [
    {"label": "fixed_R2",        "kind": "fixed",  "fixed_R": 2, "min_R": 2},
    {"label": "fixed_R3",        "kind": "fixed",  "fixed_R": 3, "min_R": 3},
    {"label": "fixed_R4",        "kind": "fixed",  "fixed_R": 4, "min_R": 4},
    {"label": "fixed_R6",        "kind": "fixed",  "fixed_R": 6, "min_R": 6},
    {"label": "kl_minR3_t0.005", "kind": "kl_div", "theta": 0.005, "min_R": 3},
]

CACHE_KS      = [4, 8, 16, 32, 64]
BANDWIDTH_GBS = 16.0          # PCIe default (RAM -> GPU)
N_PIN_BLOCKS  = 16            # Anzahl der gepinnten Bloecke bei hot_block_pinned

# ────────────────────────────────────────────────────────────────────────────────
# Stopping-Kriterien (analog zu adaptive_stopping.py)
# ────────────────────────────────────────────────────────────────────────────────

def _kl(lnew, lold):
    lp = F.log_softmax(lnew, dim=-1)
    lq = F.log_softmax(lold, dim=-1)
    return (torch.exp(lp) * (lp - lq)).sum(dim=-1)


def _entropy(logits):
    p = torch.softmax(logits, dim=-1)
    return -(p * torch.log(p + 1e-9)).sum(dim=-1)


def compute_r_stop(logits_stack, cfg, R_max, device):
    """logits_stack: (R, B, T, V) -> r_stop (B, T) 1-basiert."""
    R, B, T, V = logits_stack.shape
    kind  = cfg["kind"]
    min_R = cfg.get("min_R", 1)

    if kind == "fixed":
        return torch.full((B, T), min(cfg["fixed_R"], R_max),
                          dtype=torch.long, device=device)

    r_stop  = torch.full((B, T), R_max, dtype=torch.long, device=device)
    stopped = torch.zeros(B, T, dtype=torch.bool, device=device)

    if kind == "kl_div":
        theta = cfg["theta"]
        for ri in range(1, R):
            r1b = ri + 1
            kl  = _kl(logits_stack[ri], logits_stack[ri - 1])
            trg = (kl < theta) & (r1b >= min_R) & ~stopped
            r_stop  = torch.where(trg, torch.full_like(r_stop, r1b), r_stop)
            stopped = stopped | trg

    elif kind == "entropy_drop":
        theta  = cfg["theta"]
        H_prev = _entropy(logits_stack[0])
        for ri in range(1, R):
            r1b   = ri + 1
            H_cur = _entropy(logits_stack[ri])
            trg   = ((H_prev - H_cur).abs() < theta) & (r1b >= min_R) & ~stopped
            r_stop  = torch.where(trg, torch.full_like(r_stop, r1b), r_stop)
            stopped = stopped | trg
            H_prev  = H_cur

    return r_stop


# ────────────────────────────────────────────────────────────────────────────────
# Sammlung: ein Forward-Pass fuer Traces + r_stops
# ────────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect(model, data, stopping_cfgs, n_batches, bs, seq_len, device):
    """
    Einzel-Pass: traces (R,B,T,k) + r_stop pro Config + Domains.
    logits_stack wird pro Batch berechnet und sofort verworfen.
    """
    model.eval()
    all_traces  = []
    all_r_stops = {c["label"]: [] for c in stopping_cfgs}
    all_doms    = []

    for _ in range(n_batches):
        toks, tgt, mask, dom = data.batch(bs, seq_len, device=device, mode="contiguous")
        logits_all, aux = model(toks)                              # (list R*(B,T,V)), aux
        logits_stack = torch.stack(logits_all, dim=0)             # (R, B, T, V)
        R_actual = logits_stack.shape[0]
        trace = np.stack([a["topk_idx"].cpu().numpy()
                          for a in aux["iters"]])                  # (R, B, T, k)

        for cfg in stopping_cfgs:
            rs = compute_r_stop(logits_stack, cfg, R_actual, device=device)
            all_r_stops[cfg["label"]].append(rs.cpu().numpy())    # (B, T)

        all_traces.append(trace)
        all_doms.append(dom.cpu().numpy())

    return all_traces, all_r_stops, all_doms


# ────────────────────────────────────────────────────────────────────────────────
# Core-Analyse: Hot-Blocks, Hot-Cores, Transitions
# ────────────────────────────────────────────────────────────────────────────────

def analyze_cores(traces, doms, domain_names):
    """
    Verwendet r=0 (r1) als Core-Identitaet — r1 = Adressierung, r2+ = Wiederverwendung.
    Gibt zurueck: hot_blocks, hot_cores, Transitions, domain-spezifische Block-Dichten.
    """
    block_counter = Counter()
    core_counter  = Counter()
    trans_counter = Counter()
    dom_block     = {d: Counter() for d in domain_names}

    for trace, dom in zip(traces, doms):
        R, B, T, k = trace.shape
        for b in range(B):
            d = int(dom[b])
            dname = domain_names[d] if 0 <= d < len(domain_names) else None
            prev_core = None
            for t in range(T):
                core = frozenset(int(x) for x in trace[0, b, t])   # r=0 = Core
                core_counter[core] += 1
                for blk in core:
                    block_counter[blk] += 1
                    if dname:
                        dom_block[dname][blk] += 1
                if prev_core is not None:
                    trans_counter[(prev_core, core)] += 1
                prev_core = core

    # Prefetch-Map: fuer jeden Core -> wahrscheinlichster Nachfolge-Core
    next_core_map: dict[frozenset, frozenset] = {}
    from_ctr: dict[frozenset, Counter] = {}
    for (src, dst), cnt in trans_counter.items():
        from_ctr.setdefault(src, Counter())[dst] += cnt
    for src, ctr in from_ctr.items():
        next_core_map[src] = ctr.most_common(1)[0][0]

    hot_blocks = block_counter.most_common(32)
    hot_cores  = core_counter.most_common(50)

    # Code vs. andere Domänen: Block-Jaccard
    code_top  = {b for b, _ in dom_block.get("code", Counter()).most_common(16)}
    other_top = set()
    for d in domain_names:
        if d != "code":
            other_top.update(b for b, _ in dom_block.get(d, Counter()).most_common(16))
    jaccard = len(code_top & other_top) / max(len(code_top | other_top), 1)

    return {
        "n_unique_cores":           len(core_counter),
        "hot_blocks":               hot_blocks,          # list[(int, int)]
        "hot_cores":                hot_cores,           # list[(frozenset, int)]
        "next_core_map":            next_core_map,       # frozenset -> frozenset (intern)
        "top_transitions":          [
            {"from": sorted(s), "to": sorted(d), "count": c}
            for (s, d), c in trans_counter.most_common(20)
        ],
        "dom_block":                dom_block,           # {str: Counter}
        "code_other_block_jaccard": round(jaccard, 3),
    }


# ────────────────────────────────────────────────────────────────────────────────
# Cache-Simulationen
# ────────────────────────────────────────────────────────────────────────────────

def make_streams(traces, r_stops_batches):
    """
    Baut per-Sequenz-Streams mit Trimming auf r_stop.
    Jeder Stream: Liste von np-Arrays (Block-IDs pro Iterationsschritt).
    In SR-Core sind die Block-IDs pro Token ueber alle Iters identisch
    (r1 waehlt, r2+ nutzen denselben Core) — Trimming reduziert nur Laenge.
    """
    streams = []
    for trace, r_stop in zip(traces, r_stops_batches):
        R, B, T, k = trace.shape
        for b in range(B):
            seq = []
            for t in range(T):
                r_max = int(r_stop[b, t])
                for r in range(r_max):
                    seq.append(np.unique(trace[r, b, t]))
            streams.append(seq)
    return streams


def _flat(streams):
    out = []
    for seq in streams:
        for req in seq:
            out.extend(int(b) for b in req)
    return out


def sim_lru(flat, K):
    cache = OrderedDict()
    miss = 0
    for b in flat:
        if b in cache:
            cache.move_to_end(b)
        else:
            miss += 1
            cache[b] = True
            if len(cache) > K:
                cache.popitem(last=False)
    return miss


def sim_lfu(flat, K):
    """Global-Frequency LFU — evictet den am seltensten zugegriffenen Block."""
    global_freq = Counter()
    cache_freq  = {}
    miss = 0
    for b in flat:
        global_freq[b] += 1
        if b in cache_freq:
            cache_freq[b] = global_freq[b]
        else:
            miss += 1
            if len(cache_freq) >= K:
                victim = min(cache_freq, key=lambda x: cache_freq[x])
                del cache_freq[victim]
            cache_freq[b] = global_freq[b]
    return miss


def sim_hot_block_pinned(flat, K, pinned):
    """
    `pinned` immer resident (kein Evict, kein Miss-Zählen).
    LRU fuer die verbleibenden K - |pinned| Slots.
    """
    pinned_set = set(pinned)
    k_lru = max(0, K - len(pinned_set))
    cache = OrderedDict()
    miss  = 0
    for b in flat:
        if b in pinned_set:
            continue                         # immer resident -> immer Hit
        if b in cache:
            cache.move_to_end(b)
        else:
            miss += 1
            cache[b] = True
            if len(cache) > k_lru:
                cache.popitem(last=False)
    return miss


def sim_transition_prefetch(traces, r_stops_batches, K, next_core_map):
    """
    Per-Token-Simulation mit 1-Step Transition-Prefetch.
    Wenn Core C angefragt wird: lade sofort den wahrscheinlichsten Nachfolge-Core D
    in den Cache. Naechster Token findet D bereits resident.
    Misst Misses, die trotz Prefetch auftreten (Mispredictions, erste Zugriife).
    """
    cache = OrderedDict()
    total_miss = 0

    for trace, r_stop in zip(traces, r_stops_batches):
        R, B, T, k = trace.shape
        for b in range(B):
            for t in range(T):
                core_blocks = [int(x) for x in np.unique(trace[0, b, t])]
                core_fset   = frozenset(core_blocks)

                # Lade Core (Miss wenn nicht resident)
                for blk in core_blocks:
                    if blk in cache:
                        cache.move_to_end(blk)
                    else:
                        total_miss += 1
                        cache[blk] = True
                        if len(cache) > K:
                            cache.popitem(last=False)

                # Prefetch: wahrscheinlichster Nachfolge-Core
                predicted = next_core_map.get(core_fset)
                if predicted:
                    for blk in predicted:
                        if blk not in cache:
                            cache[blk] = True    # lade asynchron in Cache
                            if len(cache) > K:
                                cache.popitem(last=False)
                        else:
                            cache.move_to_end(blk)

    return total_miss


def reuse_dists(streams):
    dists = []
    for seq in streams:
        last = {}
        i = 0
        for req in seq:
            for b in req:
                b = int(b)
                if b in last:
                    dists.append(i - last[b])
                last[b] = i
            i += 1
    if not dists:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0}
    d = np.array(dists)
    return {"p50": round(float(np.percentile(d, 50)), 1),
            "p90": round(float(np.percentile(d, 90)), 1),
            "p99": round(float(np.percentile(d, 99)), 1)}


# ────────────────────────────────────────────────────────────────────────────────
# Metriken pro Policy × K
# ────────────────────────────────────────────────────────────────────────────────

def run_cache_sims(streams, traces, r_stops_batches, Ks, block_bytes_fp16,
                   pinned_blocks, next_core_map, bw_bps, n_tokens, flops_per_token):
    """Fuehrt alle 4 Policies × alle Ks durch. Gibt dict zurueck."""
    flat   = _flat(streams)
    rd     = reuse_dists(streams)

    results = {}
    for policy in ("lru", "lfu", "hot_block_pinned", "transition_prefetch"):
        per_K = {}
        for K in Ks:
            if policy == "lru":
                miss = sim_lru(flat, K)
            elif policy == "lfu":
                miss = sim_lfu(flat, K)
            elif policy == "hot_block_pinned":
                miss = sim_hot_block_pinned(flat, K, pinned_blocks)
            else:   # transition_prefetch
                miss = sim_transition_prefetch(traces, r_stops_batches, K, next_core_map)

            mpt  = miss / max(n_tokens, 1)
            b16  = mpt * block_bytes_fp16
            b8   = mpt * block_bytes_fp16 / 2
            b4   = mpt * block_bytes_fp16 / 4
            ai   = flops_per_token / b16 if b16 > 0 else float("nan")
            per_K[K] = {
                "miss_at_K":            round(miss / max(len(flat), 1), 4),
                "misses_per_token":     round(mpt, 3),
                "bytes_per_token_fp16": int(round(b16)),
                "bytes_per_token_int8": int(round(b8)),
                "bytes_per_token_int4": int(round(b4)),
                "transfer_us_fp16":     round(b16 / bw_bps * 1e6, 2),
                "arith_intensity":      round(ai, 1) if ai == ai else None,
            }
        results[policy] = {"per_K": per_K, "reuse": rd}

    return results


# ════════════════════════════════════════════════════════════════════════════════
# LEITERBAHN-SIMULATOR — K-Sweep x Policy-Sweep
# ════════════════════════════════════════════════════════════════════════════════

LB_KS          = [8, 16, 24, 32, 48, 64]   # Cache-Groessen fuer den Sweep
LB_N_PIN       = 8                           # gepinnte Bloecke (global-pin policies)
LB_N_PIN_DOM   = 6                           # gepinnte Bloecke pro Domaene
LB_N_HOT_CORE  = 1                           # Anzahl Top-Cores fuer hot_core_pin
LB_TARGET_CFG  = "kl_minR3_t0.005"          # Stopping-Config fuer Leiterbahn-Analyse

LEITERBAHN_POLICIES = [
    "lru",               # Baseline
    "transition_pf",     # 1-Step Transitions-Prefetch
    "hot_block_pin",     # Top-k globale Bloecke resident
    "domain_hot_pin",    # Top-k Bloecke pro Domaene resident (oracle label)
    "hot_core_pin",      # Bloecke des Top-1 Cores resident
    "trans+domain_hot",  # Domain-Pin + Prefetch (oracle label)
    "trans+hot_core",    # Hot-Core-Pin + Prefetch (kein Domain-Label noetig)
    "trans+inferred_dom",# Domain aus Routing-Overlap inferred, kein oracle
]


def _lb_pin_sets(core_analysis, domain_names):
    """Baut pin-Mengen fuer alle Policy-Typen (K-unabhaengig).

    Gibt zurueck:
      hot_blk_global  -- liste der globalen Top-N hot blocks
      dom_hot         -- {dname: [block_id, ...]} fuer oracle domain-pin
      hot_core_blk    -- liste der Top-1-Core-Bloecke
      dom_hot_fsets   -- list[frozenset] indiziert nach domain-index, fuer domain-inference
    """
    hot_blk_global = [b for b, _ in core_analysis["hot_blocks"][:LB_N_PIN]]
    dom_hot        = {d: [b for b, _ in core_analysis["dom_block"]
                          .get(d, Counter()).most_common(LB_N_PIN_DOM)]
                      for d in domain_names}
    top_core_fset  = (core_analysis["hot_cores"][0][0]
                      if core_analysis["hot_cores"] else frozenset())
    hot_core_blk   = sorted(top_core_fset)[:LB_N_PIN]
    # frozensets pro Domain-Index fuer schnelle Overlap-Berechnung
    dom_hot_fsets  = [frozenset(dom_hot.get(d, [])) for d in domain_names]
    return hot_blk_global, dom_hot, hot_core_blk, dom_hot_fsets


def _lb_pol_cfg(pol, hot_blk_global, dom_hot, hot_core_blk):
    """Policy-Config-Dict fuer eine Policy."""
    cfg = {}
    if pol == "transition_pf":
        cfg["do_prefetch"] = True
    elif pol == "hot_block_pin":
        cfg["pinned_global"] = hot_blk_global
    elif pol == "domain_hot_pin":
        cfg["pinned_domain"] = dom_hot          # oracle: echte Domain-Labels
    elif pol == "hot_core_pin":
        cfg["pinned_global"] = hot_core_blk
    elif pol == "trans+domain_hot":
        cfg["pinned_domain"] = dom_hot          # oracle: echte Domain-Labels
        cfg["do_prefetch"]   = True
    elif pol == "trans+hot_core":
        cfg["pinned_global"] = hot_core_blk
        cfg["do_prefetch"]   = True
    elif pol == "trans+inferred_dom":
        cfg["pinned_domain"] = dom_hot          # Profile aus Calib (kein oracle-Label zur Laufzeit)
        cfg["do_prefetch"]   = True
        cfg["infer_domain"]  = True             # Domain per Routing-Overlap inferieren
    return cfg


def _infer_domain(core_fset, dom_hot_fsets):
    """Inferiert Domain-Index aus Block-Overlap mit kalibrierten Domain-Profilen."""
    best_d, best_ov = 0, -1
    for d_idx, hot_fset in enumerate(dom_hot_fsets):
        ov = len(core_fset & hot_fset)
        if ov > best_ov:
            best_ov, best_d = ov, d_idx
    return best_d


def simulate_unified(traces, r_stops_batches, doms, domain_names,
                     K, pol_cfg, next_core_map, dom_hot_fsets=None):
    """
    Einheitliche per-Sequenz-Simulation fuer alle 8 Policies.
    Trennt Demand-Misses und Prefetch (load/hit/waste) klar auf.

    pol_cfg keys:
      pinned_global  list[int]        — immer resident (global)
      pinned_domain  dict[str,list]   — domain-spezifisch resident (oracle label)
      do_prefetch    bool             — Transition-Prefetch aktiv
      infer_domain   bool             — Domain per Routing-Overlap inferieren (kein oracle)

    dom_hot_fsets: list[frozenset] indexed by domain-index, fuer infer_domain.
    """
    pinned_g      = set(pol_cfg.get("pinned_global", []))
    pinned_dom    = pol_cfg.get("pinned_domain", {})
    do_pf         = pol_cfg.get("do_prefetch", False)
    do_inf_dom    = pol_cfg.get("infer_domain", False)

    demand_miss   = 0
    demand_access = 0
    pf_load = pf_hit = pf_waste = 0
    dom_dmiss   = Counter()
    dom_daccess = Counter()

    for trace, r_stop, dom_arr in zip(traces, r_stops_batches, doms):
        R, B, T, kk = trace.shape
        for b in range(B):
            # --- Oracle domain (fuer Stats + oracle-pin-policies) ---
            d_idx_oracle = int(dom_arr[b])
            dname_oracle = (domain_names[d_idx_oracle]
                            if 0 <= d_idx_oracle < len(domain_names) else None)

            # --- Pinned-Menge aufbauen ---
            pinned = set(pinned_g)
            if do_inf_dom and dom_hot_fsets:
                # Domain aus erstem Core der Sequenz inferieren
                first_core = frozenset(int(x) for x in np.unique(trace[0, b, 0]))
                inf_d_idx  = _infer_domain(first_core, dom_hot_fsets)
                inf_dname  = (domain_names[inf_d_idx]
                              if 0 <= inf_d_idx < len(domain_names) else None)
                if inf_dname:
                    pinned.update(pinned_dom.get(inf_dname, []))
            elif dname_oracle:
                pinned.update(pinned_dom.get(dname_oracle, []))

            k_lru = max(0, K - len(pinned))
            cache      = OrderedDict()
            pf_pending = set()

            for t in range(T):
                core      = [int(x) for x in np.unique(trace[0, b, t])]
                core_fset = frozenset(core)

                # -- Demand ---------------------------------------------------
                for blk in core:
                    demand_access += 1
                    if dname_oracle:
                        dom_daccess[dname_oracle] += 1

                    if blk in pinned:
                        if blk in pf_pending:
                            pf_hit += 1; pf_pending.discard(blk)
                        continue

                    if blk in cache:
                        cache.move_to_end(blk)
                        if blk in pf_pending:
                            pf_hit += 1; pf_pending.discard(blk)
                    else:
                        demand_miss += 1
                        if dname_oracle:
                            dom_dmiss[dname_oracle] += 1
                        cache[blk] = True
                        cache.move_to_end(blk)
                        if len(cache) > k_lru:
                            ev, _ = cache.popitem(last=False)
                            if ev in pf_pending:
                                pf_waste += 1; pf_pending.discard(ev)

                # -- Prefetch -------------------------------------------------
                if do_pf:
                    predicted = next_core_map.get(core_fset)
                    if predicted:
                        for blk in predicted:
                            if blk in pinned or blk in cache:
                                if blk in cache:
                                    cache.move_to_end(blk)
                                continue
                            pf_load += 1
                            cache[blk] = True
                            cache.move_to_end(blk)
                            pf_pending.add(blk)
                            if len(cache) > k_lru:
                                ev, _ = cache.popitem(last=False)
                                if ev in pf_pending:
                                    pf_waste += 1; pf_pending.discard(ev)

    n_tokens  = sum(rs.size for rs in r_stops_batches)
    pf_useful = pf_hit + pf_waste
    return {
        "n_tokens":     n_tokens,
        "demand_miss":  demand_miss,
        "demand_access": demand_access,
        "demand_mpt":   round(demand_miss / max(n_tokens, 1), 3),
        "hit_rate":     round(1 - demand_miss / max(demand_access, 1), 4),
        "pf_load":      pf_load,
        "pf_hit":       pf_hit,
        "pf_waste":     pf_waste,
        "pf_accuracy":  round(pf_hit / max(pf_useful, 1), 3),
        "dom_miss":     dict(dom_dmiss),
        "dom_access":   dict(dom_daccess),
    }


def run_leiterbahn_sweep(traces_test, r_stops_test, doms_test, domain_names,
                         core_analysis, block_bytes, bw_bps, Ks=None):
    """
    K-Sweep x Policy-Sweep.

    Trennung Calibration / Test:
      core_analysis  -- aus Calib-Traces (seen) gelernt (hot_blocks, transitions, profiles)
      traces_test    -- Held-out Traces fuer die eigentliche Simulation

    Gibt {K: {policy: full_metrics}} zurueck.
    """
    if Ks is None:
        Ks = LB_KS
    next_core_map                              = core_analysis["next_core_map"]
    hot_blk_g, dom_hot, hcore_blk, dom_hot_fs = _lb_pin_sets(core_analysis, domain_names)
    n_tokens = sum(rs.size for rs in r_stops_test)
    results  = {}

    for K in Ks:
        results[K] = {}
        for pol in LEITERBAHN_POLICIES:
            pol_cfg = _lb_pol_cfg(pol, hot_blk_g, dom_hot, hcore_blk)
            sim     = simulate_unified(
                traces_test, r_stops_test, doms_test, domain_names,
                K, pol_cfg, next_core_map,
                dom_hot_fsets=dom_hot_fs)
            b_dem = sim["demand_mpt"]  * block_bytes
            b_pf  = sim["pf_load"] / max(n_tokens, 1) * block_bytes
            results[K][pol] = {
                **sim,
                "bytes_demand_fp16":   int(round(b_dem)),
                "bytes_demand_int8":   int(round(b_dem / 2)),
                "bytes_pf_fp16":       int(round(b_pf)),
                "bytes_total_fp16":    int(round(b_dem + b_pf)),
                "transfer_us_demand":  round(b_dem / bw_bps * 1e6, 2),
            }
    return results


ORACLE_POLICIES = {"domain_hot_pin", "trans+domain_hot"}  # brauchen echte Domain-Labels

def _pol_label(pol):
    sfx = " (oracle)" if pol in ORACLE_POLICIES else ""
    return (pol + sfx)[:28]


def print_leiterbahn_matrix(lb, domain_names, block_bytes, k=8, K_dom=32):
    Ks = sorted(lb.keys())
    pf_pols = [p for p in LEITERBAHN_POLICIES if "trans" in p or p == "transition_pf"]

    # -- Demand bytes/token Tabelle -------------------------------------------
    print(f"\n  Leiterbahn-Sweep '{LB_TARGET_CFG}' -- Demand bytes/token [fp16 KB]")
    print(f"  (Policies gelernt auf Calib/Seen, getestet auf Held-out)")
    hdr = f"  {'Policy':<30}" + "".join(f"  K={K:<5}" for K in Ks)
    print(hdr); print(f"  {'-'*(len(hdr)-2)}")
    for pol in LEITERBAHN_POLICIES:
        row = f"  {_pol_label(pol):<30}"
        for K in Ks:
            bk = lb[K][pol]["bytes_demand_fp16"] / 1024
            row += f"  {bk:>6.1f} "
        print(row)

    # -- Total bytes inkl. Prefetch-Waste (nur PF-Policies) -------------------
    print(f"\n  Total bytes/token inkl. Prefetch-Waste [fp16 KB]  (Demand + PF-Waste)")
    hdr2 = f"  {'Policy':<30}" + "".join(f"  K={K:<5}" for K in Ks)
    print(hdr2); print(f"  {'-'*(len(hdr2)-2)}")
    for pol in pf_pols:
        row = f"  {_pol_label(pol):<30}"
        for K in Ks:
            bt = lb[K][pol]["bytes_total_fp16"] / 1024
            row += f"  {bt:>6.1f} "
        print(row)

    # -- Prefetch-Accuracy -------------------------------------------------------
    print(f"\n  Prefetch-Accuracy [pf_hit / (pf_hit + pf_waste)]")
    hdr3 = f"  {'Policy':<30}" + "".join(f"  K={K:<5}" for K in Ks)
    print(hdr3); print(f"  {'-'*(len(hdr3)-2)}")
    for pol in pf_pols:
        row = f"  {_pol_label(pol):<30}"
        for K in Ks:
            acc = lb[K][pol]["pf_accuracy"]
            row += f"  {acc:>6.3f} "
        print(row)

    # -- Beste Policy pro K (nur non-oracle) -------------------------------------
    non_oracle = [p for p in LEITERBAHN_POLICIES if p not in ORACLE_POLICIES]
    print(f"\n  Beste Non-Oracle-Policy (min Demand-Bytes) pro K:")
    for K in Ks:
        best = min(non_oracle, key=lambda p: lb[K][p]["bytes_demand_fp16"])
        bk   = lb[K][best]["bytes_demand_fp16"] / 1024
        lru  = lb[K]["lru"]["bytes_demand_fp16"] / 1024
        gain = (lru - bk) / lru * 100
        print(f"    K={K:<3}: {best:<26}  {bk:.1f} KB  ({gain:+.1f}% vs LRU)")

    # -- Vergleich oracle vs. inferred bei K=16 ---------------------------------
    k16 = 16
    if k16 in lb:
        print(f"\n  Oracle vs. Inferred bei K={k16}:")
        pairs = [("trans+domain_hot", "trans+inferred_dom"),
                 ("domain_hot_pin",   None)]
        for oracle_p, inferred_p in pairs:
            if oracle_p not in lb[k16]:
                continue
            b_or = lb[k16][oracle_p]["bytes_demand_fp16"] / 1024
            line = f"    {oracle_p:<26} (oracle)   {b_or:.1f} KB"
            if inferred_p and inferred_p in lb[k16]:
                b_inf = lb[k16][inferred_p]["bytes_demand_fp16"] / 1024
                gap   = b_inf - b_or
                line += f"  vs  {inferred_p:<26} {b_inf:.1f} KB  (gap {gap:+.1f} KB)"
            print(line)

    # -- Per-Domain Demand bytes/token bei K=K_dom ----------------------------------
    print(f"\n  Per-Domain Demand bytes/token [fp16 KB] bei K={K_dom}:")
    hdr4 = f"  {'Policy':<30}" + "".join(f"  {d:<9}" for d in domain_names)
    print(hdr4); print(f"  {'-'*(len(hdr4)-2)}")
    for pol in LEITERBAHN_POLICIES:
        sim = lb[K_dom][pol]
        row = f"  {_pol_label(pol):<30}"
        for d in domain_names:
            d_miss   = sim["dom_miss"].get(d, 0)
            d_access = sim["dom_access"].get(d, 1)
            d_mpt    = d_miss * k / max(d_access, 1)
            d_bpt    = d_mpt * block_bytes / 1024
            row += f"  {d_bpt:>9.1f}"
        print(row)


# ────────────────────────────────────────────────────────────────────────────────
# Haupt-Seed-Lauf
# ────────────────────────────────────────────────────────────────────────────────

def run_seed(seed, ck_path, data, data_heldout, stopping_cfgs, Ks,
             n_batches, bs, seq_len, device, bw_gbs):
    print(f"\n{'='*64}\n=== Seed {seed} ===\n{'='*64}", flush=True)

    model, arch, step = model_io.load_checkpoint(ck_path, data.vocab_size, device)
    n_blocks     = arch["n_blocks"]
    k            = arch["k"]
    R_max        = arch["R"]
    d            = arch.get("d_model", 256)
    block_params = sum(p.numel()
                       for p in model_io.blocks_of(model, arch)[0].parameters())
    block_bytes  = block_params * 2           # fp16
    bw_bps       = bw_gbs * 1e9
    domains      = data.domains

    print(f"  {model_io.label(arch, step)}  n_blocks={n_blocks}  k={k}  R={R_max}",
          flush=True)
    print(f"  block_params={block_params:,}  block_bytes_fp16={block_bytes/1024:.1f} KB",
          flush=True)

    # ── Sammlung ───────────────────────────────────────────────────────────────
    print("  Sammle Traces + r_stops (1 Pass) ...", flush=True)
    traces, r_stops_by_cfg, doms = collect(
        model, data, stopping_cfgs, n_batches, bs, seq_len, device)

    n_tokens = sum(rs.size for rs in next(iter(r_stops_by_cfg.values())))

    # ── Core-Analyse ──────────────────────────────────────────────────────────
    print("  Analysiere Cores ...", flush=True)
    ca = analyze_cores(traces, doms, domains)

    # Gepinnte Bloecke: die N_PIN_BLOCKS haeufigsten Einzel-Bloecke
    pinned_blocks = [b for b, _ in ca["hot_blocks"][:N_PIN_BLOCKS]]
    next_core_map = ca["next_core_map"]

    # ── Pro Stopping-Config ────────────────────────────────────────────────────
    cfg_results = {}
    for cfg in stopping_cfgs:
        lbl       = cfg["label"]
        r_batches = r_stops_by_cfg[lbl]

        # R-Statistiken
        all_rs = np.concatenate([rs.ravel() for rs in r_batches])
        mean_R = float(np.mean(all_rs))
        std_R  = float(np.std(all_rs))
        R_hist = {int(r): int((all_rs == r).sum())
                  for r in range(1, R_max + 1) if (all_rs == r).sum() > 0}

        # Arithmetische Intensitaet (FLOPs pro Token, approx)
        # ~8d² pro Block-Applikation (Attn + FFN grob)
        flops_per_token = float(mean_R * k * 8 * d * d)

        # Streams (auf r_stop getrimmt)
        streams = make_streams(traces, r_batches)

        # Per-Domain mean_R
        dom_mean_R = {}
        d_r_sum = Counter(); d_r_n = Counter()
        for r_batch, dom in zip(r_batches, doms):
            B, T = r_batch.shape
            for b in range(B):
                d_idx = int(dom[b])
                dname = domains[d_idx] if 0 <= d_idx < len(domains) else None
                if dname:
                    d_r_sum[dname] += float(r_batch[b].mean())
                    d_r_n[dname]   += 1
        for dname in domains:
            if d_r_n[dname] > 0:
                dom_mean_R[dname] = round(d_r_sum[dname] / d_r_n[dname], 3)

        # Cache-Simulationen
        cache_sim = run_cache_sims(
            streams, traces, r_batches, Ks,
            block_bytes, pinned_blocks, next_core_map, bw_bps,
            n_tokens, flops_per_token)

        k16_lru = cache_sim["lru"]["per_K"].get(16, {})
        print(f"  {lbl:<25} mean_R={mean_R:.2f}  saved={1-mean_R/R_max:.1%}"
              f"  miss@16(LRU)={k16_lru.get('miss_at_K',0):.3f}"
              f"  by/tok16={k16_lru.get('bytes_per_token_fp16',0)/1024:.0f}KB",
              flush=True)

        cfg_results[lbl] = {
            "mean_R":         round(mean_R, 3),
            "std_R":          round(std_R, 3),
            "compute_saved":  round(1.0 - mean_R / R_max, 3),
            "R_histogram":    R_hist,
            "domain_mean_R":  dom_mean_R,
            "flops_per_token_approx": int(flops_per_token),
            "cache_sim":      cache_sim,
        }

    # ── Leiterbahn-Sweep: Calib = Seen, Test = Held-out ──────────────────────
    lb_r_stops_calib = r_stops_by_cfg.get(LB_TARGET_CFG)
    if lb_r_stops_calib is not None and data_heldout is not None:
        lb_cfg = [c for c in stopping_cfgs if c["label"] == LB_TARGET_CFG]
        print(f"  Sammle Held-out Traces (Leiterbahn-Test) ...", flush=True)
        traces_ho, r_stops_ho_by_cfg, doms_ho = collect(
            model, data_heldout, lb_cfg, n_batches, bs, seq_len, device)
        r_stops_ho = r_stops_ho_by_cfg.get(LB_TARGET_CFG)
        if r_stops_ho is not None:
            print(f"  Leiterbahn-Sweep ({LB_TARGET_CFG}, K={LB_KS}) ...", flush=True)
            lb_results = run_leiterbahn_sweep(
                traces_ho, r_stops_ho, doms_ho, domains, ca, block_bytes, bw_bps)
        else:
            lb_results = {}
    else:
        lb_results = {}

    return {
        "seed":             seed,
        "n_blocks":         n_blocks,
        "k":                k,
        "R_max":            R_max,
        "block_params":     int(block_params),
        "block_bytes_fp16": int(block_bytes),
        "bandwidth_gbs":    bw_gbs,
        "n_tokens":         int(n_tokens),
        "core_analysis": {
            "n_unique_cores":          ca["n_unique_cores"],
            "hot_blocks_top20":        [(b, c) for b, c in ca["hot_blocks"][:20]],
            "pinned_blocks":           pinned_blocks,
            "hot_core_count":          len(ca["hot_cores"]),
            "top_transitions":         ca["top_transitions"][:10],
            "code_other_block_jaccard": ca["code_other_block_jaccard"],
            "domain_hot_blocks": {
                d: [(b, c) for b, c in ca["dom_block"].get(d, Counter()).most_common(8)]
                for d in domains
            },
        },
        "stopping_configs": cfg_results,
        "leiterbahn":       lb_results,
        "_ca_internal":     ca,    # fuer print_leiterbahn_matrix, nicht im JSON
        "_block_bytes":     block_bytes,
        "_k":               k,
        "_domains":         domains,
    }


# ────────────────────────────────────────────────────────────────────────────────
# Output-Tabellen
# ────────────────────────────────────────────────────────────────────────────────

def _sep(n=96): return "-" * n


def print_stopping_table(result, K=16):
    """Stopping-Config Vergleich (K=16, LRU)."""
    print(f"\n{'Stopping-Config Vergleich (K=16, LRU)':^96}")
    print(_sep())
    print(f"  {'Label':<25} {'mean_R':>7} {'saved':>7} {'miss@K':>8}"
          f" {'by/tok16KB':>11} {'by/tok8KB':>10} {'arith_int':>10}")
    print(_sep())
    for lbl, r in result["stopping_configs"].items():
        lru_k = r["cache_sim"]["lru"]["per_K"].get(K, {})
        ai    = lru_k.get("arith_intensity") or 0.0
        print(f"  {lbl:<25} {r['mean_R']:>7.2f} {r['compute_saved']:>7.1%}"
              f" {lru_k.get('miss_at_K',0):>8.3f}"
              f" {lru_k.get('bytes_per_token_fp16',0)/1024:>11.1f}"
              f" {lru_k.get('bytes_per_token_int8',0)/1024:>10.1f}"
              f" {ai:>10.0f}")
    print(_sep())


def print_policy_table(result, target="kl_minR3_t0.005", K=16):
    """Cache-Policy Vergleich fuer Best Candidate (K=16)."""
    cfg = result["stopping_configs"].get(target)
    if not cfg:
        return
    print(f"\n  Policy-Vergleich '{target}' (K={K}):")
    print(f"  {'Policy':<22} {'miss@K':>8} {'by/tok16KB':>11} {'by/tok8KB':>10} {'arith_int':>10}")
    print(f"  {'-'*65}")
    for pol, pdata in cfg["cache_sim"].items():
        pk = pdata["per_K"].get(K, {})
        ai = pk.get("arith_intensity") or 0.0
        print(f"  {pol:<22} {pk.get('miss_at_K',0):>8.3f}"
              f" {pk.get('bytes_per_token_fp16',0)/1024:>11.1f}"
              f" {pk.get('bytes_per_token_int8',0)/1024:>10.1f}"
              f" {ai:>10.0f}")


def print_domain_table(result, target="kl_minR3_t0.005"):
    """Per-Domain mean_R fuer Best Candidate."""
    cfg = result["stopping_configs"].get(target)
    if not cfg:
        return
    print(f"\n  Per-Domain mean_R '{target}':")
    for d, mr in cfg["domain_mean_R"].items():
        print(f"    {d:<12} {mr:.3f}")


def print_core_summary(result):
    """Zusammenfassung der Core-Analyse."""
    ca = result["core_analysis"]
    print(f"\n  Core-Analyse:")
    print(f"    unique_cores   : {ca['n_unique_cores']:,}")
    print(f"    code/other-jac : {ca['code_other_block_jaccard']:.3f}  "
          f"(0=disjunkt, 1=identisch)")
    print(f"    pinned_blocks  : {ca['pinned_blocks']}")
    print(f"    hot_blocks_top5: "
          f"{[(b, c) for b, c in ca['hot_blocks_top20'][:5]]}")
    dom_hb = ca.get("domain_hot_blocks", {})
    if dom_hb:
        print(f"    domain hot blocks (top-3 pro Domäne):")
        for d, blist in dom_hb.items():
            print(f"      {d:<12}: {[b for b, _ in blist[:3]]}")


# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def _jsonify(obj):
    """Rekursive JSON-Serialisierung (frozenset -> sorted list, numpy -> Python)."""
    if isinstance(obj, dict):
        return {(str(sorted(k)) if isinstance(k, frozenset) else str(k)): _jsonify(v)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(i) for i in obj]
    if isinstance(obj, frozenset):
        return sorted(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",        default="cuda")
    ap.add_argument("--seeds",         nargs="+", type=int, default=[0, 1])
    ap.add_argument("--n_batches",     type=int, default=40)
    ap.add_argument("--bs",            type=int, default=16)
    ap.add_argument("--seq_len",       type=int, default=128)
    ap.add_argument("--bandwidth_gbs", type=float, default=BANDWIDTH_GBS)
    a = ap.parse_args()

    data = HeteroMiniData()
    data_heldout = (HeteroMiniData(out_dir=HELDOUT_DATA_ROOT)
                    if os.path.isdir(HELDOUT_DATA_ROOT) else None)
    if data_heldout is None:
        print("[warn] Kein Held-out-Datensatz gefunden — Leiterbahn laeuft auf Seen-Daten")
    bw   = a.bandwidth_gbs

    all_results = {}
    for seed in a.seeds:
        ck_rel = CKPTS.get(seed)
        if not ck_rel:
            print(f"[skip] Kein Checkpoint fuer seed={seed}"); continue
        ck = os.path.join(ROOT, ck_rel)
        if not os.path.exists(ck):
            print(f"[skip] seed={seed}: {ck} nicht gefunden"); continue

        result = run_seed(seed, ck, data, data_heldout, STOPPING_CONFIGS, CACHE_KS,
                          a.n_batches, a.bs, a.seq_len, a.device, bw)
        all_results[seed] = result

        print_stopping_table(result)
        print_policy_table(result)
        print_domain_table(result)
        print_core_summary(result)

        if result.get("leiterbahn"):
            print_leiterbahn_matrix(
                result["leiterbahn"],
                result["_domains"],
                result["_block_bytes"],
                k=result["_k"])

    # JSON ohne interne Felder
    def _clean(r):
        return {k: v for k, v in r.items() if not k.startswith("_")}

    seed_tag = "_".join(str(s) for s in sorted(all_results.keys()))
    out_path = os.path.join(RESULTS, f"offload_trace_adaptive_s{seed_tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_jsonify({s: _clean(r) for s, r in all_results.items()}),
                  f, indent=2, ensure_ascii=False)
    print(f"\nGespeichert: {out_path}")


if __name__ == "__main__":
    main()
