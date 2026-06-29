"""Vollständige Eval-Pipeline für srcore_b64_k8_R6 @10k.

Läuft nacheinander:
  1. gain_su       — seen vs. unknown gain (anytime-Kurve)
  2. adaptive_stopping — offline KL/top1/entropy stopping simulation
  3. routing_analysis  — unique_cores, gini, domain_jaccard
  4. offload_leiterbahn — trace-getriebener Cache-Sweep K=[16,24,32,48,64]

Checkpoint: results/hm_srcore_b64_k8_R6_s0_10k.pt
Held-out:   data/heteromini_v1_heldout

Nutzung:
  python scripts/eval_b64.py --device cuda
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import Counter, OrderedDict
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rblm.heteromini import HeteroMiniData, DATA_ROOT
from rblm import model_io

RESULTS      = os.path.join(ROOT, "results")
CKPT_PATH    = os.path.join(RESULTS, "hm_srcore_b64_k8_R6_s0_10k.pt")
HELDOUT_ROOT = DATA_ROOT + "_heldout"

# K-Sweep: b64 Cache-Regime
#   K=16 → 25% Bank  (b64-Offloading-Regime)
#   K=32 → 50% Bank  (= b32 K=16 Äquivalent)
#   K=48 → 75% Bank
#   K=64 → 100% Bank (b64-Saturation)
LB_KS = [16, 24, 32, 48, 64]

STOPPING_CONFIGS = [
    {"label": "fixed_R2",        "kind": "fixed",   "fixed_R": 2, "min_R": 2},
    {"label": "fixed_R3",        "kind": "fixed",   "fixed_R": 3, "min_R": 3},
    {"label": "fixed_R4",        "kind": "fixed",   "fixed_R": 4, "min_R": 4},
    {"label": "fixed_R6",        "kind": "fixed",   "fixed_R": 6, "min_R": 6},
    {"label": "kl_minR3_t0.005", "kind": "kl_div",  "theta": 0.005, "min_R": 3},
]
LB_TARGET_CFG = "kl_minR3_t0.005"

LB_N_PIN     = 8
LB_N_PIN_DOM = 6
LEITERBAHN_POLICIES = [
    "lru", "transition_pf", "hot_block_pin", "domain_hot_pin",
    "hot_core_pin", "trans+domain_hot", "trans+hot_core", "trans+inferred_dom",
]
ORACLE_POLICIES = {"domain_hot_pin", "trans+domain_hot"}

BANDWIDTH_GBS = 16.0


# ─── Adaptive Stopping ────────────────────────────────────────────────────────

def compute_r_stop(logits_stack, cfg, R_max, device="cpu"):
    kind = cfg.get("kind", "fixed")
    B, T = logits_stack.shape[1], logits_stack.shape[2]
    r_stop = torch.full((B, T), R_max, dtype=torch.long, device=device)
    if kind == "fixed":
        r_stop.fill_(cfg["fixed_R"])
        return r_stop
    min_R = cfg.get("min_R", 1)
    stopped = torch.zeros(B, T, dtype=torch.bool, device=device)
    if kind == "top1_stable":
        stable_cnt = torch.zeros(B, T, dtype=torch.long, device=device)
        win = cfg.get("window", 2)
        prev = logits_stack[0].argmax(-1)
        for r in range(1, R_max):
            cur = logits_stack[r].argmax(-1)
            stable_cnt = torch.where(cur == prev, stable_cnt + 1, torch.zeros_like(stable_cnt))
            prev = cur
            if r >= min_R:
                fire = (~stopped) & (stable_cnt >= win)
                r_stop[fire] = r
                stopped |= fire
    elif kind == "entropy_drop":
        theta = cfg.get("theta", 0.1)
        H_prev = -(F.softmax(logits_stack[0], -1) * F.log_softmax(logits_stack[0], -1)).sum(-1)
        for r in range(1, R_max):
            H_cur = -(F.softmax(logits_stack[r], -1) * F.log_softmax(logits_stack[r], -1)).sum(-1)
            if r >= min_R:
                fire = (~stopped) & ((H_prev - H_cur) < theta)
                r_stop[fire] = r
                stopped |= fire
            H_prev = H_cur
    elif kind == "kl_div":
        theta = cfg.get("theta", 0.005)
        p_prev = F.softmax(logits_stack[0], -1)
        for r in range(1, R_max):
            p_cur = F.softmax(logits_stack[r], -1)
            kl = (p_prev * (p_prev.log() - p_cur.log())).sum(-1)
            if r >= min_R:
                fire = (~stopped) & (kl < theta)
                r_stop[fire] = r
                stopped |= fire
            p_prev = p_cur
    return r_stop


# ─── Data Collection ──────────────────────────────────────────────────────────

@torch.no_grad()
def collect(model, data, stopping_cfgs, n_batches, bs, seq_len, device):
    traces, doms = [], []
    r_stops_by_cfg = {c["label"]: [] for c in stopping_cfgs}
    for _ in range(n_batches):
        toks, _, _, dom = data.batch(bs, seq_len, device=device, mode="contiguous")
        logits_all, aux = model(toks)
        R_actual = len(logits_all)
        logits_stack = torch.stack(logits_all, dim=0)
        trace = np.stack([a["topk_idx"].cpu().numpy() for a in aux["iters"]])
        traces.append(trace)
        doms.append(dom.cpu().numpy())
        for cfg in stopping_cfgs:
            rs = compute_r_stop(logits_stack, cfg, R_actual, device=device)
            r_stops_by_cfg[cfg["label"]].append(rs.cpu().numpy())
    return traces, r_stops_by_cfg, doms


# ─── Core Analysis ────────────────────────────────────────────────────────────

def analyze_cores(traces, doms, domain_names):
    block_counter = Counter()
    core_counter  = Counter()
    next_core_raw = {}
    dom_block     = {d: Counter() for d in domain_names}
    prev_cores    = {}

    for trace, dom_arr in zip(traces, doms):
        R, B, T, k = trace.shape
        for b in range(B):
            d_idx = int(dom_arr[b])
            dname = domain_names[d_idx] if 0 <= d_idx < len(domain_names) else None
            prev_core = None
            for t in range(T):
                blks = [int(x) for x in np.unique(trace[0, b, t])]
                cfset = frozenset(blks)
                for bl in blks:
                    block_counter[bl] += 1
                    if dname:
                        dom_block[dname][bl] += 1
                core_counter[cfset] += 1
                if prev_core is not None:
                    if prev_core not in next_core_raw:
                        next_core_raw[prev_core] = Counter()
                    next_core_raw[prev_core][cfset] += 1
                prev_core = cfset

    next_core_map = {c: max(nc, key=nc.get) for c, nc in next_core_raw.items()}
    hot_cores     = core_counter.most_common(50)
    hot_blocks    = block_counter.most_common()

    # code vs. non-code Jaccard
    code_blks  = set(dom_block.get("code", {}).keys())
    other_blks = set()
    for d in domain_names:
        if d != "code":
            other_blks.update(dom_block.get(d, {}).keys())
    jaccard = (len(code_blks & other_blks) / max(1, len(code_blks | other_blks)))

    return {
        "hot_blocks":   hot_blocks,
        "hot_cores":    hot_cores,
        "next_core_map": next_core_map,
        "dom_block":    dom_block,
        "n_unique_cores": len(core_counter),
        "code_other_block_jaccard": round(jaccard, 3),
        "top_transitions": [
            (sorted(k), sorted(v), c)
            for k, (v, c) in sorted(
                [(k, max(nc.items(), key=lambda x: x[1]))
                 for k, nc in next_core_raw.items()],
                key=lambda x: -x[1][1])[:20]
        ],
    }


# ─── Leiterbahn Simulation ────────────────────────────────────────────────────

def _lb_pin_sets(ca, domain_names):
    hot_blk_g  = [b for b, _ in ca["hot_blocks"][:LB_N_PIN]]
    dom_hot    = {d: [b for b, _ in ca["dom_block"].get(d, Counter()).most_common(LB_N_PIN_DOM)]
                  for d in domain_names}
    top_core   = (ca["hot_cores"][0][0] if ca["hot_cores"] else frozenset())
    hot_core_b = sorted(top_core)[:LB_N_PIN]
    dom_hot_fs = [frozenset(dom_hot.get(d, [])) for d in domain_names]
    return hot_blk_g, dom_hot, hot_core_b, dom_hot_fs


def _pol_cfg(pol, hot_blk_g, dom_hot, hot_core_b):
    c = {}
    if pol == "transition_pf":     c["do_prefetch"] = True
    elif pol == "hot_block_pin":   c["pinned_global"] = hot_blk_g
    elif pol == "domain_hot_pin":  c["pinned_domain"] = dom_hot
    elif pol == "hot_core_pin":    c["pinned_global"] = hot_core_b
    elif pol == "trans+domain_hot":
        c["pinned_domain"] = dom_hot; c["do_prefetch"] = True
    elif pol == "trans+hot_core":
        c["pinned_global"] = hot_core_b; c["do_prefetch"] = True
    elif pol == "trans+inferred_dom":
        c["pinned_domain"] = dom_hot; c["do_prefetch"] = True; c["infer_domain"] = True
    return c


def _infer_domain(core_fset, dom_hot_fs):
    best_d, best_ov = 0, -1
    for i, hf in enumerate(dom_hot_fs):
        ov = len(core_fset & hf)
        if ov > best_ov:
            best_ov, best_d = ov, i
    return best_d


def simulate_unified(traces, r_stops, doms, domain_names, K, pol_cfg, next_core_map,
                     dom_hot_fs=None):
    pinned_g   = set(pol_cfg.get("pinned_global", []))
    pinned_dom = pol_cfg.get("pinned_domain", {})
    do_pf      = pol_cfg.get("do_prefetch", False)
    do_inf     = pol_cfg.get("infer_domain", False)

    demand_miss = demand_access = 0
    pf_load = pf_hit = pf_waste = 0
    dom_dmiss = Counter(); dom_daccess = Counter()

    for trace, r_stop, dom_arr in zip(traces, r_stops, doms):
        R, B, T, kk = trace.shape
        for b in range(B):
            d_idx = int(dom_arr[b])
            dname = domain_names[d_idx] if 0 <= d_idx < len(domain_names) else None

            pinned = set(pinned_g)
            if do_inf and dom_hot_fs:
                fc = frozenset(int(x) for x in np.unique(trace[0, b, 0]))
                inf_d = _infer_domain(fc, dom_hot_fs)
                inf_dn = domain_names[inf_d] if 0 <= inf_d < len(domain_names) else None
                if inf_dn:
                    pinned.update(pinned_dom.get(inf_dn, []))
            elif dname:
                pinned.update(pinned_dom.get(dname, []))

            k_lru = max(0, K - len(pinned))
            cache = OrderedDict(); pf_pending = set()

            for t in range(T):
                core = [int(x) for x in np.unique(trace[0, b, t])]
                cfset = frozenset(core)
                for blk in core:
                    demand_access += 1
                    if dname: dom_daccess[dname] += 1
                    if blk in pinned:
                        if blk in pf_pending: pf_hit += 1; pf_pending.discard(blk)
                        continue
                    if blk in cache:
                        cache.move_to_end(blk)
                        if blk in pf_pending: pf_hit += 1; pf_pending.discard(blk)
                    else:
                        demand_miss += 1
                        if dname: dom_dmiss[dname] += 1
                        cache[blk] = True; cache.move_to_end(blk)
                        if len(cache) > k_lru:
                            ev, _ = cache.popitem(last=False)
                            if ev in pf_pending: pf_waste += 1; pf_pending.discard(ev)
                if do_pf:
                    pred = next_core_map.get(cfset)
                    if pred:
                        for blk in pred:
                            if blk in pinned or blk in cache:
                                if blk in cache: cache.move_to_end(blk)
                                continue
                            pf_load += 1; cache[blk] = True; cache.move_to_end(blk)
                            pf_pending.add(blk)
                            if len(cache) > k_lru:
                                ev, _ = cache.popitem(last=False)
                                if ev in pf_pending: pf_waste += 1; pf_pending.discard(ev)

    n_tok = sum(rs.size for rs in r_stops)
    pf_useful = pf_hit + pf_waste
    return {
        "n_tokens": n_tok,
        "demand_miss": demand_miss,
        "demand_mpt": round(demand_miss / max(n_tok, 1), 3),
        "hit_rate": round(1 - demand_miss / max(demand_access, 1), 4),
        "pf_load": pf_load, "pf_hit": pf_hit, "pf_waste": pf_waste,
        "pf_accuracy": round(pf_hit / max(pf_useful, 1), 3),
        "dom_miss": dict(dom_dmiss), "dom_access": dict(dom_daccess),
    }


def run_leiterbahn(traces_test, r_stops_test, doms_test, domain_names,
                   ca, block_bytes, bw_bps, Ks=None):
    if Ks is None:
        Ks = LB_KS
    ncm = ca["next_core_map"]
    hot_blk_g, dom_hot, hot_core_b, dom_hot_fs = _lb_pin_sets(ca, domain_names)
    n_tok = sum(rs.size for rs in r_stops_test)
    out   = {}
    for K in Ks:
        out[K] = {}
        for pol in LEITERBAHN_POLICIES:
            pc  = _pol_cfg(pol, hot_blk_g, dom_hot, hot_core_b)
            sim = simulate_unified(traces_test, r_stops_test, doms_test,
                                   domain_names, K, pc, ncm, dom_hot_fs)
            b_dem = sim["demand_mpt"] * block_bytes
            b_pf  = sim["pf_load"] / max(n_tok, 1) * block_bytes
            out[K][pol] = {**sim,
                "bytes_demand_fp16": int(round(b_dem)),
                "bytes_pf_fp16":     int(round(b_pf)),
                "bytes_total_fp16":  int(round(b_dem + b_pf)),
            }
    return out


# ─── Gain Seen vs. Unknown ────────────────────────────────────────────────────

@torch.no_grad()
def gain_su(model, data_seen, data_heldout, domain_names,
            n_batches, bs, seq_len, device):
    """Anytime-Kurve: gain_seen vs. gain_unknown pro R und Domain."""
    R = model.n_iters

    def _collect_gains(data):
        losses = np.zeros((R, len(domain_names)))
        counts = np.zeros(len(domain_names))
        for _ in range(n_batches):
            toks, tgt, mask, dom = data.batch(bs, seq_len, device=device, mode="contiguous")
            logits_all, _ = model(toks)
            for r, lg in enumerate(logits_all):
                nll = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1),
                                      reduction="none").reshape(toks.shape)
                nll = (nll * mask).sum(-1) / mask.sum(-1).clamp(min=1)
                for d_idx in range(len(domain_names)):
                    mask_d = (dom == d_idx)
                    if mask_d.any():
                        losses[r, d_idx] += nll[mask_d].mean().item()
                        if r == 0:
                            counts[d_idx] += mask_d.sum().item()
        counts = np.maximum(counts, 1)
        return losses / (n_batches)

    l_seen   = _collect_gains(data_seen)
    l_heldout = _collect_gains(data_heldout)

    gain_seen  = l_seen[0]   - l_seen[-1]
    gain_unk   = l_heldout[0] - l_heldout[-1]
    ratio      = gain_unk / np.maximum(gain_seen, 1e-6)

    # anytime-Kurve pro R
    anytime_seen = [float(l_seen[0, d] - l_seen[r, d])
                    for r in range(R) for d in range(len(domain_names))]
    ratio_by_R = []
    for r in range(R):
        gs = l_seen[0] - l_seen[r]
        gu = l_heldout[0] - l_heldout[r]
        ratio_by_R.append({
            "R": r + 1,
            "compute_frac": round((r + 1) / R, 3),
            "ratio_per_domain": {domain_names[d]: round(float(gu[d] / max(gs[d], 1e-6)), 3)
                                  for d in range(len(domain_names))},
        })

    return {
        "gain_seen":   {domain_names[d]: round(float(gain_seen[d]), 4)
                        for d in range(len(domain_names))},
        "gain_unk":    {domain_names[d]: round(float(gain_unk[d]), 4)
                        for d in range(len(domain_names))},
        "ratio":       {domain_names[d]: round(float(ratio[d]), 3)
                        for d in range(len(domain_names))},
        "ratio_by_R":  ratio_by_R,
    }


# ─── Print ────────────────────────────────────────────────────────────────────

def print_gain_table(gsu, domain_names):
    print("\n  gain_su:")
    print("  %-10s  %8s  %8s  %8s" % ("domain", "gain_seen", "gain_unk", "ratio"))
    print("  " + "-" * 42)
    for d in domain_names:
        print("  %-10s  %8.4f  %8.4f  %8.3f" % (
            d, gsu["gain_seen"][d], gsu["gain_unk"][d], gsu["ratio"][d]))
    print("\n  Anytime ratio pro R:")
    print("  %-4s  %-7s  " % ("R", "compute") +
          "  ".join("%-7s" % d[:7] for d in domain_names))
    for row in gsu["ratio_by_R"]:
        vals = "  ".join("%7.3f" % row["ratio_per_domain"].get(d, 0) for d in domain_names)
        print("  R=%-2d  %-7.0f%%  %s" % (row["R"], row["compute_frac"]*100, vals))


def _pol_label(pol):
    sfx = " (oracle)" if pol in ORACLE_POLICIES else ""
    return (pol + sfx)[:30]


def print_leiterbahn(lb, domain_names, block_bytes, k=8):
    Ks = sorted(lb.keys())
    n_blocks = 64

    print("\n  Leiterbahn b64/k8/R6 -- Demand bytes/token [fp16 KB]")
    print("  (Calib: Seen-Traces, Test: Held-out-Traces)")
    print("  Cache-Regime: K/n_blocks  K=16=25%%, K=32=50%%, K=48=75%%, K=64=100%%")
    hdr = "  %-32s" % "Policy" + "".join("  K=%-4d" % K for K in Ks)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for pol in LEITERBAHN_POLICIES:
        row = "  %-32s" % _pol_label(pol)
        for K in Ks:
            bk = lb[K][pol]["bytes_demand_fp16"] / 1024
            row += "  %6.1f" % bk
        print(row)

    pf_pols = [p for p in LEITERBAHN_POLICIES if "trans" in p or p == "transition_pf"]
    print("\n  Total bytes/token inkl. Prefetch-Waste [fp16 KB]")
    hdr2 = "  %-32s" % "Policy" + "".join("  K=%-4d" % K for K in Ks)
    print(hdr2); print("  " + "-" * (len(hdr2) - 2))
    for pol in pf_pols:
        row = "  %-32s" % _pol_label(pol)
        for K in Ks:
            bt = lb[K][pol]["bytes_total_fp16"] / 1024
            row += "  %6.1f" % bt
        print(row)

    print("\n  Prefetch-Accuracy [pf_hit/(pf_hit+pf_waste)]")
    hdr3 = "  %-32s" % "Policy" + "".join("  K=%-4d" % K for K in Ks)
    print(hdr3); print("  " + "-" * (len(hdr3) - 2))
    for pol in pf_pols:
        row = "  %-32s" % _pol_label(pol)
        for K in Ks:
            row += "  %6.3f" % lb[K][pol]["pf_accuracy"]
        print(row)

    non_oracle = [p for p in LEITERBAHN_POLICIES if p not in ORACLE_POLICIES]
    print("\n  Beste Non-Oracle-Policy pro K:")
    for K in Ks:
        best = min(non_oracle, key=lambda p: lb[K][p]["bytes_demand_fp16"])
        bk   = lb[K][best]["bytes_demand_fp16"] / 1024
        lru  = lb[K]["lru"]["bytes_demand_fp16"] / 1024
        pct  = (lru - bk) / lru * 100
        cache_frac = K / n_blocks * 100
        print("    K=%-3d (%3.0f%% Bank): %-26s  %6.1f KB  (%+.1f%% vs LRU)" % (
            K, cache_frac, best, bk, pct))

    # Vergleich oracle vs inferred bei K=32
    k32 = 32
    if k32 in lb:
        print("\n  Oracle vs. Inferred bei K=%d (50%% Bank):" % k32)
        for orac, inf in [("trans+domain_hot", "trans+inferred_dom"),
                          ("trans+hot_core", None)]:
            if orac not in lb[k32]: continue
            b_o = lb[k32][orac]["bytes_demand_fp16"] / 1024
            line = "    %-26s  %6.1f KB" % (orac + (" (oracle)" if orac in ORACLE_POLICIES else ""), b_o)
            if inf and inf in lb[k32]:
                b_i = lb[k32][inf]["bytes_demand_fp16"] / 1024
                line += "  vs  %-26s  %6.1f KB  (gap %+.1f KB)" % (inf, b_i, b_i - b_o)
            print(line)


# ─── Main ─────────────────────────────────────────────────────────────────────

def _jsonify(obj):
    if isinstance(obj, dict):
        return {(str(sorted(k)) if isinstance(k, frozenset) else str(k)): _jsonify(v)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(i) for i in obj]
    if isinstance(obj, frozenset):
        return sorted(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    return obj


def _load_any(ck_path, device):
    """Lädt self-describing (model_io) oder Cont-Snapshot-Format."""
    raw = torch.load(ck_path, map_location=device, weights_only=False)
    if "arch" in raw:
        return model_io.load_checkpoint(ck_path, device=device)
    # Cont-Format: {"model": state_dict, "config": {...}, "step": N}
    cfg  = raw["config"]
    step = int(raw.get("step", 0))
    from experiments.tinystories_exp import make_model
    from rblm.heteromini import HeteroMiniData
    vocab = HeteroMiniData().vocab_size
    model, _ = make_model(vocab, n_blocks=cfg["n_blocks"], k=cfg["k"],
                          R=cfg["R"], device=device, core_mode=cfg["core_mode"])
    model.load_state_dict(raw["model"])
    arch = {"n_blocks": cfg["n_blocks"], "k": cfg["k"], "R": cfg["R"],
            "core_mode": cfg["core_mode"], "variant": cfg.get("variant", "sparse")}
    return model, arch, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",     default="cuda")
    ap.add_argument("--n_batches",  type=int, default=40)
    ap.add_argument("--bs",         type=int, default=16)
    ap.add_argument("--seq_len",    type=int, default=128)
    ap.add_argument("--bw_gbs",     type=float, default=BANDWIDTH_GBS)
    ap.add_argument("--ckpt",       default=CKPT_PATH,
                    help="Checkpoint-Pfad (self-describing oder Cont-Format)")
    ap.add_argument("--out",        default=None,
                    help="Ausgabe-JSON (default: auto aus Checkpoint-Name)")
    a = ap.parse_args()

    ck_path = a.ckpt
    print(f"=== b64/k8/R6 Eval: {os.path.basename(ck_path)} ===", flush=True)
    model, arch, step = _load_any(ck_path, device=a.device)
    label = f"srcore_b64_k8_R6@{step}"
    n_blocks = arch["n_blocks"]
    k        = arch["k"]
    block_params = sum(p.numel() for p in model.bank.blocks[0].parameters())
    block_bytes  = block_params * 2
    bw_bps       = a.bw_gbs * 1e9
    print(f"  {label}  n_blocks={n_blocks}  k={k}  R={arch['R']}", flush=True)
    print(f"  block_bytes_fp16={block_bytes/1024:.1f} KB", flush=True)

    data        = HeteroMiniData()
    data_heldout = HeteroMiniData(out_dir=HELDOUT_ROOT)
    domain_names = data.domains

    # 1. Gain Seen vs. Unknown
    print("\n--- 1. Gain Seen vs. Unknown ---", flush=True)
    model.eval()
    gsu = gain_su(model, data, data_heldout, domain_names,
                  a.n_batches, a.bs, a.seq_len, a.device)
    print_gain_table(gsu, domain_names)

    # 2. Adaptive Stopping + Routing + Leiterbahn (ein Pass auf Seen)
    print("\n--- 2. Traces sammeln (Seen, Calib) ---", flush=True)
    traces_seen, r_stops_seen_by_cfg, doms_seen = collect(
        model, data, STOPPING_CONFIGS, a.n_batches, a.bs, a.seq_len, a.device)

    # Adaptive Stopping Stats
    print("\n--- 3. Adaptive Stopping ---", flush=True)
    R_max = arch["R"]
    for cfg in STOPPING_CONFIGS:
        lbl  = cfg["label"]
        rs   = r_stops_seen_by_cfg[lbl]
        all_r = np.concatenate([r.ravel() for r in rs])
        mean_R = float(np.mean(all_r))
        saved  = 1 - mean_R / R_max
        print(f"  {lbl:<25} mean_R={mean_R:.2f}  saved={saved:.1%}")

    # Routing Analysis
    print("\n--- 4. Routing-Analyse ---", flush=True)
    ca = analyze_cores(traces_seen, doms_seen, domain_names)
    print(f"  unique_cores  : {ca['n_unique_cores']:,}")
    print(f"  code/other-jac: {ca['code_other_block_jaccard']:.3f}")
    print(f"  hot_blocks_top5: {[(b, c) for b, c in ca['hot_blocks'][:5]]}")
    for d in domain_names:
        top3 = [b for b, _ in ca["dom_block"].get(d, Counter()).most_common(3)]
        print(f"    {d:<12}: {top3}")

    # 3. Held-out Traces für Leiterbahn-Test
    print("\n--- 5. Traces sammeln (Held-out, Test) ---", flush=True)
    lb_cfg = [c for c in STOPPING_CONFIGS if c["label"] == LB_TARGET_CFG]
    traces_ho, r_stops_ho_by_cfg, doms_ho = collect(
        model, data_heldout, lb_cfg, a.n_batches, a.bs, a.seq_len, a.device)
    r_stops_ho = r_stops_ho_by_cfg[LB_TARGET_CFG]

    # Leiterbahn Sweep
    print(f"\n--- 6. Leiterbahn-Sweep K={LB_KS} ---", flush=True)
    lb = run_leiterbahn(traces_ho, r_stops_ho, doms_ho, domain_names,
                        ca, block_bytes, bw_bps, Ks=LB_KS)
    print_leiterbahn(lb, domain_names, block_bytes, k=k)

    # Output
    result = {
        "model": label, "n_blocks": n_blocks, "k": k, "R": R_max,
        "block_bytes_fp16": int(block_bytes),
        "gain_su": gsu,
        "adaptive_stopping": {
            cfg["label"]: {
                "mean_R": float(np.mean(np.concatenate([r.ravel()
                           for r in r_stops_seen_by_cfg[cfg["label"]]]))),
                "compute_saved": round(1 - float(np.mean(np.concatenate(
                    [r.ravel() for r in r_stops_seen_by_cfg[cfg["label"]]]))) / R_max, 3),
            }
            for cfg in STOPPING_CONFIGS
        },
        "routing": {
            "n_unique_cores": ca["n_unique_cores"],
            "code_other_jaccard": ca["code_other_block_jaccard"],
            "hot_blocks_top20": [(b, c) for b, c in ca["hot_blocks"][:20]],
        },
        "leiterbahn": {
            str(K): {p: {kk: v for kk, v in vv.items()
                          if kk not in ("dom_miss", "dom_access")}
                     for p, vv in pol_dict.items()}
            for K, pol_dict in lb.items()
        },
    }
    base     = os.path.splitext(os.path.basename(ck_path))[0]
    out_path = a.out or os.path.join(RESULTS, f"eval_{base}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(result), f, indent=2, ensure_ascii=False)
    print(f"\nGespeichert: {out_path}")


if __name__ == "__main__":
    main()
