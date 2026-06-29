"""Vergleichs-Eval fuer zwei Checkpoints — fokussiert auf Routing-Diagnose.

Neue Metriken gegenueber eval_b64.py:
  soft_overlap_raw   -- Jaccard auf voller Router-Verteilung (eval-Mode, kein Noise)
  soft_overlap_sharp -- Sharpened Jaccard alpha=2 (eval-Mode)
  hard_overlap_eval  -- Hard Top-k Jaccard (eval-Mode)
  router_entropy     -- Shannon-Entropie der Router-Verteilung
  topk_margin        -- P_k - P_{k+1} (Wahrsch.-Luecke letzter-sel/erster-excl)
  unique_cores       -- Anzahl einzigartiger Top-k-Kombinationen
  dead_blocks        -- Blocks die in keiner Top-k-Auswahl vorkamen
  reuseP50/P90/P99   -- Block-Reuse-Distanz-Perzentile
  K=[8,16,24,32,48]  -- Leiterbahn-Sweep (K=8 = minimales Cache-Fenster)

Nutzung:
  python scripts/eval_compare.py --device cuda
  python scripts/eval_compare.py --ckpt_a results/A.pt --ckpt_b results/B.pt
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
from rblm.reg_losses import soft_full_jaccard_loss, soft_sharp_jaccard_loss, router_entropy

RESULTS      = os.path.join(ROOT, "results")
HELDOUT_ROOT = DATA_ROOT + "_heldout"

DEFAULT_A = os.path.join(RESULTS, "hm_cont_hm_srcore_b64_k8_R6_ctrl_17k_s0.pt")
DEFAULT_B = os.path.join(RESULTS, "hm_cont_hm_srcore_b64_k8_R6_coreloc_softfull_r1_lam001_s0.pt")

LB_KS        = [8, 16, 24, 32, 48]
SHARP_ALPHA  = 2.0
N_BLOCKS     = 64

LEITERBAHN_POLICIES = [
    "lru", "transition_pf", "hot_block_pin",
    "hot_core_pin", "trans+hot_core",
]
STOPPING_CFG = {"label": "kl_minR3_t0.005", "kind": "kl_div", "theta": 0.005, "min_R": 3}


# ─── Checkpoint laden ────────────────────────────────────────────────────────

def _load_any(ck_path, device):
    raw = torch.load(ck_path, map_location=device, weights_only=False)
    if "arch" in raw:
        return model_io.load_checkpoint(ck_path, device=device)
    cfg  = raw["config"]
    step = int(raw.get("step", 0))
    from experiments.tinystories_exp import make_model
    vocab = HeteroMiniData().vocab_size
    model, _ = make_model(vocab, n_blocks=cfg["n_blocks"], k=cfg["k"],
                          R=cfg["R"], device=device, core_mode=cfg["core_mode"])
    model.load_state_dict(raw["model"])
    arch = {"n_blocks": cfg["n_blocks"], "k": cfg["k"], "R": cfg["R"],
            "core_mode": cfg["core_mode"], "variant": cfg.get("variant", "sparse")}
    return model, arch, step


# ─── Collect: Traces + Router-Verteilung ─────────────────────────────────────

@torch.no_grad()
def collect_routing(model, data, n_batches, bs, seq_len, device):
    """Sammelt Traces + full_probs fuer Routing-Diagnose-Metriken."""
    traces      = []   # list of (R, B, T, k) int arrays
    full_probs_list = []   # list of (B, T, n_blocks) float32 tensors
    doms        = []

    for _ in range(n_batches):
        toks, _, _, dom = data.batch(bs, seq_len, device=device, mode="contiguous")
        _, aux = model(toks)
        B, T = toks.shape
        k = aux["iters"][0]["topk_idx"].shape[-1]
        R = len(aux["iters"])

        trace = np.stack([a["topk_idx"].cpu().numpy() for a in aux["iters"]])  # (R,B,T,k)
        traces.append(trace)
        doms.append(dom.cpu().numpy())

        # full_probs ist detached in aux["iters"][0], Shape (N, n_blocks) mit N=B*T
        fp = aux["iters"][0]["full_probs"]   # (B*T, n_blocks) detached
        full_probs_list.append(fp.view(B, T, -1).cpu())

    return traces, full_probs_list, doms


# ─── Routing-Diagnose-Metriken ────────────────────────────────────────────────

def routing_diag_metrics(full_probs_list, traces, k):
    """Berechnet alle Routing-Diagnose-Metriken aus gesammelten Verteilungen."""
    soft_ov_raw   = []
    soft_ov_sharp = []
    hard_ov       = []
    entropies     = []
    margins       = []

    for fp, trace in zip(full_probs_list, traces):
        # fp: (B, T, n_blocks), trace: (R, B, T, k)
        B, T, nb = fp.shape

        # Soft Jaccard (raw + sharpened)
        loss_raw   = soft_full_jaccard_loss(fp)
        loss_sharp = soft_sharp_jaccard_loss(fp, alpha=SHARP_ALPHA)
        soft_ov_raw.append(float(1.0 - loss_raw.item()))
        soft_ov_sharp.append(float(1.0 - loss_sharp.item()))

        # Router-Entropie
        entropies.append(router_entropy(fp))

        # Top-k Margin: P_k - P_{k+1} (letzte gewaehlt vs. erste ausgeschlossen)
        sorted_p = fp.sort(dim=-1, descending=True).values  # (B, T, nb)
        margin = (sorted_p[:, :, k-1] - sorted_p[:, :, k]).mean()
        margins.append(float(margin.item()))

        # Hard Jaccard (eval-Mode, kein Noise)
        topk = torch.from_numpy(trace[0])   # (B, T, k) — nur R=1 (Routing-Schritt)
        nb_est = int(topk.max().item()) + 1
        oh = torch.zeros(B, T, max(nb_est, nb), dtype=torch.bool)
        oh.scatter_(2, topk, True)
        inter = (oh[:, 1:, :] & oh[:, :-1, :]).sum(dim=-1).float()
        union = (oh[:, 1:, :] | oh[:, :-1, :]).sum(dim=-1).clamp_min(1).float()
        hard_ov.append(float((inter / union).mean().item()))

    return {
        "soft_overlap_raw":   round(float(np.mean(soft_ov_raw)), 4),
        "soft_overlap_sharp": round(float(np.mean(soft_ov_sharp)), 4),
        "hard_overlap_eval":  round(float(np.mean(hard_ov)), 4),
        "router_entropy":     round(float(np.mean(entropies)), 4),
        "topk_margin":        round(float(np.mean(margins)), 6),
    }


# ─── Core-Analyse (unique_cores, dead_blocks) ─────────────────────────────────

def core_stats(traces):
    core_set   = set()
    used_blocks = set()
    for trace in traces:
        R, B, T, k = trace.shape
        for b in range(B):
            for t in range(T):
                blks = tuple(sorted(int(x) for x in np.unique(trace[0, b, t])))
                core_set.add(blks)
                used_blocks.update(blks)
    dead = N_BLOCKS - len(used_blocks)
    return {"unique_cores": len(core_set), "dead_blocks": max(0, dead)}


def hot_blocks_top(traces, n=8):
    bc = Counter()
    for trace in traces:
        R, B, T, k = trace.shape
        for b in range(B):
            for t in range(T):
                for x in np.unique(trace[0, b, t]):
                    bc[int(x)] += 1
    return [b for b, _ in bc.most_common(n)]


# ─── Reuse-Distanz P50/P90/P99 ───────────────────────────────────────────────

def reuse_percentiles(traces):
    dists = []
    for trace in traces:
        R, B, T, k = trace.shape
        for b in range(B):
            last = {}; i = 0
            for t in range(T):
                for blk in np.unique(trace[0, b, t]):
                    blk = int(blk)
                    if blk in last:
                        dists.append(i - last[blk])
                    last[blk] = i
                    i += 1
    if not dists:
        return {"p50": 0, "p90": 0, "p99": 0}
    d = np.array(dists)
    return {"p50": float(np.percentile(d, 50)),
            "p90": float(np.percentile(d, 90)),
            "p99": float(np.percentile(d, 99))}


# ─── Leiterbahn (vereinfacht, LRU + hot_core) ────────────────────────────────

def _compute_r_stop_kl(logits_all, device, min_R=3, theta=0.005):
    R = len(logits_all)
    B, T = logits_all[0].shape[:2]
    r_stop = torch.full((B, T), R, dtype=torch.long, device=device)
    stopped = torch.zeros(B, T, dtype=torch.bool, device=device)
    p_prev = F.softmax(logits_all[0], -1)
    for r in range(1, R):
        p_cur = F.softmax(logits_all[r], -1)
        kl = (p_prev * (p_prev.log() - p_cur.log())).sum(-1)
        if r >= min_R:
            fire = (~stopped) & (kl < theta)
            r_stop[fire] = r
            stopped |= fire
        p_prev = p_cur
    return r_stop


@torch.no_grad()
def collect_lb(model, data, n_batches, bs, seq_len, device):
    traces, r_stops, doms = [], [], []
    for _ in range(n_batches):
        toks, _, _, dom = data.batch(bs, seq_len, device=device, mode="contiguous")
        logits_all, aux = model(toks)
        R, B, T = len(logits_all), *toks.shape
        trace = np.stack([a["topk_idx"].cpu().numpy() for a in aux["iters"]])
        rs = _compute_r_stop_kl(logits_all, device).cpu().numpy()
        traces.append(trace); r_stops.append(rs); doms.append(dom.cpu().numpy())
    return traces, r_stops, doms


def sim_lru(traces, r_stops, K):
    demand_miss = demand_access = n_tok = 0
    for trace, r_stop in zip(traces, r_stops):
        R, B, T, k = trace.shape
        n_tok += B * T
        for b in range(B):
            cache = OrderedDict()
            for t in range(T):
                for blk in np.unique(trace[0, b, t]):
                    blk = int(blk)
                    demand_access += 1
                    if blk in cache:
                        cache.move_to_end(blk)
                    else:
                        demand_miss += 1
                        cache[blk] = True
                        if len(cache) > K:
                            cache.popitem(last=False)
    return demand_miss, demand_access, n_tok


def sim_hot_core_pin(traces, r_stops, K, hot_core_b):
    pinned = set(hot_core_b[:min(K, len(hot_core_b))])
    k_lru = max(0, K - len(pinned))
    demand_miss = demand_access = n_tok = 0
    for trace, r_stop in zip(traces, r_stops):
        R, B, T, k = trace.shape
        n_tok += B * T
        for b in range(B):
            cache = OrderedDict()
            for t in range(T):
                for blk in np.unique(trace[0, b, t]):
                    blk = int(blk)
                    demand_access += 1
                    if blk in pinned:
                        continue
                    if blk in cache:
                        cache.move_to_end(blk)
                    else:
                        demand_miss += 1
                        cache[blk] = True
                        if len(cache) > k_lru:
                            cache.popitem(last=False)
    return demand_miss, demand_access, n_tok


def run_lb_sweep(traces, r_stops, block_bytes, hot_core_b, Ks=LB_KS):
    out = {}
    for K in Ks:
        miss_lru, acc_lru, n_tok = sim_lru(traces, r_stops, K)
        miss_hot, acc_hot, _     = sim_hot_core_pin(traces, r_stops, K, hot_core_b)
        bpt_lru  = miss_lru / max(n_tok, 1) * block_bytes / 1024
        bpt_hot  = miss_hot / max(n_tok, 1) * block_bytes / 1024
        out[K] = {
            "lru":          round(bpt_lru, 1),
            "hot_core_pin": round(bpt_hot, 1),
            "vs_lru_pct":   round((bpt_lru - bpt_hot) / max(bpt_lru, 1e-6) * 100, 1),
        }
    return out


# ─── Print ────────────────────────────────────────────────────────────────────

def print_comparison(label_a, label_b, m_a, m_b):
    def row(name, key, fmt=".4f"):
        va = m_a.get(key, float("nan"))
        vb = m_b.get(key, float("nan"))
        delta = vb - va if isinstance(va, float) else 0
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<28}  {va:{fmt}}  {vb:{fmt}}  ({sign}{delta:{fmt}})")

    print(f"\n  {'Metrik':<28}  {label_a:<10}  {label_b:<10}  Delta")
    print("  " + "-" * 65)
    print("  --- Routing-Verteilung (eval, kein Noise) ---")
    row("soft_overlap_raw",    "soft_overlap_raw")
    row("soft_overlap_sharp",  "soft_overlap_sharp")
    row("hard_overlap_eval",   "hard_overlap_eval")
    row("router_entropy",      "router_entropy")
    row("topk_margin",         "topk_margin", fmt=".5f")
    print("  --- Core-Struktur ---")
    row("unique_cores",        "unique_cores", fmt=".0f")
    row("dead_blocks",         "dead_blocks",  fmt=".0f")
    print("  --- Reuse-Distanz ---")
    row("reuseP50",            "reuseP50", fmt=".1f")
    row("reuseP90",            "reuseP90", fmt=".1f")
    row("reuseP99",            "reuseP99", fmt=".1f")

    print("\n  --- Leiterbahn: LRU bytes/token [KB] ---")
    print(f"  {'K':<8}  {label_a:<12}  {label_b:<12}  Delta  vs_LRU_a  vs_LRU_b")
    print("  " + "-" * 60)
    for K in LB_KS:
        ka = m_a.get("lb", {}).get(K, {})
        kb = m_b.get("lb", {}).get(K, {})
        la, lb_ = ka.get("lru", float("nan")), kb.get("lru", float("nan"))
        ha, hb_ = ka.get("hot_core_pin", float("nan")), kb.get("hot_core_pin", float("nan"))
        dlru = lb_ - la
        sign = "+" if dlru >= 0 else ""
        print(f"  K={K:<5}  lru={la:6.1f} KB    lru={lb_:6.1f} KB    ({sign}{dlru:.1f})")
        print(f"  {'':<7}  hot={ha:6.1f} KB    hot={hb_:6.1f} KB    "
              f"({'+' if hb_-ha>=0 else ''}{hb_-ha:.1f})  "
              f"vs_lru: {ka.get('vs_lru_pct',0):+.1f}%  {kb.get('vs_lru_pct',0):+.1f}%")


# ─── Haupt-Eval pro Checkpoint ────────────────────────────────────────────────

def eval_one(ck_path, data, data_heldout, n_batches, bs, seq_len, device, label):
    print(f"\n=== Eval: {label} ({os.path.basename(ck_path)}) ===", flush=True)
    model, arch, step = _load_any(ck_path, device)
    k = arch["k"]
    block_bytes = sum(p.numel() for p in model.bank.blocks[0].parameters()) * 2
    print(f"  step={step}  n_blocks={arch['n_blocks']}  k={k}  R={arch['R']}", flush=True)
    model.eval()

    # Routing-Diagnose (Seen-Daten)
    print("  Routing-Diagnose...", flush=True)
    traces_s, fp_list, doms_s = collect_routing(model, data, n_batches, bs, seq_len, device)
    diag = routing_diag_metrics(fp_list, traces_s, k)
    cs   = core_stats(traces_s)
    rdist = reuse_percentiles(traces_s)
    hot_b = hot_blocks_top(traces_s, n=k)

    # Leiterbahn (Held-out)
    print("  Leiterbahn-Sweep...", flush=True)
    traces_h, r_stops_h, doms_h = collect_lb(model, data_heldout, n_batches, bs, seq_len, device)
    lb = run_lb_sweep(traces_h, r_stops_h, block_bytes, hot_b, LB_KS)

    metrics = {**diag, **cs,
               "reuseP50": rdist["p50"], "reuseP90": rdist["p90"], "reuseP99": rdist["p99"],
               "block_bytes": block_bytes, "lb": lb}
    return metrics, step


# ─── Main ────────────────────────────────────────────────────────────────────

def _jsonify(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_jsonify(i) for i in obj]
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_a", default=DEFAULT_A,
                    help="Checkpoint A (Control, default: ctrl_17k)")
    ap.add_argument("--ckpt_b", default=DEFAULT_B,
                    help="Checkpoint B (Probe, default: softfull_17k)")
    ap.add_argument("--label_a", default="ctrl@17k")
    ap.add_argument("--label_b", default="softfull@17k")
    ap.add_argument("--device",    default="cuda")
    ap.add_argument("--n_batches", type=int, default=40)
    ap.add_argument("--bs",        type=int, default=16)
    ap.add_argument("--seq_len",   type=int, default=128)
    ap.add_argument("--out", default=os.path.join(RESULTS, "eval_compare_17k.json"))
    a = ap.parse_args()

    data         = HeteroMiniData()
    data_heldout = HeteroMiniData(out_dir=HELDOUT_ROOT)

    m_a, step_a = eval_one(a.ckpt_a, data, data_heldout,
                           a.n_batches, a.bs, a.seq_len, a.device, a.label_a)
    m_b, step_b = eval_one(a.ckpt_b, data, data_heldout,
                           a.n_batches, a.bs, a.seq_len, a.device, a.label_b)

    print_comparison(a.label_a, a.label_b, m_a, m_b)

    result = {
        "label_a": a.label_a, "label_b": a.label_b,
        "step_a": step_a,     "step_b": step_b,
        "metrics_a": _jsonify(m_a),
        "metrics_b": _jsonify(m_b),
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nGespeichert: {a.out}", flush=True)


if __name__ == "__main__":
    main()
