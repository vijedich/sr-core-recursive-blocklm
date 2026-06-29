"""Routing-Analyse fuer Cross-Seed-Vergleich k8_R6 Seeds 0/1/2.

Vergleicht: gini, dead_blocks, unique_cores, top1_coverage,
cache_miss_k16, domain_jaccard, code_core_overlap.

Nutzt heteromini_eval._collect + _domain_usage intern.

Nutzung:
  python scripts/routing_analysis.py --device cuda
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rblm.heteromini import HeteroMiniData
from rblm import model_io
from experiments.heteromini_eval import _collect, _domain_usage, _domain_jaccard, _streams, _cache_miss
from experiments.tinystories_exp import gini

RESULTS = os.path.join(ROOT, "results")
CKPTS = {
    0: "results/BASELINE_srcore_b32_k8_R6_15k.pt",
    1: "results/hm_cont_hm_srcore_b32_k8_R6_s1.pt",
    2: "results/hm_cont_hm_srcore_b32_k8_R6_s2.pt",
}


def routing_stats(traces, doms, n_blocks, n_domains, cache_K=16):
    """Berechnet Routing-Metriken aus Traces (R, bs, T, k)."""
    # Block-Nutzung und Domain-Stats
    frac, toks, membership, labels = _domain_usage(traces, doms, n_blocks, n_domains)
    usage = (frac * toks[:, None]).sum(0)

    # Gini + Dead
    g = float(gini(usage))
    dead = int((usage == 0).sum())

    # Unique Cores + Top-1-Coverage
    core_counter: dict = {}
    total_toks = 0
    for tr, dom in zip(traces, doms):    # tr: (R, bs, T, k)
        R, B, T, k = tr.shape
        for b in range(B):
            for t in range(T):
                core = frozenset(tr[:, b, t, :].flatten().tolist())
                core_counter[core] = core_counter.get(core, 0) + 1
                total_toks += 1

    unique_cores = len(core_counter)
    top1_coverage = max(core_counter.values()) / max(total_toks, 1) if core_counter else 0.0

    # Cache Miss @ K
    streams = _streams(traces)
    miss_k = _cache_miss(streams, [cache_K], n_blocks).get(cache_K, 0.0)

    # Domain Jaccard
    _, joff = _domain_jaccard(frac)

    # Code vs Rest Jaccard
    code_dom_idx = None
    domains_list = None
    code_overlap = None

    return {
        "unique_cores":   unique_cores,
        "total_tokens":   total_toks,
        "top1_coverage":  round(top1_coverage, 4),
        "gini":           round(g, 4),
        "dead_blocks":    dead,
        f"cache_miss_k{cache_K}": round(miss_k, 4),
        "domain_jaccard_offdiag": round(joff, 3),
        "block_usage":    [round(float(u), 4) for u in usage],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",    default="cuda")
    ap.add_argument("--n_batches", type=int, default=40)
    ap.add_argument("--bs",        type=int, default=16)
    ap.add_argument("--seq_len",   type=int, default=128)
    a = ap.parse_args()

    data = HeteroMiniData()
    all_results = {}

    for seed, ck_rel in CKPTS.items():
        ck = os.path.join(ROOT, ck_rel)
        if not os.path.exists(ck):
            print(f"[skip] seed={seed}: nicht gefunden")
            continue
        print(f"\n=== Seed {seed} ===", flush=True)
        model, arch, step = model_io.load_checkpoint(ck, data.vocab_size, a.device)
        name = model_io.label(arch, step)
        print(f"  {name}  n_blocks={arch.get('n_blocks')}  k={arch.get('k')}", flush=True)

        traces, doms, _, R, k = _collect(
            model, data, "contiguous", a.n_batches, a.bs, a.seq_len, a.device)

        stats = routing_stats(traces, doms, arch.get("n_blocks", 32), data.n_domains)
        stats["seed"] = seed
        stats["name"] = name
        stats["step"] = step
        all_results[seed] = stats

        for key, val in stats.items():
            if key not in ("block_usage", "seed", "name", "step"):
                print(f"  {key:28}: {val}")

    out = os.path.join(RESULTS, "routing_analysis_crossseed_k8_R6.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nGespeichert: {out}")

    # Vergleichstabelle
    print("\n=== Cross-Seed Routing-Vergleich k8_R6 ===")
    hdr = f"{'Seed':>5} {'unique_cores':>13} {'top1_cov':>9} {'gini':>6} {'dead':>5} {'miss_k16':>9} {'dom_jac':>8}"
    print(hdr)
    print("-" * len(hdr))
    for seed in sorted(all_results):
        r = all_results[seed]
        miss_key = "cache_miss_k16"
        print(f"{seed:>5} {r['unique_cores']:>13} {r['top1_coverage']:>9.4f} "
              f"{r['gini']:>6.3f} {r['dead_blocks']:>5} "
              f"{r.get(miss_key, 0):>9.4f} {r['domain_jaccard_offdiag']:>8.3f}")


if __name__ == "__main__":
    main()
