"""Domain-Partition-Analyse fuer SR-Core-Modelle.

Kernfrage (Viktor, k8_R6): Lernt k8 feingranulare Kompetenzkerne oder hauptsaechlich
vier Domaenen-Cores? (4 Domaenen * 8 Bloecke = 32 Bloecke -> perfekte Partition moeglich.)

Analysiert fuer SR-Core (core_mode='per_token'):
  - Welche Block-Kombination (Core) wird pro Domaene am haeufigsten gewaehlt? (r1 = fixed core)
  - Wie viele einzigartige Cores gibt es gesamt vs. pro Domaene?
  - Top-Core-Coverage: welcher Anteil der Token nutzt die top-1/5/10 Cores?
  - Block-Overlap zwischen Domaenen (Jaccard der genutzten Bloecke)
  - Dominanz: fuer jeden Block, welche Domaene nutzt ihn am staerksten?

Fuer Naked Sparse: analysiert r1-Routing (Iteration 0), da kein fixer Core existiert.
Interpretation gilt analog, aber Cores sind pro-Iteration, nicht reused.

Nutzung:
  python -m experiments.domain_partition --glob "results/hm_cont_*.pt"
  python -m experiments.domain_partition --checkpoints results/hm_cont_hm_srcore_b32_k8_R6_s0.pt
"""
from __future__ import annotations
import argparse, json, os
from collections import Counter
import numpy as np
import torch

from rblm.heteromini import HeteroMiniData
from rblm import model_io
from experiments.heteromini_eval import _collect, _domain_jaccard
from experiments.tinystories_exp import RESULTS


@torch.no_grad()
def analyze(model, data, n_batches=40, bs=16, seq_len=128, device="cpu"):
    """Vollstaendige Domain-Partition-Analyse eines SR-Core- oder Naked-Modells."""
    traces, doms, losses, R, k = _collect(model, data, "contiguous", n_batches, bs, seq_len, device)
    nb = model.cfg.n_blocks
    n_domains = data.n_domains

    # r1-Core pro Token (fuer SR-Core: das ist der feste Core; fuer Naked: nur Iteration 0)
    cores_per_dom = {d: [] for d in range(n_domains)}    # domain -> list of frozensets
    all_cores = []
    all_domains = []

    for tr, dom in zip(traces, doms):        # tr: (R, bs, T, k)
        _, B, T, _ = tr.shape
        for b in range(B):
            d = int(dom[b])
            if d < 0:
                continue
            for t in range(T):
                core = tuple(sorted(int(x) for x in tr[0, b, t, :]))  # r1 blocks, sorted
                cores_per_dom[d].append(core)
                all_cores.append(core)
                all_domains.append(d)

    # 1. Einzigartige Cores gesamt und pro Domaene
    n_unique_total = len(set(all_cores))
    n_unique_per_dom = {data.domains[d]: len(set(cores_per_dom[d]))
                        for d in range(n_domains)}

    # 2. Top-Cores pro Domaene (haeufigste Blockkombinationen)
    top_cores_per_domain = {}
    for d in range(n_domains):
        cnt = Counter(cores_per_dom[d])
        total = len(cores_per_dom[d])
        top_cores_per_domain[data.domains[d]] = [
            {"blocks": list(core), "count": count, "frac": round(count / max(1, total), 4)}
            for core, count in cnt.most_common(10)
        ]

    # 3. Top-Core-Coverage: welcher Anteil der Token nutzt die top-N Cores?
    def coverage_at_n(cores, ns=(1, 5, 10, 20)):
        cnt = Counter(cores)
        total = len(cores)
        result = {}
        cumul = 0
        top = cnt.most_common()
        for i, (_, count) in enumerate(top):
            cumul += count
            n_rank = i + 1
            if n_rank in ns:
                result[f"top_{n_rank}"] = round(cumul / max(1, total), 3)
        for n in ns:
            if f"top_{n}" not in result:
                result[f"top_{n}"] = round(cumul / max(1, total), 3)
        return result

    core_coverage = {data.domains[d]: coverage_at_n(cores_per_dom[d])
                     for d in range(n_domains)}

    # 4. Block-Nutzungsmatrix pro Domaene (fuer Jaccard)
    frac = np.zeros((n_domains, nb))
    for d in range(n_domains):
        for core in cores_per_dom[d]:
            for b in core:
                frac[d, b] += 1
        n_tok = len(cores_per_dom[d])
        if n_tok > 0:
            frac[d] /= n_tok

    # 5. Block-Overlap zwischen Domaenen (Jaccard der Bloecke mit frac > threshold)
    thresh = 0.01   # Block gilt als "genutzt" wenn Domaene ihn in >1% der Token waehlt
    dom_blocks_active = {data.domains[d]: set(np.where(frac[d] > thresh)[0].tolist())
                         for d in range(n_domains)}

    block_overlap = {}
    dom_names = data.domains
    for i in range(n_domains):
        for j in range(i + 1, n_domains):
            bi = dom_blocks_active[dom_names[i]]
            bj = dom_blocks_active[dom_names[j]]
            inter = len(bi & bj)
            union = len(bi | bj)
            block_overlap[f"{dom_names[i]}_x_{dom_names[j]}"] = {
                "intersection": inter,
                "union": union,
                "jaccard": round(inter / max(1, union), 3),
                "blocks_i": len(bi),
                "blocks_j": len(bj),
            }

    # 6. Domain-Jaccard der top-k Bloecke (wie in heteromini_eval)
    jm, joff = _domain_jaccard(frac, cap=k)
    domain_jaccard_matrix = {dom_names[i]: {dom_names[j]: round(jm[i][j], 3)
                                            for j in range(n_domains)}
                             for i in range(n_domains)}

    # 7. Block-Dominanz: fuer jeden Block, welche Domaene nutzt ihn am staerksten?
    dominant_domain = {}
    for b in range(nb):
        col = frac[:, b]
        if col.sum() < 1e-6:
            dominant_domain[b] = {"domain": None, "frac": 0.0, "status": "dead"}
        else:
            best = int(np.argmax(col))
            dominant_domain[b] = {
                "domain": dom_names[best],
                "frac": round(float(col[best]), 3),
                "all_fracs": {dom_names[di]: round(float(col[di]), 3) for di in range(n_domains)},
            }
    dom_counts = Counter(v["domain"] for v in dominant_domain.values() if v["domain"])
    blocks_per_dom = {dom_names[d]: dom_counts.get(dom_names[d], 0) for d in range(n_domains)}

    # 8. Partition-Score: wie "exklusiv" ist die Domaenenzuordnung?
    # Fuer jeden Block: dominanz_ratio = max_frac / sum_frac (1.0 = voellig exklusiv, 0.25 = gleich)
    exclusivity = []
    for b in range(nb):
        col = frac[:, b]
        s = col.sum()
        if s > 1e-6:
            exclusivity.append(float(col.max() / s))
    mean_exclusivity = round(float(np.mean(exclusivity)) if exclusivity else 0.0, 3)
    # Erwartungswert bei zufaelliger Gleichverteilung: 1/n_domains; bei perfekter Partition: 1.0
    chance_exclusivity = round(1.0 / n_domains, 3)

    return {
        "n_blocks": nb,
        "k_active": k,
        "R": R,
        "n_domains": n_domains,
        "domains": dom_names,
        "n_tokens_analyzed": len(all_cores),
        "n_unique_cores_total": n_unique_total,
        "n_unique_cores_per_domain": n_unique_per_dom,
        "top_cores_per_domain": top_cores_per_domain,
        "core_coverage": core_coverage,
        "block_overlap_between_domains": block_overlap,
        "domain_jaccard_offdiag_mean": round(joff, 3),
        "domain_jaccard_matrix": domain_jaccard_matrix,
        "dominant_domain_per_block": dominant_domain,
        "blocks_dominated_per_domain": blocks_per_dom,
        "mean_block_exclusivity": mean_exclusivity,
        "chance_exclusivity_uniform": chance_exclusivity,
        "loss_r1": round(losses[0], 4),
        "loss_rR": round(losses[-1], 4),
    }


def print_report(res, name=""):
    dom = res["domains"]
    nb = res["n_blocks"]
    k = res["k_active"]
    n_dom = res["n_domains"]

    print(f"\n=== DOMAIN-PARTITION-ANALYSE  {name} ===")
    print(f"n_blocks={nb}  k={k}  R={res['R']}  Domaenen={n_dom}  "
          f"(k*Domaenen={k*n_dom}  Bankgroesse={nb}  "
          f"{'Perfekte Partition MOEGLICH' if k*n_dom==nb else f'Ratio={k*n_dom}/{nb}'})")
    print(f"Tokens analysiert: {res['n_tokens_analyzed']:,}  "
          f"Einzigartige Cores gesamt: {res['n_unique_cores_total']:,}")

    print(f"\n--- Einzigartige Cores pro Domaene ---")
    for d, n in res["n_unique_cores_per_domain"].items():
        print(f"  {d:8}: {n:5} unique cores")

    print(f"\n--- Top-Core-Coverage (Anteil Token, der top-N Cores nutzt) ---")
    for d, cov in res["core_coverage"].items():
        line = "  " + d.ljust(8) + ": " + "  ".join(f"top{k}={v:.1%}" for k, v in cov.items())
        print(line)

    print(f"\n--- Blocks dominiert pro Domaene (Block zaehlt zur Domaene mit hoechster Nutzung) ---")
    for d, cnt in res["blocks_dominated_per_domain"].items():
        print(f"  {d:8}: {cnt:3} Bloecke")
    print(f"  Block-Exklusivitaet: mean={res['mean_block_exclusivity']:.3f}  "
          f"(Zufall={res['chance_exclusivity_uniform']:.3f}  Perfekt=1.000)")

    print(f"\n--- Block-Overlap zwischen Domaenen (Jaccard der aktiven Bloecke) ---")
    for pair, v in res["block_overlap_between_domains"].items():
        print(f"  {pair:25}: Jaccard={v['jaccard']:.3f}  "
              f"Schnittmenge={v['intersection']}  ({v['blocks_i']} vs {v['blocks_j']} Bloecke)")
    print(f"  Domain-Jaccard (top-k Bloecke): offdiag-mean={res['domain_jaccard_offdiag_mean']:.3f}")

    print(f"\n--- Top-Core pro Domaene ---")
    for d, cores in res["top_cores_per_domain"].items():
        if not cores:
            continue
        top = cores[0]
        print(f"  {d:8}: {top['blocks']}  ({top['frac']:.1%} der Token)")

    print(f"\n--- Partition-Diagnose ---")
    excl = res["mean_block_exclusivity"]
    chance = res["chance_exclusivity_uniform"]
    joff = res["domain_jaccard_offdiag_mean"]

    if excl > 0.85 and joff < 0.15:
        verdict = "STARKE PARTITION — k{} lernt hauptsaechlich Domaenen-Macro-Cores.".format(k)
    elif excl > 0.65 and joff < 0.30:
        verdict = "MODERATE PARTITION — Domaenen-Tendenz erkennbar, aber gemischt."
    else:
        verdict = "SCHWACHE PARTITION — Bloecke werden domaenenuebergreifend genutzt."
    print(f"  {verdict}")
    print(f"  (excl={excl:.3f} vs chance={chance:.3f}, domain-Jaccard={joff:.3f})")


def run(path, n_batches=40, bs=16, seq_len=128, device="cpu"):
    data = HeteroMiniData()
    model, arch, step = model_io.load_checkpoint(path, data.vocab_size, device)
    if model_io.is_dense(arch):
        print(f"[DomainPartition] {path}: Dense-Modell uebersprungen (kein Router).")
        return None
    name = model_io.label(arch, step)
    res = analyze(model, data, n_batches, bs, seq_len, device)
    res["experiment"] = name
    res["step"] = step
    out = os.path.join(RESULTS, f"domain_partition_{name}_s0.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print_report(res, name)
    print(f"\nGespeichert: {out}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/hm_cont_*.pt")
    ap.add_argument("--checkpoints", nargs="*", default=None)
    ap.add_argument("--n_batches", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    cks = a.checkpoints or model_io.discover(a.glob)
    if not cks:
        raise SystemExit(f"Keine Checkpoints (glob={a.glob!r}).")
    for p in cks:
        run(p, n_batches=a.n_batches, bs=a.bs, seq_len=a.seq_len, device=a.device)
