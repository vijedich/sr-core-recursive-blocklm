# Recursive Block-Sparse Language Models — Research Monograph

**Status: Self-published research monograph / technical report. Not peer-reviewed, not a thesis.**

This directory is a monograph-style write-up of the SR-Core research project — a block-sparse
recursive language model with a hard working-set guarantee (WS=k). It covers the full
experimental arc: Phase-1 feasibility, bank-size scaling, the CPU dispatch tax, the HeteroMini
multi-domain experiments, entropy-based router consolidation, a real-hardware RAM→VRAM streaming
prototype, and the param/compute-matched quality-cost characterization.

The work is comprehensive and book-length but produced by a single author without formal
supervision or defense. It is a **research monograph / technical report** — not a doctoral
dissertation and does not claim to be.

---

## Chapter status

| Chapter | Title | Completeness | Notes |
|---|---|---|---|
| [01_motivation.md](01_motivation.md) | Motivation and Problem Statement | ~90% | Stable |
| [02_architecture.md](02_architecture.md) | SR-Core Architecture and WS=k | ~90% | Stable |
| [03_scaling.md](03_scaling.md) | Bank Size Scaling | ~95% | All tables + cross-seed ablation |
| [04_dispatch.md](04_dispatch.md) | Dispatch Overhead | ~90% | Benchmark + CPU recorded (i7-10700) |
| [05_heteromini.md](05_heteromini.md) | HeteroMini Experiments | ~97% | + streaming (5b.4), A-sweep (5d), quality gap (budget-matched, 5a.5) |
| [06_entropy_minimization.md](06_entropy_minimization.md) | Entropy-based Consolidation | ~95% | Long-form paper; λ-sweep now 4 seeds |
| [07_discussion.md](07_discussion.md) | Discussion and Future Work | ~90% | Claim boundaries updated to measured results |
| [08_related_work.md](08_related_work.md) | Related Work | ~90% | MoE, looped depth, offloading, routing regularization, positioning table |

---

## The four load-bearing results

1. **WS=k guarantee** — architectural and empirical across bank sizes (Ch 2–3).
2. **Quality cost ~0.5 nats, not explained by budget.** Param- *and* compute-matched SR-Core
   still trails Dense d24 by ~0.5 nats (Ch 5a.5). Matching parameters and compute does not
   close the gap; the evidence points to the narrow-active-set + recursion format. Lower bound
   — both models undertrained at 15k; convergence open (see 7.5.6).
3. **Real wall-clock streaming win, measured.** RAM→VRAM prototype on an RTX 2060: ~1.6× over
   dense layer-offloading once per-block launch overhead is removed and transfer/compute
   overlapped (Ch 5b.4). The chain *fewer bytes → more tokens/s* is measured, not simulated.
4. **Recursion has a useful-depth band** (Ch 5d): floor at A ≈ 1M; per-step gain saturates by
   r≈4; magnitude is seed-dominated.

## What is open (needs hardware/curiosity beyond this program)

- **Convergence of the quality gap** (Ch 7.5.6) — the one cheap, decisive run left: train the
  matched models to ~40k steps and see if the ~0.5-nat gap narrows. ~6–8 h on the same 2060.
- **Deployment-scale streaming** — ≫VRAM models, large blocks, batched serving (7.4 / 7.5.1).
- **Scale validation** (7B+), downstream tasks, the Leiterbahn index (7.5.2–7.5.3).

## Trained artifacts (hosted on Hugging Face)

Checkpoints are **not** committed to git (see the repository root README for the Hugging Face
link); the local `data/checkpoints/` directory is gitignored. The HeteroMini set is complete
for the quality argument:
- SR-Core b64 k8 R6: seeds **s0–s3** (4 seeds)
- Dense d24 (17k), Dense d48 (compute-rough baseline, bs=8 — superseded by the param-matched run)
- **Param/compute-matched SR-Core** (b64 k16 R4 d256h192, 3 seeds) — the definitive quality-cost evidence
- A-sweep: b64 R6 k∈{2,4,8,16} + k8 ×3 seeds
- Entropy sweep: lam003 / lam005 now at **4 seeds** (s0–s3); lam001/lam007/ctrl as before

Loose end: the cross-seed offload *cache* table for λ s2/s3 was not recomputed (`offload_sim`
labels by architecture, not by λ-variant); the 4-seed quality numbers are in the trajectories.

## Data sources

```
data/eval/phase1/          Phase-1 synthetic + TinyStories + cpu_benchmark (CPU now recorded)
data/eval/heteromini/      HeteroMini matrix, long-run, cross-seed, A-sweep + param-matched anytime
data/eval/entmin/          Entropy-minimization sweep (Chapter 6)
data/eval/streaming/       Real RAM→VRAM measurements — Chapter 5b.4
                           (streaming_results.json = Stage 1; streaming_stage{2,3}.json = grouped/overlap)
```

Figures: `figures/` — Chapter 6 (entropy Pareto, router consolidation, dense-vs-sparse) and
`fig_asweep_depth.png` (Chapter 5d).

---

*Last updated: 2026-06-28. Core results frozen; convergence run (7.5.6) left as the open experiment.*
