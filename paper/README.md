# Entropy-based Router Consolidation for Cache-Efficient Block-Sparse Language Models

Technical report and full experimental artifacts for the SR-Core entropy minimization study.

**PDF:** [`paper/latex/main_compact.pdf`](latex/main_compact.pdf)

---

## What this work shows / does not show

**Shown:**
- SR-Core provides a hard working-set guarantee: exactly *k* blocks are active per token,
  independent of bank size *n* or recursion depth *R*
- Entropy-based router consolidation (λ·H(p) at r=1, pre-topk) induces a controllable
  cache/locality axis: increasing λ monotonically reduces router entropy, unique routing
  combinations, and simulated bytes-in-motion
- λ=0.003 Pareto-dominates the unregularized baseline on both cache efficiency and
  held-out code generalization, replicated across two independent seeds
- λ=0.005 achieves the largest cache improvement (−7.5% at K=24 LRU) with seed-dependent
  quality effects — not a consistent cost or benefit
- Under LRU simulation at K=8, sparse SR-Core requires ~4.1× fewer bytes/token than
  dense layer-offloading (simulated estimate, not measured wall-clock)
- Softfull (cross-token Jaccard penalty) moves routing in the *opposite* direction,
  confirming pre-selection entropy as the correct intervention point

**Not shown:**
- Wall-clock inference speedup (dispatch overhead ~2.8× slower than compute-matched
  dense in CPU benchmark; RAM→VRAM prototype not built)
- Dense-quality parity (dense d24 remains the quality upper bound in all four domains)
- Scaling beyond n=64 blocks or the small-scale HeteroMini-v1 corpus (~6.6M tokens)
- Downstream task evaluation

---

## Repository structure

```
paper/
├── latex/
│   ├── main_compact.pdf        Compiled report (10 pages, single-column)
│   ├── main_compact.tex        Compact layout (1-inch margins, placeins)
│   ├── main.tex                Venue-neutral layout (2.5 cm margins)
│   ├── sections/               Section .tex files (00_abstract – 07_related_work)
│   ├── references.bib          9 BibTeX entries
│   └── Makefile                make / make watch / make clean
│
├── figures/
│   ├── fig_router_consolidation.png   2×2 routing metrics across λ sweep
│   ├── fig_entropy_pareto.png         Cache–quality Pareto plot
│   └── fig_dense_vs_sparse_quality.png  Dense vs. sparse quality by domain
│
├── data/
│   ├── checkpoints/            Model checkpoints (.pt) — see note below
│   └── eval/                   All eval JSON files used by the paper figures
│
├── scripts/
│   ├── build_figures.py        Regenerate all 3 figures from data/eval/
│   ├── eval_compare.py         Routing metrics comparison
│   ├── eval_quality_compare.py Per-domain quality evaluation
│   └── _check_fig_data.py      Data consistency check before figure build
│
└── docs/
    ├── results_note_entmin_sweep.md   Full experimental documentation
    └── writeup_skeleton_entmin.md     Section skeleton with claim boundaries
```

---

## Building the PDF

Requires a LaTeX distribution (MiKTeX, TeX Live, or MacTeX):

```bash
cd paper/latex
make              # pdflatex × 2 + bibtex + pdflatex
# output: main_compact.pdf
```

Or compile manually:
```bash
pdflatex -interaction=nonstopmode -jobname=main_compact main_compact.tex
bibtex main_compact
pdflatex -interaction=nonstopmode -jobname=main_compact main_compact.tex
pdflatex -interaction=nonstopmode -jobname=main_compact main_compact.tex
```

---

## Reproducing figures

All figures are generated from JSON eval data in `data/eval/` — no model checkpoints needed:

```bash
cd paper
python scripts/build_figures.py
# writes to figures/fig_*.png
```

To verify data consistency before building:
```bash
python scripts/_check_fig_data.py
```

---

## Model checkpoints

| Checkpoint | Role |
|---|---|
| `hm_cont_hm_srcore_b64_k8_R6_ctrl_17k_s0/s1.pt` | Unregularized baseline (2 seeds) |
| `hm_cont_hm_dense_d24_17k_s0.pt` | Dense quality upper bound |
| `hm_cont_hm_srcore_b64_k8_R6_entmin_r1_lam003_s0/s1.pt` | λ=0.003 (2 seeds) |
| `hm_cont_hm_srcore_b64_k8_R6_entmin_r1_lam005_s0/s1.pt` | λ=0.005 (2 seeds) |
| `hm_cont_hm_srcore_b64_k8_R6_entmin_lam007_s0.pt` | λ=0.007 |
| `hm_cont_hm_srcore_b64_k8_R6_coreloc_softfull_r1_lam001_s0.pt` | Negative control: softfull |
| `hm_cont_hm_srcore_b64_k8_R6_coreloc_softsharp_a2_lam001_s0.pt` | Negative control: softsharp |
| `hm_cont_hm_srcore_b64_k8_R6_reduced_noise_0p1_s0.pt` | Negative control: reduced noise |
| `hm_cont_hm_srcore_b64_k8_R6_target_H375/H370_s0.pt` | Target-entropy variants |

`.pt` files are tracked via [Git LFS](https://git-lfs.com). If not available locally,
the figures can be reproduced from `data/eval/` JSON files without the checkpoints.

---

## Citation

This is an independent research report, not a peer-reviewed publication.
If you build on this work, please cite as:

```
Jedich, V. (2026). Entropy-based Router Consolidation for Cache-Efficient
Block-Sparse Language Models. Technical report.
```
