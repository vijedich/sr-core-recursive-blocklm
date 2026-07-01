# Monograph scripts

Two categories. Run all of them from the **repository root** (so that `rblm/` and
`experiments/` are importable).

## A. Figure regeneration — no checkpoints needed

These read only the committed eval JSON under `../data/eval/` and write to `../figures/`.
They are self-contained and verified to reproduce the published figures:

```bash
python monograph/scripts/build_figures.py    # entropy Pareto, router consolidation, dense-vs-sparse
python monograph/scripts/plot_asweep.py      # A-sweep depth (fig_asweep_depth.png)
python monograph/scripts/_check_fig_data.py  # prints the numbers behind the figures (no plotting)
```

`build_pdf.py` renders the whole monograph (chapters 00–08) to `../SR-Core-Monograph.pdf`
via pandoc + xelatex. The committed PDF is built by this script; regenerate with:

```bash
python monograph/scripts/build_pdf.py
```

It needs pandoc (on PATH, or `pip install pypandoc_binary`), a LaTeX distribution with
xelatex (MiKTeX / TeX Live), and the DejaVu fonts. `latex_header.tex` is the shared LaTeX
preamble it includes.

## B. Eval / training pipeline — requires checkpoints

`eval_four_seed_quality.py` is the auditable source for the per-seed quality tables in
Chapters 5a.5 and 6.5.1. It re-measures every checkpoint (SR-Core k8 R6, param-matched
k16 R4, Dense d24, entropy lam003/lam005) under one fixed protocol on the seen and held-out
corpora, and writes `../data/eval/heteromini/eval_quality_four_seed_summary.json` (that JSON
*is* committed — only the checkpoints it reads are not):

```bash
python monograph/scripts/eval_four_seed_quality.py --entropy   # ~20 min on CPU
```

Every other script in this directory is part of the original evaluation pipeline that
*produced* the JSON files in `../data/eval/`. They load trained checkpoints and are
provided to document how the evals were generated — not as a no-setup reproduction step.

To use them, download the checkpoints from Hugging Face (see the repository root
[README](../../README.md)) and place them under `../data/checkpoints/`. Paths and corpora
follow `rblm/` and `experiments/`; expect a GPU for the training-adjacent scripts.

These scripts retain their original interfaces and were written for the experiment
campaign; they are reference material, not part of the load-bearing reproduction path.
