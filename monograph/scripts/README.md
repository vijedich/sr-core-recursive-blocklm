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

## B. Eval / training pipeline — requires checkpoints

Every other script in this directory is part of the original evaluation pipeline that
*produced* the JSON files in `../data/eval/`. They load trained checkpoints and are
provided to document how the evals were generated — not as a no-setup reproduction step.

To use them, download the checkpoints from Hugging Face (see the repository root
[README](../../README.md)) and place them under `../data/checkpoints/`. Paths and corpora
follow `rblm/` and `experiments/`; expect a GPU for the training-adjacent scripts.

These scripts retain their original interfaces and were written for the experiment
campaign; they are reference material, not part of the load-bearing reproduction path.
