# Recursive Block-Sparse Language Models — Research Monograph

**Self-published research monograph / technical report. Not peer-reviewed, not a thesis.**
Author: Viktor Jedich. Text licensed [CC BY 4.0](../LICENSE-CC-BY-4.0.md).

This directory holds the full monograph and everything needed to verify it: the chapter
text, every eval artifact that backs a numerical claim, the publication figures, and the
scripts that regenerate those figures from the eval data.

## Contents

```
docs/chapters/    The monograph — 8 chapters (Markdown). Start at README.md, then 01–08.
data/eval/        All eval JSON artifacts, organized by experiment:
  phase1/           Phase-1 synthetic + TinyStories scaling + CPU benchmark + offload sim
  heteromini/       HeteroMini matrix, long-run, cross-seed, A-sweep, param-matched anytime
  entmin/           Entropy-minimization sweep (Chapter 6)
  streaming/        Real RAM→VRAM measurements (Chapter 5b.4)
figures/          Publication figures (PNG), regenerated from data/eval/
scripts/          Figure-generation and evaluation scripts
```

Model checkpoints are **not** stored here (too large); they are hosted on Hugging Face —
see the repository root [README](../README.md).

## Where to start

- Read the chapters in order beginning with [`docs/chapters/README.md`](docs/chapters/README.md),
  which carries the chapter map and the four load-bearing results.
- To verify a number, find its data source (each chapter cites the relevant
  `data/eval/...` file) and inspect the JSON directly.
- To regenerate the figures: `python scripts/build_figures.py` and
  `python scripts/plot_asweep.py` (run from the repository root, which provides `rblm/`).

## License

The monograph text and figures are licensed under [CC BY 4.0](../LICENSE-CC-BY-4.0.md).
The scripts in `scripts/` are licensed under [MIT](../LICENSE), like the rest of the code.
