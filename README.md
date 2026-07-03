# SR-Core: Recursive Block-Sparse Language Models

**Research monograph and code for SR-Core** — a block-sparse recursive language model with a
hard working-set guarantee (WS=k) and entropy-based router consolidation for cache-efficient
RAM→VRAM parameter streaming.

> Not peer-reviewed. Self-published research monograph. See license section below.

---

## The one-sentence idea

A shared block bank is applied recursively; routing is decided once (at r=1) and reused for
all subsequent steps — so exactly **k blocks are active per token**, independent of bank size
or recursion depth. This makes the active weight footprint per token statically bounded,
enabling predictable RAM→VRAM prefetching.

---

## How to read this

This is not a claim-maximizing conference paper. It is a self-contained research monograph
documenting an end-to-end investigation of one architectural idea under consumer-hardware
constraints (RTX 2060, 6 GB VRAM). The work asks what can be learned about sparse recursive
offloading architectures when you actually build and measure them on small models — including
where the first naive implementation fails, why, and what makes it work.

The "small 2060, small models, but consistently measured" constraint is not a shortcoming.
It is the scope. The monograph documents what SR-Core *can* demonstrably do at this scale,
and explicitly marks what remains open for anyone with more hardware or more curiosity.

---

## Four load-bearing results

| # | Result | Where |
|---|---|---|
| 1 | **WS=k guarantee holds** across bank sizes (n=32 to 64) and recursion depths (R=2 to 6), architectural and empirical | Ch. 2–3 |
| 2 | **Quality cost ~0.5 nats, not explained by budget:** a param- *and* compute-matched SR-Core (8.75M params, 6.29M apps/token = Dense d24) still trails by ~0.5 nats. The evidence points to the narrow-active-set + recursion format. Lower bound — convergence open (7.5.6). | Ch. 5a.5 |
| 3 | **Real wall-clock streaming win measured** on an RTX 2060: ~1.6× over dense layer-offloading once per-block kernel-launch overhead is removed (grouped matmul) and transfer/compute overlap is added. Chain "fewer bytes → more tokens/s" is measured end-to-end, not simulated. | Ch. 5b.4 |
| 4 | **Recursion useful-depth band**: floor at active-brain A ≈ 1M params; per-step gain saturates by r ≈ 4; magnitude is seed-dominated. | Ch. 5d |

---

## Repository structure

```
rblm/                       Core model library (model, router, trainer, tokenizer, eval)
experiments/                Training and evaluation scripts (heteromini_long.py, streaming_prototype.py, …)
monograph/
  docs/chapters/            Research monograph — 8 chapters (Markdown)
  data/eval/                All eval JSON artifacts backing every claim in the text
  figures/                  Publication-ready figures (PNG)
  scripts/                  Figure-generation and eval scripts
data/                       Training corpus — not committed (rebuild with rblm/heteromini.py)
```

---

## Research monograph

The full write-up is in [`monograph/docs/chapters/`](monograph/docs/chapters/).
Eight chapters, all numerical claims backed by eval JSON files in `monograph/data/eval/`.
A rendered PDF of the whole monograph is committed at
[`monograph/SR-Core-Monograph.pdf`](monograph/SR-Core-Monograph.pdf)
(rebuild: `python monograph/scripts/build_pdf.py` — needs pandoc + xelatex).

| Chapter | Topic |
|---|---|
| 01 | Motivation and Problem Statement |
| 02 | SR-Core Architecture and WS=k |
| 03 | Bank Size Scaling |
| 04 | CPU Dispatch Overhead |
| 05 | HeteroMini Experiments (quality, streaming, A-sweep) |
| 06 | Entropy-based Router Consolidation |
| 07 | Discussion and Future Work |
| 08 | Related Work (MoE, looped depth, offloading, routing regularization) |

---

## Reproducing figures

Figures depend only on `monograph/data/eval/` — no checkpoints needed:

```bash
# Entropy Pareto + router consolidation + dense-vs-sparse quality
python monograph/scripts/build_figures.py

# A-sweep depth plot
python monograph/scripts/plot_asweep.py
```

Output goes to `monograph/figures/`.

---

## Training from scratch

```bash
pip install -r requirements.txt   # torch, numpy, matplotlib + datasets, tokenizers (corpus rebuild)

# Single HeteroMini run (GPU required, ~1.5h on RTX 2060 per 15k steps)
python -m experiments.heteromini_long \
    --variant sparse --core_mode per_token \
    --n_blocks 64 --k 8 --R 6 \
    --max_steps 15000 --seed 0

# Param-matched comparison (k=16 R=4 d256 h192)
python -m experiments.heteromini_long \
    --variant sparse --core_mode per_token \
    --n_blocks 64 --k 16 --R 4 \
    --d_model 256 --block_hidden 192 \
    --max_steps 15000 --seed 0

# Dense baseline
python -m experiments.heteromini_long \
    --variant dense --depth 24 \
    --max_steps 17000 --seed 0

# RAM→VRAM streaming prototype (all stages)
python -m experiments.streaming_prototype
```

Checkpoints are saved at milestones (2500 / 5000 / 10000 / 15000 steps) for crash recovery.

---

## Checkpoints

Model checkpoints are not committed to this repository (too large).
They are hosted on Hugging Face: **[vijedich/sr-core-recursive-blocklm](https://huggingface.co/vijedich/sr-core-recursive-blocklm)**

Key checkpoints:
- `hm_cont_hm_srcore_b64_k8_R6_s{0-3}.pt` — SR-Core b64 k8 R6, 4 seeds (quality evidence)
- `hm_cont_hm_dense_d24_s0.pt` — Dense d24 quality ceiling
- `hm_cont_hm_srcore_b64_k16_R4_d256h192_s{0-2}.pt` — Param/compute-matched (quality gap evidence)

---

## Training data

**HeteroMini-v1**: ~6.6M tokens, 4 domains (Web / Wikipedia / Code / Literature), BPE vocab=8000.
Not committed. Rebuild with:

```bash
python -m rblm.heteromini  # downloads and tokenizes from public sources
```

Sources: FineWeb-Edu (web), Wikipedia (wiki), codeparrot-clean-valid (code), Project Gutenberg (literature).

---

## Hardware used

All experiments: **NVIDIA RTX 2060 (6 GB VRAM)**, Intel i7-10700 (8C/16T), 16 GB RAM.
No cloud compute. Training times: ~1.5h per 15k-step run; streaming prototype stages: minutes.

---

## License

| What | License |
|---|---|
| Code (`rblm/`, `experiments/`, `monograph/scripts/`) and model checkpoints | [MIT](LICENSE) |
| Research monograph (`monograph/docs/`) | [CC BY 4.0](LICENSE-CC-BY-4.0.md) |

Copyright (c) 2026 Viktor Jedich

---

## Citation

This work is archived on Zenodo with a citable DOI:
**[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21146547.svg)](https://doi.org/10.5281/zenodo.21146547)**

The concept DOI `10.5281/zenodo.21146547` always resolves to the latest version; the
v1.0 release is `10.5281/zenodo.21146548`.

If you use this work, please cite:

```bibtex
@misc{jedich2026srcore,
  title   = {Entropy-based Router Consolidation for Cache-Efficient
             Recursive Block-Sparse Language Models},
  author  = {Jedich, Viktor},
  year    = {2026},
  note    = {Self-published research monograph},
  doi     = {10.5281/zenodo.21146547},
  url     = {https://github.com/vijedich/sr-core-recursive-blocklm}
}
```

---

## Status

Results frozen as of 2026-06-28. The one open empirical question is the **convergence
run** (§7.5.6): training the param-matched pair to ~40k steps to see whether the
~0.5-nat quality gap narrows. Everything else (scale validation, deployment-scale
streaming, Leiterbahn index) requires hardware beyond a consumer RTX 2060.

**Not peer-reviewed. Claims and their limits are documented in Chapter 7.**
