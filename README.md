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

## Four load-bearing results

| # | Result | Where |
|---|---|---|
| 1 | **WS=k guarantee holds** across bank sizes (n=32 to 64) and recursion depths (R=2 to 6), architectural and empirical | Ch. 2–3 |
| 2 | **Quality cost ~0.5 nats is intrinsic** to the format: a param- *and* compute-matched SR-Core (8.75M params, 6.29M apps/token = Dense d24) still trails by ~0.5 nats. Matching budget does not close the gap. | Ch. 5a.5 |
| 3 | **Real wall-clock streaming win measured** on an RTX 2060: ~1.6× over dense layer-offloading once per-block kernel-launch overhead is removed (grouped matmul) and transfer/compute overlap is added. Chain "fewer bytes → more tokens/s" is measured end-to-end, not simulated. | Ch. 5b.4 |
| 4 | **Recursion useful-depth band**: floor at active-brain A ≈ 1M params; per-step gain saturates by r ≈ 4; magnitude is seed-dominated. | Ch. 5d |

---

## Repository structure

```
rblm/                       Core library (model, router, trainer, eval)
experiments/                Experiment scripts (heteromini_long.py, streaming_prototype.py, …)
scripts/                    Utilities (plot_asweep.py, build_figures.py, …)
queues/                     JSON job queues used for overnight training runs
dissertation/
  docs/dissertation_draft/  Research monograph (8 chapters, Markdown)
  data/eval/                All eval JSON artifacts (no checkpoints)
  figures/                  Publication-ready figures (PNG)
paper/                      Condensed paper drafts (draft_v3.md = submission-ready)
results/                    Raw JSON results from all experiments
data/                       Training corpus — not committed (see below)
```

---

## Research monograph

The full write-up is in [`dissertation/docs/dissertation_draft/`](dissertation/docs/dissertation_draft/).
Eight chapters, all numerical claims backed by eval JSON files in `dissertation/data/eval/`.

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

Figures depend only on `dissertation/data/eval/` — no checkpoints needed:

```bash
# Entropy Pareto + router consolidation + dense-vs-sparse quality
python scripts/build_figures.py

# A-sweep depth plot
python scripts/plot_asweep.py
```

Output goes to `dissertation/figures/`.

---

## Training from scratch

```bash
pip install torch numpy matplotlib

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
They are hosted on Hugging Face: **[HUGGINGFACE-REPO]**

Key checkpoints:
- `hm_cont_hm_srcore_b64_k8_R6_s{0-3}.pt` — SR-Core b64 k8 R6, 4 seeds (quality evidence)
- `hm_cont_hm_dense_d24_s0.pt` — Dense d24 quality ceiling
- `hm_cont_hm_srcore_b64_k16_R4_d256h192_s{0-2}.pt` — Param/compute-matched (intrinsic gap evidence)

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
| Code (`rblm/`, `experiments/`, `scripts/`) and model checkpoints | [MIT](LICENSE) |
| Research monograph (`dissertation/`) and paper drafts (`paper/`) | [CC BY 4.0](LICENSE-CC-BY-4.0.md) |

Copyright (c) 2026 Viktor Jedich

---

## Citation

If you use this work, please cite:

```bibtex
@techreport{jedich2026srcore,
  title   = {Entropy-based Router Consolidation for Cache-Efficient
             Recursive Block-Sparse Language Models},
  author  = {Jedich, Viktor},
  year    = {2026},
  note    = {Self-published research monograph.
             \url{https://github.com/[GITHUB-USERNAME]/sr-core-recursive-blocklm}}
}
```

*Replace `[GITHUB-USERNAME]` with your actual username after publishing.*

---

## Status

Results frozen as of 2026-06-28. The one open empirical question is the **convergence
run** (§7.5.6): training the param-matched pair to ~40k steps to see whether the
~0.5-nat quality gap narrows. Everything else (scale validation, deployment-scale
streaming, Leiterbahn index) requires hardware beyond a consumer RTX 2060.

**Not peer-reviewed. Claims and their limits are documented in Chapter 7.**
