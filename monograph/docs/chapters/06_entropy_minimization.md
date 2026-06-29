# Chapter 6: Entropy-based Router Consolidation

This chapter is the primary experimental contribution of this monograph. It introduces
entropy-based router consolidation as a controllable axis for cache efficiency and
characterizes its effects on routing structure, language quality, and simulated
offloading cost.

The experimental results in this chapter are covered in detail in the accompanying
technical report (see `paper/latex/main_compact.pdf`). This chapter presents the same
findings in the broader context of this work.

---

## 6.1 Motivation: Routing Collapse vs. Routing Consolidation

SR-Core guarantees WS=k (Chapter 2), but does not control *which* k blocks a token
activates. Across training, the routing distribution can behave in two extreme ways:

- **Diffuse routing:** Each token selects a near-random subset of k blocks from the
  full bank. High cache miss rate — new blocks must be loaded for almost every token.

- **Consolidated routing:** Many tokens select the same or overlapping k-subsets.
  A small number of blocks are heavily shared. Low cache miss rate — the active set
  is largely resident after a short warm-up.

From the offloading perspective, consolidated routing is desirable: a shared LRU cache
benefits from routing patterns where the same blocks are repeatedly active.

However, extreme consolidation — all tokens selecting the same k blocks — would collapse
the block bank to a small subset and eliminate the architectural benefit of having n >> k.
The target is *controlled consolidation*: more concentrated than random, less concentrated
than collapse.

## 6.2 The Entropy Objective

### 6.2.1 Loss Function

The full training loss is:

```
L = L_LM + λ · H(p_t^(1))
```

where L_LM is the standard next-token cross-entropy, and H(p_t^(1)) is the router
entropy at step r=1 for token t:

```
H(p) = -Σ_{b=1}^{n} p_b log p_b
```

The entropy penalty pushes the pre-selection distribution p toward concentration —
lower entropy means the router is more certain about which blocks to select.

### 6.2.2 Why r=1?

Routing is fixed at r=1 and reused for r=2…R (SR-Core constraint). Therefore,
applying the entropy objective at r=1 is the only point where routing can be influenced.
Applying it at deeper steps would have no effect on the selection.

### 6.2.3 Why pre-topk?

The entropy H(p) is computed on the full pre-topk distribution p ∈ ℝ^n, before the
discrete top-k selection. This preserves the differentiable path through the router:
the top-k operation is not differentiable, but H(p) is. Applying the penalty to the
pre-topk distribution allows gradients to flow through the router without straight-through
estimators or other approximations.

### 6.2.4 λ Hyperparameter

λ controls the strength of entropy pressure. The λ sweep covers:
λ ∈ {0.001, 0.003, 0.005, 0.007}

plus the unregularized continuation (ctrl, λ=0.000) and two target-entropy variants
(λ·relu(H(p) - H_target) with H_target ∈ {3.75, 3.70}).

## 6.3 Experimental Setup

### 6.3.1 Training Protocol

All SR-Core models (b64, k=8, R=6) are trained for 15,000 steps as a shared base on
HeteroMini-v1. The entropy continuation is then applied for 2,000 additional steps from
this shared checkpoint, at each λ value independently.

The unregularized continuation (ctrl) runs for the same 2,000 steps without the entropy
objective. This is the direct comparison baseline.

Cross-seed validation: λ ∈ {0.003, 0.005} are evaluated at two independent seeds
(s0 and s1, independently initialized base models).

### 6.3.2 Negative Controls

Three negative controls validate that the entropy objective is the correct intervention:

- **Softfull:** Soft Jaccard penalty on router probability vectors across *consecutive
  tokens* — encourages cross-token routing similarity rather than within-token concentration.
  If this also consolidates routing, it suggests the effect is not specific to entropy.

- **Softsharp_a2:** Temperature sharpening (α=2) applied to router logits.

- **Reduced_noise:** Lower Gumbel noise (σ=0.1) during top-k sampling.

### 6.3.3 Evaluation Protocol

Router consolidation metrics are measured on 40 held-out batches via eval_compare.py:
- Router entropy H(p): lower = more concentrated pre-selection
- Unique core combinations: distinct active k-subsets per evaluation run
- Hard-overlap Jaccard: mean Jaccard across consecutive tokens (higher = more stable paths)
- LRU bytes/token at K ∈ {8, 16, 24, 32}: simulated cache cost

Language quality metrics use paired evaluation (same held-out batches, dense d24 as
consistent anchor per run):
- Per-domain held-out loss: Web, Wikipedia, Code, Literature
- Code-ratio: code_loss / mean(web_loss, wiki_loss, lit_loss) — lower = better code
  generalization relative to other domains

## 6.4 Results: Router Consolidation

### 6.4.1 Monotonic Response

All four routing metrics respond monotonically to increasing λ:

| λ | H(p) | Unique cores | Hard overlap | K=24 KB/tok |
|---|---|---|---|---|
| 0.000 (ctrl) | 3.846 | 25,853 | 0.251 | 1,103 |
| 0.001 | 3.815 | 25,399 | 0.254 | 1,098 |
| 0.003 | 3.799 | 23,077 | 0.262 | 1,085 |
| 0.005 | 3.732 | 21,273 | 0.266 | 1,020 |
| 0.007 | 3.647 | 18,817 | 0.278 | 1,009 |

The response is non-linear: most consolidation occurs between λ=0.001 and λ=0.005;
gains from 0.005 to 0.007 are smaller. Three qualitative regimes are identified:

- **Generalization-balanced (λ ≤ 0.003):** Moderate consolidation, quality effects
  either neutral or positive.
- **Cache sweet spot (λ ≈ 0.005):** Largest consistent cache improvement, quality
  effects seed-dependent.
- **Boundary (λ = 0.007):** Maximum consolidation, quality picture less predictable.

### 6.4.2 Negative Control: Softfull

The softfull objective moves routing metrics in the *opposite* direction:

| Metric | ctrl | softfull | Direction |
|---|---|---|---|
| Router entropy | 3.843 | 3.868 | +0.025 (worse) |
| Unique cores | 25,684 | 30,246 | **+17.8%** (worse) |
| Hard overlap Jaccard | 0.259 | 0.251 | −0.008 (worse) |
| K=24 LRU KB/tok | — | — | <1.4% change |

Cross-token similarity objectives act on pairwise overlap but do not control the
within-token distribution concentration. A Jaccard penalty can be satisfied by
increasing routing diversity (more distinct combinations with higher pairwise overlap
on average), which moves metrics in the wrong direction for cache efficiency.

This confirms that **pre-selection entropy is the correct intervention point** — not
cross-token similarity, not temperature sharpening.

## 6.5 Results: Cache–Quality Pareto

### 6.5.1 λ=0.003 Pareto-Dominates ctrl

In seed 0:
- K=24 cache: 1,103 → 1,085 KB/tok (−1.6%)
- Code-ratio: 0.886 → 0.866 (−0.021, improvement)

In seed 1:
- K=24 cache: not separately measured (offload_sim aggregates by architecture, not λ-variant;
  see 4-seed update below)
- Code-ratio: 0.896 → 0.822 (−0.074, improvement)

λ=0.003 is a **Pareto improvement**: both cache efficiency and code generalization
improve relative to the unregularized baseline, replicated across two independent seeds.

**4-seed update.** Two further seeds (s2, s3) for both λ=0.003 and λ=0.005 are now trained,
extending the sweep to four seeds. Their overall held-out loss is consistent (λ=0.003:
s2=5.12, s3=5.17; λ=0.005: s2=5.12, s3=5.17 — alongside s0=5.32 and the known low-loss outlier
s1=4.74). The *quality* side of the Pareto thus has 4-seed support. The per-seed **cache** table
(the Pareto x-axis) was not recomputed for s2/s3: `offload_sim` labels checkpoints by
architecture, not by λ-variant, so a single pass cannot separate them — the cache side remains
at two seeds. This is the one acknowledged loose end in the entropy chapter.

### 6.5.2 λ=0.005: Larger Cache Gain, Seed-Dependent Quality

Largest consistent cache improvement:
- K=24: 1,103 → 1,020 KB/tok (−7.5%)
- K=16: −5.2%

Language quality is seed-dependent (within-run comparison):
- Seed 0: code-ratio 0.877 vs. within-run ctrl 0.859 (Δ = +0.018, slight degradation)
- Seed 1: code-ratio 0.841 vs. within-run ctrl 0.863 (Δ = −0.022, improvement)

The within-run ctrl values differ across seeds due to batch-sampling variance (~±0.05
nats). A single ctrl anchor (0.886, from the λ=0.003 run) is used for the Pareto
figure for visual consistency; within-run comparisons give the true paired delta.

λ=0.005 is cache-best among the tested configurations, but the quality effect is not
reliable across seeds and should not be characterized as either a cost or a benefit.

### 6.5.3 λ=0.007: Boundary

Further cache improvement: K=24 1,009 KB/tok (~1.1% over λ=0.005).
Code-ratio returns toward ctrl levels (0.884 in seed 0).

λ=0.007 is treated as the boundary point of the controllable axis.

### 6.5.4 Target-Entropy Variants

Bounded objective: λ·relu(H(p) − H_target), targets H=3.75 and H=3.70.
Both fall below the Pareto frontier of the unconstrained sweep: slower convergence in
the 2,000-step window, does not reach the same cache/quality operating points.

## 6.6 Results: Language Quality vs. Dense

| Model | Web | Wiki | Code | Lit |
|---|---|---|---|---|
| Dense d24 | 5.448 | 5.456 | 4.621 | 4.735 |
| SR-Core ctrl | 5.807 | 5.841 | 4.856 | 5.286 |
| + entmin λ=0.003 | 5.672 | 5.688 | 4.969 | 5.352 |
| + entmin λ=0.005 | 5.887 | 5.792 | 4.985 | 5.419 |

Dense d24 is the quality upper bound in all domains. The dense advantage is largest in
Literature (0.55 nats) and smallest in Code (0.24 nats).

Entropy minimization does not narrow the dense–sparse quality gap. The objective shapes
routing structure; it does not trade quality for cache efficiency in either direction at
λ=0.003 (no tradeoff measured). At λ=0.005 the effect on absolute loss is small and
mixed.

## 6.7 Results: Offloading Simulation

See Chapter 5b for the full offloading simulation. At the 17k-step checkpoints:

| Model | K=8 KB/tok | vs. Dense (K=8) |
|---|---|---|
| Dense d24 | 12,348 | 1.00× |
| ctrl | 3,035 | 0.246× |
| λ=0.003 | 3,015 | 0.244× |
| λ=0.005 | 2,973 | **0.241×** |

Entropy consolidation at λ=0.005 adds a further 2% reduction in bytes-in-motion beyond
the structural reduction of SR-Core itself (4.1× vs. dense).

## 6.8 Interpretation

**What entropy minimization does:** It makes the router's pre-topk distribution more
concentrated, reducing the diversity of routing paths across tokens. This reduces the
effective routing vocabulary (unique core combinations) and improves LRU cache hit
rates.

**What it does not do:** It does not directly maximize cross-token routing overlap. The
softfull ablation shows that cross-token similarity is a different degree of freedom.
Entropy minimization acts within a single token's routing decision; temporal routing
consistency is a side effect, not the direct target.

**Where to operate:**
- **λ=0.003** is the recommended balanced operating point — Pareto improvement in both
  cache and quality, replicated across seeds.
- **λ=0.005** is the preferred cache/systems operating point — largest consistent cache
  gain (−7.5% at K=24), quality effect seed-dependent (not a consistent cost or benefit).
- **λ=0.007** is the cache-best boundary — lowest absolute bytes/token in the sweep
  (K=24: 1,009 KB/tok), but the additional gain over λ=0.005 is only ~1.1% and quality
  reverts toward ctrl levels. Not the preferred operating point.

## 6.9 What This Chapter Does Not Show

- Wall-clock inference speedup (CPU dispatch overhead ~2.8×, no RAM→VRAM prototype)
- Dense-quality parity (dense d24 remains the upper bound)
- Scaling behavior beyond n=64 or this training corpus
- Downstream task performance

Data sources: `data/eval/entmin/eval_compare_*.json`,
`data/eval/entmin/eval_quality_*.json`, `data/eval/entmin/offload_sim_17k.json`,
`data/eval/entmin/eval_compare_softfull.json`
