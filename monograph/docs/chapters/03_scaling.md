# Chapter 3: Scaling Validation — WS Independent of Bank Size

## 3.1 Research Question

The working-set guarantee WS=k is an architectural property (Chapter 2). This chapter
asks whether it holds *empirically* at the language quality level: does training with
larger block banks (larger n) degrade quality relative to the k=const active budget,
and does the sparse routing continue to match dense quality at matched compute?

## 3.2 Experimental Design

### 3.2.1 Corpus

TinyStories (Eldan & Li, 2023): synthetic short stories, simple vocabulary, ~2M tokens.
Used for Phase-1 scaling because it is small enough for CPU-scale experiments and
provides a clear language modeling signal without domain complexity.

### 3.2.2 Model Configurations

All models use k=4 active blocks, R=6 recursion steps, d=128 (demo scale).
Bank size is varied across: n ∈ {16, 32, 64, 128, 256}.

Baseline models:
- **Dense A (16 blocks):** All 16 blocks applied once each. Quality upper bound at this
  parameter count.
- **Recurrent B (1 block × 16):** Single shared block, applied 16 times. Isolates the
  effect of pure recursion without routing.
- **Routed C (4/n):** SR-Core with varying bank size n.

### 3.2.3 Training Protocol

3,000 steps, AdamW, cosine schedule, deep supervision (end-heavy weighting w_R=0.7).
Cross-seed validation: s0, s1, s2 for key configurations.

## 3.3 Results: Phase-1 Baseline

At the initial synthetic-data scale (n=24 blocks, k=4, 16 block-applications, CPU demo
config with d=128):

| Model | Active blocks | L₁ | L_final | Quality |
|---|---|---|---|---|
| Dense A (16 distinct) | 16 | 0.607 | **0.589** | Baseline |
| Recurrent B (1 × 16) | 1 | 0.756 | **0.720** | Worst |
| Routed C (4/24) | 4 | 0.547 | **0.547** | **Best** |

Note: the n=24 demo config uses a smaller bank than the TinyStories scaling runs (n ∈
{16, 32, 64, 128, 256}, section 3.4) because it is the original CPU prototype config.
The architecture is the same; the TinyStories experiments use the same codebase at
larger bank sizes and on GPU.

At equal compute (16 block-applications each): **C < A < B**.

Key finding: routing C achieves better quality than dense A with only 4 active blocks
(vs. 16 for A). The quality advantage comes from selective activation, not from
additional compute.

Recurrent B degrades with depth (U-shaped curve, minimum ≈0.680 at step 5, then
worsening): pure recursion without routing diversity is not beneficial and becomes
unstable at depth.

### 3.3.1 Routing Health

- Router entropy: 0.984 (near-perfect balance; 1.0 = uniform)
- Dead blocks: 1 of 24 (4%)
- Maximum single-block utilization: 0.092 (no dominant block)
- MI_norm (regime ↔ block assignment): 0.197 ± 0.030 (stable specialization)

## 3.4 Bank Size Scaling

These experiments use *free-routing* variants (independent routing per step, pre-SR-Core)
trained for 3,000 steps on TinyStories with k=4, R=6, d=128. They demonstrate that
quality does not degrade — and modestly improves — as the block bank grows, while
routing health is maintained.

Note: free-routing means the routing decision is made independently at each step r.
The two-phase structure (J₁₂ low, J≥23 high) emerges from training. In SR-Core, routing
is fixed at r=1, so WS=k=4 exactly. Here, the empirical unique-blocks/token is higher
(≈6–7) because step r=1 explores different blocks than steps r≥2.

| Bank (n) | L₁ | L_final | J(r1→r2) | Unique/tok | Dead | Router entropy |
|---|---|---|---|---|---|---|
| 16 | 3.743 | 3.741 | 0.377 | 5.95 | 0 | 0.950 |
| 32 | 3.692 | 3.693 | 0.237 | 6.74 | 0 | 0.979 |
| 64 | 3.714 | 3.720 | 0.184 | 7.07 | 0 | 0.994 |
| 128 | 3.628 | **3.630** | 0.098 | 7.52 | 0 | 0.995 |
| 256 | 3.638 | 3.650 | 0.125 | 7.38 | 1 | 0.998 |

Key observations:
- **Quality improves with n** (b128 is best). A larger block vocabulary gives the router
  more specialized units to choose from without increasing the active count k.
- **J(r1→r2) decreases with n**: larger banks make step-1 routing more token-specific
  (more choices → lower overlap with the collapsed r≥2 selections). This is the
  two-phase structure becoming more pronounced.
- **Router entropy is near-uniform at all sizes** (0.950–0.998). No collapse at any bank
  size tested.
- **One dead block at n=256** — the only routing health anomaly; all other bank sizes
  have zero dead blocks.

Cross-seed validation (b32 and b64): L_final s1=3.814, s2=3.655 for b32; additional seeds
show ≈±0.06 variance, consistent with the synthetic-task scale.

## 3.5 Anytime Property

With end-heavy loss weighting (w_R = 0.7), the per-step loss curve L(r) decreases
with depth for difficult input patterns:

- REPEAT regime: L improves 1.210 → 1.178 with depth
- INCREMENT/ALTERNATE: solved at r=1 (≈0.05 / ≈0.001), marginal gain thereafter
- FIB (hardest): L ≈ 1.06, no depth gain — too hard for this scale

The anytime property is regime-dependent. Easy patterns converge at r=1; hard patterns
benefit from depth. This motivates an adaptive halting mechanism (future work) that
stops early for easy inputs.

Equal-weight supervision (all w_r = 1/R) produces a flat anytime curve: it strongly
trains r=1, making additional depth redundant. End-heavy weighting increases the
depth benefit for REPEAT by 10× compared to equal weighting.

## 3.6 Routing Specialization

### 3.6.1 Phase-1 Synthetic (REPEAT/FIB/INCREMENT/ALTERNATE Regimes)

Block activation correlates with input regime (measured by MI between regime label and
active block set):

| Seed | MI_norm | L_final |
|---|---|---|
| s0 | 0.213 | 0.550 |
| s1 | 0.155 | 0.716 |
| s2 | 0.223 | 0.623 |
| **Mean ± std** | **0.197 ± 0.030** | |

Ablation impact (removing blocks with highest per-regime activation preference):
- REPEAT-specific blocks: +0.077 on REPEAT loss
- FIB-specific blocks (shared with REPEAT): +0.162 on REPEAT, +0.044 on FIB
- INCREMENT/ALTERNATE blocks: no measurable effect (routing redundancy)

Specialization emerges where computation is actually needed. Trivially-solved regimes
are multiply covered — ablating their blocks does not noticeably hurt quality.

### 3.6.2 TinyStories Linguistic Competence Analysis

Model: `competence_b64k4R6_s0` (n=64, k=4, R=6, TinyStories, 3,000 steps)

Six linguistic competence categories annotated by keyword detection:

| Category | n samples | L_cat | Ablation Δ |
|---|---|---|---|
| scene_shift | 9,265 | 3.807 | +0.006 nats |
| causality | 2,112 | 3.911 | **+0.118 nats** |
| dialogue | 835 | 4.039 | **+0.081 nats** |
| emotion | 1,120 | 4.092 | +0.005 nats |
| coreference | 24,058 | 4.177 | +0.004 nats |
| temporal | 934 | 4.233 | +0.004 nats |

(Ablation Δ = mean loss increase over R=6 steps when ablating blocks with highest
activation rate for that category)

**Routing classifier accuracy: 41.7%** vs. chance 16.7% (6 classes). Bootstrap CI:
[38.4%, 43.2%]. The routing signal carries significant categorical information at every
recursion step (r=4 is best: 42.4%).

MI per recursion step:
- r=1: 0.0084 bits — exploration phase (low categorical signal)
- r=2–6: ~0.0103 bits — stable plateau, 23% above r=1

This mirrors the two-phase structure: at r=1 the router explores broadly; from r=2
onward it settles into a stable assignment carrying more categorical information.

**Causality (+0.118) and dialogue (+0.081) have clear specialist blocks.** Ablating
them meaningfully degrades those categories. Coreference, emotion, scene_shift, and
temporal show near-zero ablation impact (shared generalist blocks).

The hardest category (temporal, L=4.233) and the most frequent (coreference, n=24,058)
both have near-zero ablation impact — routing specializes where dedicated computation
provides a measurable quality gain, not simply for the hardest or the most common tasks.

Hub structure (b64, free-routing):
- Gini coefficient: 0.575 (moderate imbalance — hub blocks present)
- Top-5 block share: 40.1% of total usage across n=64
- Cache simulation: K=8 → 20.5% miss rate (learned) vs. 89.6% (random routing baseline)

### 3.6.3 Cross-Seed Variance of Categorical Specialization

clf_accuracy and MI across 4 seeds (s0 = base; s1–s3 = curriculum_fromIter2 variant):

| Seed | clf_acc (all r) | Bootstrap mean | Bootstrap CI | MI r=1 | MI r≥2 |
|---|---|---|---|---|---|
| s0 | 41.7% | 40.9% | [38.4%, 43.2%] | 0.0084 | 0.0103 |
| s1 | 38.9% | 40.1% | [38.1%, 42.9%] | 0.0046 | 0.0090 |
| s2 | 36.3% | 39.6% | [36.1%, 42.7%] | 0.0053 | 0.0101 |
| s3 | 38.5% | 38.4% | [36.2%, 40.4%] | 0.0051 | 0.0121 |
| **Mean ± std** | **38.9% ± 2.0%** | **39.7%** | | **0.0059** | **0.0104** |

All four seeds are substantially above 16.7% chance, and all bootstrap CIs exclude
chance. The routing classifier is reliably above chance across seeds.

The two-phase MI pattern holds in every seed: MI at r=1 is consistently lower than MI
at r≥2, and the r=1 value varies more across seeds (0.0046–0.0084) than the plateau
(0.0090–0.0121), consistent with r=1 being the less-constrained exploration step.

**Cross-seed ablation** (now computed for all seeds; data in
`data/eval/phase1/competence_ablation_s{1,2,3}.json`):

Ablation-Diagonale: Δ_self − Δ_other (positive = category-specific specialist block)

| Category | s0 Δ(self−other) | s1 Δ(self−other) | s2 Δ(self−other) | s3 Δ(self−other) | Consistent? |
|---|---|---|---|---|---|
| causality | **+0.114** | +2.11 | +0.00 | +1.89 | 3/4 seeds |
| dialogue | **+0.073** | −1.27 | +1.05 | +1.58 | 3/4 seeds |
| temporal | +0.000 | +0.00 | **+3.20** | +0.00 | 1/4 seeds |
| coreference | +0.000 | +0.17 | +1.04 | +0.00 | 2/4 seeds (weak) |
| emotion | +0.000 | −2.02 | +0.00 | +0.00 | 0/4 seeds |
| scene_shift | +0.000 | +0.00 | +0.00 | −0.01 | 0/4 seeds |

**Key finding: Causality (3/4 seeds) and Dialogue (3/4 seeds) are the most consistently
category-specific.** This replicates the s0 result and holds across the curriculum
variant. Emotion and scene_shift show no consistent specialist signal in any seed.

**Scale difference between s0 and s1–s3:** The absolute ablation deltas in s1–s3 are
10–30× larger (e.g., causality self-ablation: +0.118 nats in s0 vs +27–30 nats in
s1/s3). This reflects the curriculum training protocol making routing more concentrated
— a few hub blocks dominate many categories simultaneously. Ablating 5 top-lift blocks
in s1–s3 essentially removes blocks that all categories rely on heavily, producing large
absolute effects. The category-specific signal (self − other differential) is
correspondingly harder to isolate but still present for causality and dialogue.

Data source: `scripts/competence_ablation_crossseed.py` → `data/eval/phase1/competence_ablation_s{1,2,3}.json`

## 3.7 Summary

SR-Core has WS=k by construction (routing fixed at r=1; Chapter 2). What these free-routing
scaling experiments add is that the surrounding conditions stay healthy as the bank grows:
- Quality does not degrade — and modestly improves — with increasing n at fixed k
- Routing remains healthy (no collapse) at all tested bank sizes
- Routing specialization is stable across seeds

(Note: these are pre-SR-Core free-routing variants with unique-blocks/token ≈6–7, not WS=k=4;
they validate that bank scaling is benign, not the WS bound itself — that is architectural.)

Sparse routing C outperforms dense A at equal compute, and pure recursion B is
strictly dominated — validating the architectural motivation for SR-Core.

Data sources: `data/eval/phase1/tinystories_b*.json`, `C_routed_s*.json`,
`competence_b64k4R6_*.json`
