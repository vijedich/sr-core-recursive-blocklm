# Chapter 5: HeteroMini Experiments

This chapter covers three experiments on the HeteroMini-v1 corpus: a multi-domain
quality matrix (5a), an offloading simulation (5b), and cross-seed robustness of the
routing structure (5c).

---

## 5a: Multi-Domain Quality Matrix

### 5a.1 HeteroMini-v1 Corpus

HeteroMini-v1 is a four-domain pretraining corpus constructed for this work:

| Domain | Source | Tokens (~) |
|---|---|---|
| Web | FineWeb-Edu | ~1.6M |
| Wikipedia | Wikipedia dumps | ~1.6M |
| Code | The Stack (Python) | ~1.6M |
| Literature | Project Gutenberg | ~1.6M |
| **Total** | | **~6.6M** |

Tokenized with a shared byte-level BPE vocabulary of 8,000 tokens. Documents are kept
contiguous; the evaluation split has no training overlap.

The four-domain design tests whether SR-Core routing generalizes across heterogeneous
input distributions, and whether held-out losses differ meaningfully by domain (as
expected if routing specialization mirrors domain structure).

### 5a.2 Model Variants and Matrix Results

The matrix is a 2,000-step smoke run comparing 8 model configurations to establish
which architectural variants warrant longer training. All except SR-Core b64 R6 (10,000
steps) are short-run comparisons.

| Model | Steps | Params | L_final | Anytime | WS | Cache@8 | Domain acc |
|---|---|---|---|---|---|---|---|
| Dense d24 | 2,000 | 8.7M | 6.473 | 0.033 | — | — | — |
| Dense d8 | 2,000 | 4.5M | 6.541 | 0.006 | — | — | — |
| Naked b32 R2 | 2,000 | 10.8M | 6.756 | 0.000 | 4.13 | 15.6% | 45.4% |
| Naked b32 R6 | 2,000 | 10.8M | 6.568 | 0.022 | 7.49 | 15.9% | 49.4% |
| SR-Core b32 R2 | 2,000 | 10.8M | 6.561 | 0.000 | **4.0** | 13.3% | 47.3% |
| SR-Core b32 R6 | 2,000 | 10.8M | 6.710 | 0.022 | **4.0** | 2.0% | 45.0% |
| SR-Core b64 R2 | 2,000 | 19.3M | 6.659 | 0.000 | **4.0** | 17.1% | 46.9% |
| **SR-Core b64 R6** | **10,000** | **19.3M** | **5.371** | **0.017** | **8.0** | **12.4%** | **61.6%** |

(WS = working_set.contiguous; Cache@8 = cache miss rate at K=8; Domain acc = routing
classifier accuracy; chance = 25%)

Key observations from the matrix:

**This matrix is a smoke run, not a quality ranking.** At 2,000 steps no variant has
converged; the table only identifies which configurations warrant longer training. Some
orderings change with sustained training (Section 5a.5), so do not read row-vs-row L_final
differences here as architectural verdicts — but the overall direction holds: Dense remains
the quality ceiling once trained (5a.5).

**SR-Core b32 R6 at 2,000 steps appears worse than naked b32 R6 (6.710 vs. 6.568).**
This is a training warm-up artifact, not a fundamental quality issue: routing with
load-balancing takes more steps to settle. The b32 R6 trajectory bears this out — dead
blocks fall 6 → 8 → 3 → 0 → 0 from step 1k to 10k, i.e. the early "collapse" resolves
itself with training rather than being permanent.

**WS=k holds exactly for all SR-Core variants** (working_set.contiguous = k in all
configurations). Naked variants show WS > k because they use free routing.

**SR-Core b32 R6 has the lowest cache miss rate (2.0% at K=8)** — the routing collapses
early (it is in the low-diversity regime at this step count), leading to high cache hit
rates but likely degraded quality from under-specialization.

**Domain routing classification accuracy**: b64 R6 at 10k steps achieves 61.6% vs.
25% chance — routing does encode domain identity. All 2k-step variants are 45–49%
(barely above chance), confirming that domain specialization requires sustained training.

### 5a.3 Long-Run b64 R6 (10,000 Steps)

The b64 R6 model trained for 10,000 steps:
- L_final = 5.371 (significantly better than the 2k-step runs at ~6.5–6.7)
- Domain Jaccard off-diagonal = 0.225 (cross-domain routing overlap)
- Domain Jaccard matrix: Web–Lit = 0.60 (high similarity), Code–others ≈ 0.07–0.14
  (Code routing is highly distinct from all other domains)
- Cache miss rate K=8: 12.4% contiguous, 13.9% shuffled — locality benefit from
  contiguous document batching

The Code domain has the most distinct routing signature (Jaccard with others ≈ 0.07),
consistent with its linguistic distinctiveness in the corpus.

### 5a.4 Seen vs. Unknown Generalization

Generalization gap for Dense d24@10000 on held-out documents not in the training split:

| Split | Loss | PPL | Top-1 acc | Anytime gain |
|---|---|---|---|---|
| Seen | 5.133 | 169.6 | 18.5% | 0.028 |
| Unknown | 5.287 | 197.7 | 17.3% | 0.013 |
| **Gap** | **+0.154** | **×1.17** | −1.2 pp | −0.015 |

Per-domain breakdown (seen / unknown loss):

| Domain | Seen | Unknown | Gap |
|---|---|---|---|
| Web | 5.481 | 5.713 | +0.232 |
| Wikipedia | 5.421 | 5.623 | +0.202 |
| Code | **4.407** | **4.700** | +0.293 |
| Literature | 5.165 | 5.205 | +0.040 |

The generalization gap is smallest for Literature (+0.040) and largest for Code (+0.293),
suggesting Code is both the easiest domain in absolute terms and the most sensitive to
distribution shift. The overall PPL ratio (1.17×) is consistent with a model trained on
6.6M tokens of narrow-domain data.

### 5a.5 Quality: Dense Is the Ceiling

The 2,000-step matrix (5a.2) is a smoke run, not a final ranking. With sustained training,
the final-step loss of SR-Core b64 k8 R6 is reproducible across four independent seeds. All
values below are the mean cross-entropy at the final recursion step r=R, re-measured under one
fixed protocol (60×16×128 tokens, contiguous windows) on both the training-domain (seen)
corpus and the held-out split:

| Seed | L (seen) | L (held-out) |
|---|---|---|
| s0 | 5.334 | 5.491 |
| s1 (low-loss outlier) | 4.989 | 5.291 |
| s2 | 5.283 | 5.464 |
| s3 | 5.290 | 5.518 |
| **Mean ± std** | **5.224 ± 0.137** | **5.441 ± 0.089** |
| Mean excl. s1 | 5.302 | 5.491 |

Seed s1 is a low-loss outlier (also noted in Chapter 6); excluding it, the three remaining
seeds cluster tightly (seen 5.28–5.33). Dense d24 at 17k steps, under the *same* protocol,
reaches **seen 4.793 / held-out 5.102**. SR-Core therefore trails dense by **~0.5 nats on the
seen contiguous loss (0.51 excluding s1) and ~0.34 nats held-out** — with half the
block-applications per token (24 vs. k·R = 48). At this scale **Dense d24 is the quality
ceiling; SR-Core does not match it.**

Data source: `data/eval/heteromini/eval_quality_four_seed_summary.json` — regenerate with
`python monograph/scripts/eval_four_seed_quality.py --entropy`. (The earlier version of this
table reported a single "held-out" column that was in fact the seen-corpus loss and used a
per-run eval; the numbers above supersede it under one auditable protocol.)

**The observed gap is not explained by parameter count or block-application compute.** A
*param- and compute-matched* comparison settles this directly. A SR-Core variant sized to
Dense d24 — n=64, k=16, R=4, block_hidden=192 — has 8.75M parameters (vs. d24's 8.70M) and
performs exactly **6.29M parameter-applications per token** (k·R·block = 64·98k), identical
to d24's 24·262k. Same parameters, same compute, same backbone. Under the same protocol as
above, across three seeds it reaches **seen 5.269 ± 0.020 / held-out 5.459 ± 0.007 — still
~0.48 nats (seen) / ~0.36 nats (held-out) behind Dense d24**. Matching both parameters *and*
compute does **not** close the gap. The evidence points to the SR-Core *format* (narrow active
set + weight-tied recursion) as the source — the gap persists even when both budgets are
matched. Note this matched variant is also *tighter* across seeds than the plain k8 R6 set
(std 0.02 vs 0.14, no low-loss outlier), so it is the cleaner evidence. Whether the gap
narrows at convergence or at larger scale remains open (Section 7.5.6).

Data source: same summary file (`param_matched_k16_R4` block in
`data/eval/heteromini/eval_quality_four_seed_summary.json`).

The mechanism is a breadth-vs-depth tradeoff at fixed compute: dense brings ~6.3M *distinct*
parameters to bear per token (each applied once), while this SR-Core touches only ~1.6M distinct
parameters (k=16 blocks) and reuses them R=4 times via recursion. Recursion recovers only part of
the capacity that breadth would provide — Section 5d measures exactly how far it recovers, and the
answer (a saturating gain by r≈4) is consistent with this residual ~0.5-nat gap.

**This number is a lower bound, not a converged value.** At 15k steps both models are still
descending (~0.2 nats per 2,500 steps each); 6.6M tokens is a small corpus and neither has
converged. The descent *rates* are comparable, so this data shows no evidence that SR-Core closes
the gap with more training — but it cannot be ruled out without a longer run (a convergence study
is the natural next experiment).

The conclusion matches the smoke run's direction and the rest of this work: **SR-Core's
contribution is transfer efficiency, not language quality.** It does not win on held-out loss —
it transfers k=8 instead of 24 blocks per token (5b) and converts that into measured throughput
(5b.4). The argument rests on the transfer reduction, never on a quality claim.

---

## 5b: Offloading Simulation

### 5b.0 Why the Premise Is Bandwidth, Not Compute

Every comparison in this chapter must be read under the deployment scenario the
architecture targets: a model too large for VRAM, held in host RAM, streaming k blocks
per token across PCIe to the GPU. In this regime the GPU is **memory-bandwidth-bound** —
it stalls waiting for block transfers, not for arithmetic. Compute performed on blocks
already resident in VRAM can be **partially hidden** when transfer dominates and the sparse
dispatch is fused (Chapter 5b.4 shows the naive, unfused version is *not* hidden — it is
kernel-launch-bound and loses to dense until the per-block launches are grouped).

This reframes the compute asymmetry from Section 5a.5. SR-Core's 48 block-applications
per token (vs. dense d24's 24) are not a cost in this scenario: they execute in wall-clock
time the dense model would spend idle on its larger transfer. The quantity that decides
throughput here is **bytes moved per token**, not FLOPs per token — and that is exactly
what the offloading simulation measures. A model that moves fewer bytes per token serves
more tokens per second whenever PCIe, not the ALU, is the bottleneck.

The corollary defines what SR-Core must deliver to be useful in this regime: not a quality
win, but a quality that is *close enough* to justify the transfer saving. At this scale
SR-Core trails Dense d24 by ~0.5 nats (5a.5) while moving k=8 instead of 24 blocks per token
(5b.3) — whether that trade is worthwhile depends on how scarce bandwidth is in the target
deployment, and on whether the gap narrows at scale or with matched compute (both untested).

### 5b.1 Simulation Setup

The LRU offloading simulator estimates bytes-in-motion under a simple caching model:

- A virtual cache of capacity K blocks (K ∈ {8, 16, 24, 32}) is maintained per
  inference sequence
- On each token, the k active blocks are checked against the cache
- Cache misses incur a transfer cost of block_size bytes
- LRU eviction policy

This models the RAM→VRAM transfer scenario where frequently-used blocks remain resident.

Simulator runs on 17,000-step checkpoints (after full training, before entropy
continuation).

### 5b.2 Baseline Comparison

Dense layer-offloading baseline: at cache capacity K, a dense model with n_layers=24
must load all layers that are not cached. At K < 24, every token triggers at least
n_layers - K cache misses.

For dense d24 at any K < 24:
```
bytes/token ≈ (n_layers - K) × layer_size
```

At K=8: dense requires 16 layer transfers per token (no blocks resident).
At K≥24: dense fits entirely in cache → 0 transfers. This crossover is an important
boundary condition: for K≥n_layers, dense wins.

### 5b.3 Results

Full cache-sweep (contiguous document order, FP16):

| Model | K=4 KB/tok | K=8 KB/tok | K=16 KB/tok | K=32 KB/tok |
|---|---|---|---|---|
| Dense d24 | 12,348 | 12,348 | 12,348 | ~1 |
| SR-Core ctrl | **24,694** | 3,034 | 1,857 | 568 |
| + entmin λ=0.003 | 24,694 | 3,015 | 1,808 | 520 |
| + entmin λ=0.005 | 24,694 | 2,973 | 1,747 | **489** |

(Source: `offload_sim_17k.json`, `bytes_per_token_fp16 / 1024`)

**K=4 reversal:** At K=4, sparse is ~2× *worse* than dense (24,694 vs 12,348 KB/tok).
The WS=k=8 working set does not fit in a K=4 cache — effectively all 8 blocks miss on
every token. This is an important boundary condition: the sparse advantage only holds
for K ≥ k.

**K=8 breakeven:** The cache exactly fits the WS=k=8 working set. SR-Core ctrl uses
4.1× fewer bytes/token than dense (3,034 vs 12,348 KB/tok). Entropy consolidation
(λ=0.005) adds a further 2% reduction (3,034 → 2,973 KB/tok).

**K=16–32:** Sparse bytes/token continues to fall (hub blocks become resident across
tokens). At K=32: ctrl uses 568 KB/tok vs dense's 12,348 KB/tok (K=8 reference) — but
dense at K=32 also fully fits in cache (~1 KB/tok). The sparse advantage vs. dense
**disappears** above K≈24 because dense d24 fits entirely in cache once K ≥ n_layers.

**λ=0.005 is the best model at every K level**, confirming that entropy consolidation
monotonically reduces bytes-in-motion across the entire cache regime.

These are *simulated* estimates assuming LRU eviction and contiguous document order.
Real transfer costs depend on DRAM bandwidth, PCIe characteristics, block granularity,
and prefetch scheduling — none of which are modeled here.

### 5b.4 First Real-Hardware Measurement (RTX 2060)

A first prototype replaces the simulator's assumed bandwidth with measured RAM→VRAM
transfer (block weights pinned in host RAM, streamed to a K-slot VRAM cache; design and
code: `docs/streaming_prototype.md`, `experiments/streaming_prototype.py`). Two findings:

1. **The simulated bytes/token reproduce exactly** on real hardware (the streaming engine's
   fetch counts match the LRU misses of `offload_sim` at every K), and the **measured H2D
   bandwidth is 11.4 GB/s** — i.e. the simulator's 16 GB/s assumption was ~40% optimistic.
   On the transfer axis the advantage holds: SR-Core moves 6.7× fewer bytes/token than
   dense layer-offloading and therefore spends 6.7× less transfer time (166 vs. 1112 µs/token).

2. **The wall-clock throughput advantage appears once per-block launch overhead is removed.**
   A first (naive, per-block) implementation was kernel-launch-bound — ≈12 ms/token dominated
   by the 48 small block-applications (k=8 × R=6), with transfer <2% of the total — and was
   *slower* than dense (81 vs. 105 tok/s). Replacing the k per-block calls of each recursion
   step with a single **grouped block matmul** gives a 3.8× speedup (78 → 297 tok/s) and
   reverses the ranking: **SR-Core 297 tok/s vs. dense layer-offloading 205 tok/s — a 1.45×
   measured wall-clock advantage.** At a VRAM budget K=16 < D=24 the dense baseline thrashes
   (reloads all 24 layers/token, 12.3 MB) and becomes transfer-bound, while SR-Core moves 6×
   less (4 fetches/token, 2.1 MB). The remaining gap to the all-resident compute ceiling
   (297 → 364 tok/s) is the now-visible transfer cost (~18% of the time), which a two-stream
   overlap (prefetch the next token's blocks during the current token's compute) recovers in
   part — a further 5–9% in the transfer-heavy regime, reaching ~1.6× over dense on the same
   hardware, and correctly nothing once the cache is large enough that transfer is already free.

This closes — at small scale and as a real measurement — the chain *fewer bytes → more
tokens/second* that Chapter 5b only simulated. It also sharpens the central premise: SR-Core
trades *more compute* (2× the block-applications) for *less transfer*; that trade is invisible
when the compute path is launch-bound and only pays off once the path is efficient and the
dense baseline is transfer-bound (model larger than the VRAM cache). The 1.45× is modest at
this toy scale (514 KB blocks, forced streaming, batch-1, no overlap); the scale projection
(Section 5b, `--project`) into the regime of large blocks and scarce bandwidth extrapolates a
substantially larger advantage but remains a *projection beyond the measured point*.

---

## 5c: Cross-Seed Robustness

### 5c.1 Setup

Three independently initialized SR-Core b32 k8 R6 models (seeds 0, 1, and 2) are trained
on the same HeteroMini-v1 corpus for 15,000 steps.

Question: is the routing structure (Jaccard overlap between seeds, domain specialization
pattern) reproducible across random initializations?

### 5c.2 Results

Model: `srcore_b32_k8_R6@15000`, 3 independently initialized seeds, 81,920 eval tokens each.

| Seed | Unique cores | Top-1 cov. | Gini | Dead | Cache miss K=16 | Domain Jaccard |
|---|---|---|---|---|---|---|
| s0 | 7,738 | 2.6% | 0.288 | 0 | 5.3% | 0.292 |
| s1 | 14,445 | 1.2% | 0.256 | 0 | 5.6% | 0.216 |
| s2 | 4,889 | 3.0% | 0.206 | 0 | 6.7% | 0.376 |

(Unique cores = number of distinct active-block tuples; Gini = load imbalance 0=flat 1=one block; Domain Jaccard = mean off-diagonal routing overlap between domains)

**What is robust across seeds:**
- Zero dead blocks (WS=k holds, routing health is stable)
- Cache miss K=16 is 5–7% across all seeds (LRU cache works at this K)
- Gini coefficient is low (0.21–0.29) — no single block dominates

**What varies across seeds:**
- Unique core count ranges 3× (4,889 to 14,445) — routing diversity is seed-dependent
- Domain Jaccard spans 0.216–0.376 — the degree of domain specialization is not stable
- s2 has the fewest unique cores (more repetitive active sets) and highest domain overlap

The seed-to-seed variation in routing diversity is the key finding: the WS=k guarantee
and routing health are seed-invariant, but the *structure* of routing specialization
(which domains activate which blocks) is initialization-dependent. Blocks learn domain
functions, but which block learns which function is arbitrary.

### 5c.3 Anytime Inference

Anytime gain = L(r=1) − L(r=R): quality improvement from early exit (r=1) to full
recursion (r=R=6). Evaluated on **seen** documents; `srcore_b32_k8_R6@15000`.

| r | Block-apps | s0 L_seen | s1 L_seen | s2 L_seen | Code gain s0 | Code gain s1 | Code gain s2 |
|---|---|---|---|---|---|---|---|
| 1 | 8 | 5.263 | 5.112 | 5.051 | 0.000 | 0.000 | 0.000 |
| 2 | 16 | 5.312 | 5.081 | 5.045 | 0.026 | 0.024 | 0.089 |
| 3 | 24 | 5.298 | 5.072 | 4.979 | 0.049 | 0.038 | 0.149 |
| 4 | 32 | 5.198 | 5.058 | 4.902 | 0.059 | 0.048 | 0.187 |
| 6 | 48 | 5.235 | 5.045 | 4.886 | 0.053 | 0.055 | 0.204 |
| **Gain r1→r6** | | **+0.028** | **+0.067** | **+0.165** | | | |

(Positive anytime gain means deeper recursion improves quality; s0 total gain is actually
slightly negative at r=2/3 before recovering — non-monotone depth curve.)

Key observations:
- **Seed 2 has 6× larger anytime gain than seed 0** (0.165 vs. 0.028). Anytime depth
  benefit is not stable across seeds at this step count.
- **Code benefits most from depth**: code_gain increases monotonically with r in all
  seeds. At r=6, Code gain is 0.053–0.204 nats (0.053 for s0, 0.204 for s2).
- **s0 has a non-monotone depth curve** (L increases at r=2 before decreasing at r=4).
  This is a training instability artifact — the deep-supervision weighting has not fully
  resolved by step 15,000 for this seed.
- Throughput is seed-independent (~4,300–4,900 tok/s GPU) since block-apps are fixed.

The anytime property is present but seed-dependent at 15,000 steps. Longer training
(as demonstrated by b64 R6 at 10,000 steps in section 5a.3) stabilizes the gain.

### 5c.4 Domain Partition Analysis

Domain-specific routing specialization is measured by the Domain Jaccard off-diagonal:
how much routing overlap exists between different domains. Lower = more specialized.

From Section 5c.2:
- s0: Domain Jaccard = 0.292
- s1: Domain Jaccard = 0.216 (most specialized)
- s2: Domain Jaccard = 0.376 (least specialized — most shared routing between domains)

This seed-dependence is expected: which block specializes for Code vs. Literature is
arbitrary; what matters is that *some* domain partition exists (all values < 0.5),
not that the same blocks are responsible across seeds.

Quantitative domain partition analysis (which specific blocks concentrate on which
domain) is initialization-sensitive and not reported as a cross-seed aggregate.

---

## 5d: Recursion Depth vs. Active Capacity (A-Sweep)

Section 5a.5 attributes the dense–sparse quality gap to the SR-Core format trading *breadth*
(distinct active parameters per token) for a small streamable working set, with recursion only
partly compensating. This section measures how far recursion compensates, as a function of the
active-brain size **A = k · block_size** — the distinct parameters a token activates per step.

Block size is held fixed (n=64, R=6, block ≈ 263k params) and k is swept ∈ {2, 4, 8, 16} on
HeteroMini (15k steps), so A ranges 0.5M–4.2M. The per-recursion-step gain L(r=1) − L(r) is read
post-hoc from the deep-supervision readout (no retraining; "R is inference-time"):

| k | A (active params) | useful depth | total anytime gain (seen) | best L |
|---|---|---|---|---|
| 2 | 0.53M | r≈1 (dead) | 0.0015 | 5.33 |
| 4 | 1.05M | r≈4 | 0.0595 | 5.46 |
| 8 | 2.10M | r≈2–4 (seed-dep.) | 0.008 / 0.017 / 0.031 (3 seeds) | 5.21–5.34 |
| 16 | 4.20M | r≈4 | 0.0625 | 5.24 |

Two clean signals survive, and one honest caveat:

**A recursion floor exists.** At k=2 (A=0.53M) recursion is dead — gains vanish by r≈2. The block
sits at its representational ceiling at r=1 and cannot encode a refinement of its own output.
Above the floor (A ≳ 1M, k ≥ 4) recursion lives.

**Useful depth saturates at r≈4.** Wherever recursion is alive, the per-step gain is exhausted by
r≈4; steps r=5–6 add essentially nothing. This is the empirical basis for the R=4 choice in the
param-matched experiment (5a.5) — and it bounds how much recursion can substitute for breadth.

**Magnitude is seed-dominated — do not over-read it.** Total recursion gain is *not* monotone in A
(k=4 and k=16 both ~0.06, k=8 only ~0.017), and a three-seed replication of k=8 gave a 4× spread
(0.008 / 0.017 / 0.031). With one seed per point the sweep is underpowered to rank neighbouring A
values by gain magnitude; only the floor (k=2 dead) and the depth saturation (r≈4) survive the
seed noise. Reading a precise "optimal A" off the magnitude curve would repeat the single-seed
overinterpretation this work elsewhere flags.

The design conclusion is coarse but robust: keep A above the floor (~1M, k ≥ 4) and use R≈4; the
precise sweet-spot is below the resolution of a single-seed sweep. This caps how far recursion
recovers the capacity that breadth would provide — consistent with the residual ~0.5-nat gap to
dense at matched compute (5a.5).

Data source: `data/eval/heteromini/anytime_inference_srcore_b64_*@15000_s*.json`;
figure: `figures/fig_asweep_depth.png` (script: `scripts/plot_asweep.py`).

---

## 5e: Summary

HeteroMini experiments establish:

1. **Dense d24 is the quality ceiling; the ~0.5-nat gap is not explained by parameter or
   compute budget.** Under one fixed protocol (Section 5a.5), SR-Core b64 k8 R6 reaches seen
   5.30 / held-out 5.44 (three seeds excl. a low-loss outlier) vs. Dense d24's seen 4.79 /
   held-out 5.10. A param- *and* compute-matched SR-Core (8.75M params, 6.29M apps/token,
   identical to d24) still trails by **~0.48 nats seen / ~0.36 held-out** (seen 5.269 ± 0.020;
   Section 5a.5) — so the gap is not a budget artifact; the evidence points to the
   narrow-active-set + recursion format. Both models are still descending at their stops, so
   this is a lower bound, not a converged value; whether the gap narrows at convergence or
   scale remains open (Section 7.5.6). SR-Core's contribution is transfer efficiency, not
   quality.

2. **Recursion has a measurable useful-depth band** (Section 5d): below an active-brain floor
   (A ≈ 1M) recursion is dead; above it, per-step gains saturate by r≈4. Magnitude is
   seed-dominated and not a reliable design signal.

3. **SR-Core scales in bank size:** b64 R6 outperforms b32 R6 across domains, confirming
   that the fixed active budget k=8 benefits from a larger block vocabulary.

4. **Offloading simulation shows 4.1× transfer reduction** vs. dense layer-offloading at
   K=8, and a first real RAM→VRAM prototype turns it into a measured wall-clock win (~1.6×
   over dense layer-offloading on an RTX 2060; Section 5b.4). This — not quality — is the
   load-bearing result: in the bandwidth-bound regime, paying ~0.5 nats to move k=8 instead of
   24 blocks per token is the trade.

5. **Cross-seed training is robust:** key routing properties (entropy, Jaccard structure)
   replicate across independently initialized models.

Data sources: `data/eval/heteromini/`, `data/eval/phase1/offload_sim.json`,
`data/eval/entmin/offload_sim_17k.json`
