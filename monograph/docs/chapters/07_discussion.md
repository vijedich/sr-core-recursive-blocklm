# Chapter 7: Discussion and Future Work

## 7.1 What This Work Demonstrates

This monograph has investigated SR-Core, a recursive block-sparse language model
architecture with a hard working-set guarantee (WS=k), and characterized the effect
of entropy-based router consolidation on cache efficiency and language quality.

### Demonstrated properties

**Architectural:**
- The WS=k guarantee holds by construction (Chapter 2) and empirically at all tested
  bank sizes n ∈ {16, 32, 64, 128, 256} on TinyStories (Chapter 3)
- Sparse routing (k=4 of n) outperforms dense execution (16/16) at equal compute on
  synthetic tasks; pure recursion without routing is strictly dominated
- Block specialization is reproducible (MI_norm = 0.197 ± 0.030 over 3 seeds)

**Routing structure:**
- Two-phase routing emerges in free-routing variants: exploration at r=1 (J₁ ≈ 0.20),
  collapse at r≥2 (J ≈ 0.984)
- Entropy-based consolidation induces a controllable axis: increasing λ monotonically
  reduces routing entropy, unique core combinations, and simulated bytes-in-motion
- Pre-selection entropy is the correct intervention point (softfull negative control
  moves in the wrong direction)

**Cache efficiency:**
- λ=0.003 is a Pareto improvement over unregularized baseline: cache efficiency and
  code generalization both improve, replicated across two independent seeds
- At K=8, SR-Core requires 4.1× fewer simulated bytes/token than dense layer-offloading

**Quality:**
- Dense d24 is the quality ceiling. SR-Core b64 k8 R6 reaches L = 5.30 ± 0.05 (stable across
  4 seeds), ~0.5–0.6 nats behind Dense d24 (~4.70–4.81) — and Dense achieves this with half
  the block-applications per token (24 vs. 48). SR-Core does not win on language quality
  (Chapter 5a.5)
- The gap is not explained by parameter count or block-application compute in the tested
  regime: a param- *and* compute-matched SR-Core (8.75M params, 6.29M apps/token = Dense d24)
  still trails by ~0.5 nats (Chapter 5a.5). The evidence points to the narrow-active-set +
  recursion format as the source. Whether the gap narrows at convergence or scale remains open
  (Section 7.5.6). (Both models are still descending at 15k, so the figure is a lower bound,
  not a converged value)
- This is by design, not a shortfall: SR-Core's contribution is transfer efficiency, not
  quality. Additional block compute is not the limiting term in the transfer-bound deployment
  regime — provided sparse dispatch is fused enough that per-block kernel-launch overhead is
  sub-dominant (achieved in Chapter 5b.4 via grouped matmul). The quality gap is the price
  paid for moving k=8 instead of 24 blocks per token
- λ=0.003 entropy consolidation does not degrade quality relative to ctrl

## 7.2 What This Work Does Not Demonstrate

**Wall-clock inference speedup at scale.** A real RAM→VRAM prototype (Chapter 5b.4) now
demonstrates the wall-clock advantage *at small scale*: with a grouped block matmul (removing
the per-block kernel-launch overhead that made a naive implementation launch-bound), SR-Core
reaches 297 tok/s vs. 205 tok/s for dense layer-offloading at a VRAM budget below the dense
model size — a 1.45× measured advantage on an RTX 2060. What remains undemonstrated is that
this *grows* at deployment scale (large blocks, model ≫ VRAM, where the projection predicts a
much larger gap), under asynchronous stream overlap, and in batched (not batch-1) serving.
The mechanism is now measured; its magnitude at scale is still a projection.

**Dense-quality parity.** Entropy minimization shapes routing structure; it is not a
path toward closing the quality gap with dense models.

**Large-scale validation.** All experiments use a ~19M parameter model on a ~6.6M token
corpus. Whether the WS=k guarantee retains its cache-efficiency advantage at 7B or 70B
parameters (where n ≫ 64) is untested.

**Downstream task evaluation.** All evaluation is on held-out language modeling loss
(next-token cross-entropy). No downstream task (classification, generation quality,
reasoning) has been tested.

**Leiterbahn index.** The routing trace analysis sufficient to build a block-prefetch
index has not been implemented. The Leiterbahn concept (Section 10 of Theorie.md)
remains a design direction, not an experimental result.

## 7.3 Relationship to the Original Research Plan

The original research plan (Theorie.md) envisioned a 7-phase program:

| Phase | Plan | Status |
|---|---|---|
| 1 | Feasibility, tunnel emergence, anytime property | Complete |
| 2 | Harder routing (Gumbel, temperature curriculum) | Partial (top-k with noise) |
| 3 | 3D spatial topology, trainable coordinates | Not implemented |
| 4 | Leiterbahn index | Not implemented |
| 5 | I/O loss, hardware-cost simulation | LRU simulation + real prototype (Ch. 5b.4) |
| 6 | Dynamic halting | Not implemented |
| 7 | Real RAM→VRAM streaming | Implemented at small scale (Ch. 5b.4); deployment-scale pending |

The work presented here covers Phase 1 (fully), a component of Phase 2 (entropy
regularization as an alternative to temperature curriculum), and Phase 5 (LRU simulation
plus a first real-hardware RAM→VRAM prototype at small scale). The 3D topology and
Leiterbahn concepts remain important directions but were not the focus of this
experimental program.

The entropy regularization approach (Chapter 6) was not part of the original plan; it
emerged from the observation that routing collapse at deep steps was a training artifact,
and that the pre-topk entropy was the appropriate target for intervention.

## 7.4 The Most Important Open Question

The fundamental question was whether the transfer reduction demonstrated at simulation
level (4.1×) survives real hardware measurement as a wall-clock **throughput** gain.

A real RAM→VRAM prototype (Chapter 5b.4) has now answered this in the affirmative at small
scale. On the *transfer axis*: at the measured H2D bandwidth (~11 GB/s on an RTX 2060)
SR-Core moves 6.7× fewer bytes/token than dense layer-offloading, and the simulated
bytes/token reproduce exactly. On the *wall-clock axis*: a first naive implementation was
kernel-launch-bound and slower than dense, but once the k per-block calls of each recursion
step are fused into one grouped matmul, SR-Core reaches 297 vs. 205 tok/s — a 1.45× measured
throughput advantage, with the dense baseline transfer-bound (it reloads all layers per token
at a VRAM budget below its size) and SR-Core moving 6× less.

The chain *fewer bytes → more tokens/second* is therefore no longer only simulated; it is
measured end-to-end on real hardware. The open question that remains is one of *magnitude*,
not existence: does the advantage grow at deployment scale (large blocks, model ≫ VRAM, where
the projection predicts far more than 1.45×), under asynchronous transfer/compute overlap, and
in batched serving? The mechanism is demonstrated; its payoff at scale is the remaining work.

## 7.5 Future Directions

The open questions below are not presented as limitations to apologize for. Each is a
discrete, well-scoped experiment that a group with the right hardware or the right
curiosity can pick up where this program stopped.

| Open question | Minimal next test | What would it show? |
|---|---|---|
| Quality gap convergence | Train Dense d24 and matched SR-Core to ~40k steps from 15k snapshots (~6–8 h on an RTX 2060) | Gap stays / narrows / widens at convergence |
| Deployment-scale streaming | Measure wall-clock throughput with a block bank that genuinely exceeds VRAM by 4–10× | Whether the small-scale 1.45–1.6× advantage grows, saturates, or is eaten by batching overhead |
| Batched serving | Measure union-of-routes at batch size > 1 | Whether the working-set bound survives or the effective active set grows to dense size |
| Leiterbahn index | Record routing traces for 10k+ sequences; cluster by path; test prefetch hit rate | Whether routing structure is predictable enough to prefetch the next token's blocks before they are needed |
| Attention blocks | Replace MLP blocks with attention-like blocks, retrain | Whether higher FLOP intensity per loaded byte translates to larger throughput advantage |

### 7.5.1 Deployment-Scale Streaming

The RAM→VRAM prototype exists (Chapter 5b.4) and has cleared three milestones: the real
transfer measurement; the grouped block matmul that removed per-block kernel-launch overhead
(with which SR-Core surpasses dense layer-offloading, 1.45× at small scale); and asynchronous
two-stream overlap (prefetch the next token's blocks during the current token's compute),
which adds a further 5–9% in the transfer-heavy regime (small VRAM cache) and, as expected,
nothing when the cache is large enough that transfer is already free. The end-to-end measured
advantage on an RTX 2060 reaches ~1.6× with overlap. The remaining steps: (1) batched serving
rather than batch-1 (the union-of-routes problem — the active block *union* across batch items
widens the effective working set, potentially reducing the k/n advantage); (2) measurement at
deployment scale (large blocks, model ≫ VRAM) where the projection predicts the advantage
grows well beyond the small-scale figure, and where the compute path is FLOP-bound rather than
partly launch-bound, so overlap should hide a larger transfer fraction.

Hardware target: GPU with ≤8 GB VRAM, ≥64 GB CPU RAM, where the dense 7B+ model
does not fit in VRAM but the k=8 active blocks of SR-Core do.

> **Continuation spec:** Measure wall-clock throughput at deployment scale (model whose
> total weight bank exceeds VRAM by 4–10×, large blocks). Separately, measure batch>1
> throughput to quantify the union-of-routes effect. Expected outcome range: advantage
> grows substantially (transfer-dominated, FLOP-efficient path) or is eaten by batching
> (union collapses effective sparsity). Either result is informative.

### 7.5.2 Bank Size Scaling

The 4.1× transfer reduction at n=64 benefits from the fact that k/n = 8/64 = 12.5%.
At n=512 with k=8, the fraction drops to 1.6% — a 63× theoretical reduction vs. dense.
Whether SR-Core training remains stable at n=512 and whether the quality gap to dense
remains manageable at that scale is a critical open question.

> **Continuation spec:** Train SR-Core at n=128 and n=512 on the same HeteroMini-v1
> corpus with the same hyperparameters. Measure held-out loss and routing entropy collapse.
> If training is stable, measure offload bytes/token to confirm the theoretical k/n scaling.

### 7.5.3 3D Topology and Leiterbahn

If routing paths are reproducible (MI_norm > 0.197, routing structure replicates across
seeds), then routing trace analysis on a large validation set should reveal structure
suitable for Leiterbahn indexing. The original plan describes this step in detail. It
requires:

1. Routing trace recording on 10,000+ validation sequences
2. Tunnel clustering by entry region and path similarity
3. Index construction mapping entry signatures to block prefetch plans
4. Prefetch-precision measurement on held-out sequences

> **Continuation spec:** Run routing trace collection on the existing 15k checkpoints
> (s0–s3) against the full HeteroMini-v1 validation set. Compute per-token block
> prediction accuracy from the index (does the correct block appear in the top-3
> prefetch candidates?). If hit rate > 80%, the index is operationally useful; if it
> is near random, the assumption of reproducible routing structure at this scale is
> falsified.

### 7.5.4 Targeted Entropy Objectives

The current objective (λ·H(p)) applies uniform pressure across all tokens. A targeted
variant — penalizing only when H(p) exceeds a threshold H_target — concentrates pressure
on high-entropy tokens. The target-entropy variants tested in Chapter 6 converge slowly
in 2,000 steps but may be more effective at longer continuation. The interaction between
target-entropy objectives and load-balancing terms (which encourage entropy at the batch
level) is also unexplored.

> **Continuation spec:** Resume target-entropy models from their 2,000-step checkpoints
> to 15,000 steps. Compare cache efficiency and quality against the standard λ·H(p)
> baseline at matched training budget. The expected result is faster cache convergence;
> the risk is quality degradation on high-entropy tokens that currently carry semantic load.

### 7.5.5 Attention Blocks

MLP blocks have low arithmetic intensity (~0.25 FLOPs/Byte at d=256, h=512). Attention
blocks have substantially higher intensity (~128 FLOPs/Byte at seq=512, d=256), making
them more favorable for the RAM→VRAM scenario: more computation per loaded byte.
Replacing MLP blocks with attention blocks while maintaining SR-Core routing would
increase the transfer-efficiency of the architecture.

> **Continuation spec:** Replace the MLP block with a multi-head attention block, keep
> the routing mechanism unchanged. Measure (a) arithmetic intensity per loaded byte,
> (b) quality on HeteroMini-v1 at matched parameters, (c) wall-clock throughput under
> forced offloading. The hypothesis is that the throughput advantage grows because the
> GPU has more useful work to do per byte transferred; the counter-risk is that attention
> routing dynamics differ from MLP and routing collapse requires different regularization.

### 7.5.6 Convergence of the Quality Gap (Cheap, Decisive)

The ~0.5-nat quality gap (Chapter 5a.5) is measured at 15k steps, where both the
param/compute-matched SR-Core and Dense d24 are still descending (~0.2 nats per 2,500 steps
each). It is therefore a lower bound, not a converged value. Training both to convergence
(~40k+ steps; ~6–8 h on the same RTX 2060, with auto-resume from the 15k snapshots — no new
setup) would settle whether the gap narrows, holds, or widens at convergence. The descent rates
are comparable at 15k, so this data shows no sign of SR-Core catching up — but a weight-tied
recursive core may simply need more training to "drill in" the shared refinement operator, and
that is exactly what this run would test. It is the one open question in this work that the
available hardware can answer cleanly; the rest (scale, downstream tasks, the Leiterbahn index)
require resources and curiosity beyond this program.

> **Continuation spec:** Resume both models from their 15k checkpoints to 40k steps
> (auto-resume, no new setup, ~6–8 h on the same RTX 2060). Measure held-out loss every
> 2,500 steps. Three outcomes: (1) gap narrows → weight-tied recursion was under-trained,
> SR-Core may be competitive at longer horizon; (2) gap holds → the ~0.5-nat cost is
> stable, the current claim is confirmed; (3) gap widens → format under-performs as
> training length increases, a stronger negative result than the current lower bound.

## 7.6 Claim Boundaries

This monograph claims:

> SR-Core provides a hard working-set guarantee WS=k, and entropy-based router
> consolidation creates a reproducible cache/locality axis without degrading language
> quality at the Pareto-optimal operating point (λ=0.003, two seeds).

It also claims, now backed by measurement rather than simulation:

> Under RAM→VRAM offloading on consumer hardware (RTX 2060), SR-Core achieves a real
> wall-clock throughput advantage over dense layer-offloading (~1.6×) at small scale, at a
> measured quality cost of ~0.5 nats (param- and compute-matched, at this training horizon and
> scale; whether the gap changes at convergence or larger scale remains open).

It does **not** claim:

> dense-quality parity, a converged quality gap, or demonstrated efficiency *at deployment
> scale* (≫VRAM models, batched serving).

The honest framing is: the chain *fewer bytes → more tokens/second* is now demonstrated
end-to-end on real hardware, and the quality price for it is measured and not attributable to
parameter or compute budget at this training horizon. What remains open is magnitude at scale —
whether the small-scale throughput win and the quality gap both move favourably as models grow
— which requires hardware beyond this program.

## 7.7 Conclusion

Block-sparse recursive language models with shared routing (SR-Core) combine two
properties that are jointly necessary for predictable parameter streaming: a hard active-
set bound (WS=k) and, with entropy regularization, a controllable cache locality axis.

Both properties have been demonstrated at the scale studied. The primary remaining task
is demonstrating that they survive the translation from simulation to real hardware —
a demonstration that requires a RAM→VRAM prototype with GPU-native sparse dispatch and
block sizes large enough to amortize transfer overhead.
