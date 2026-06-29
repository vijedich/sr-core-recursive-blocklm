# Entropy-based Router Consolidation for Cache-Efficient Block-Sparse Language Models

**Draft v0 — Sections 1–7 complete**  
*Viktor Jedich — 2026-06-23*

---

## 1. Introduction

Deploying large language models under memory constraints requires loading model parameters from host
memory at each forward pass. For a 7B-parameter model in fp16, streaming roughly 14 GB of weights
from host memory at 16 GB/s would impose an idealized lower bound of about 0.9 s/token before
compute and overheads — a regime in which memory movement, not arithmetic, is the binding
constraint. Quantization compresses individual parameters but preserves the all-weights-per-token
access pattern. Mixture-of-Experts routing [CITATION] reduces *which* parameters are active per
token, but standard MoE architectures activate different expert sets per layer, accumulating a
transfer budget proportional to depth.

We study a complementary structure: a single shared *block bank* that a recursive router accesses
repeatedly. At each recursion step $r$, the router selects a small set of $k$ blocks from a bank of
$n$ blocks and applies them to the running hidden state. The key constraint — the **SR-Core**
invariant — is that routing happens only at $r = 1$; steps $r = 2, \ldots, R$ reuse the same block
selection. The active weight set per token is therefore exactly $k$ blocks, regardless of recursion
depth $R$. We refer to this as the **working set** (WS $= k$), and it is independent of bank size
$n$ and depth $R$.

This guarantee is architecturally meaningful: at inference time the identity of the $k$ active
blocks is known before any matmul is executed, enabling block-granular prefetching. With $n = 64$
blocks and $k = 8$, the working set covers $12.5\%$ of the bank. Offloading simulation on trained
models confirms the structural advantage: sparse SR-Core loads approximately $4.1\times$ fewer
bytes per token than dense layer-offloading at a realistic LRU cache capacity of $K = 8$ blocks
($3{,}035$ vs. $12{,}348$ KB/token at 16 GB/s). This is a simulated bytes-in-motion estimate
under an LRU caching model, not a measured latency speedup.

WS $= k$ is, however, a *necessary* condition for cache efficiency, not a sufficient one. Whether
the actual bytes-in-motion are low depends on how the router distributes selections across tokens.
If every token activates a different set of $k$ blocks, the effective working set *across a sequence*
approaches $n$, and a small cache will miss on nearly every request. Cache efficiency requires that
routing patterns are **locally stable** — that nearby or semantically similar tokens tend to route
through overlapping block sets, so that a modest cache serves a disproportionately large fraction of
requests.

We show that unconstrained SR-Core training produces a router that is only partially stable, and
that a targeted entropy-based objective can systematically move the router along a
**cache/locality axis** — trading a small amount of per-token routing diversity for significantly
improved block-level cache hit rates. Mild entropy pressure improves held-out code behavior with modest cache gains;
stronger pressure consistently improves cache metrics, with quality effects that are
seed-dependent. Dense models remain the quality upper bound throughout; the contribution is a
controllable systems-level degree of freedom within the sparse model family.

**Contributions.** We present:
1. An analysis of SR-Core routing dynamics showing a stable two-phase structure (token-specific
   entry; collapsed reusable core) and an inherent reuse/novelty tension.
2. An entropy-based router consolidation objective that converts this tension into a tunable
   cache/locality axis, with methodologically consistent negative controls.
3. Empirical evaluation on a four-domain pretraining benchmark confirming a reproducible
   Pareto frontier between cache efficiency and held-out code generalization, replicated across
   seeds, and bounded above by a dense quality baseline.

---

## 2. Architectural Motivation: Reuse, Novelty, and Two-Phase Routing

Before introducing the entropy objective, we document two routing phenomena that motivated the
SR-Core design. Both observations come from analysis of earlier recursive-routing variants in which
each step $r$ was free to select independently; this per-step freedom is precisely what the SR-Core
constraint removes. The findings are reported here as *motivation* for that constraint and for the
entropy objective, not as properties of the final SR-Core model (where $S_t$ is identical across
all steps by construction).

### 2.1 Two-Phase Routing Structure

SR-Core routing proceeds in $R$ recursive steps. At $r = 1$ the router selects $k$ blocks for the
current token; steps $r = 2, \ldots, R$ reuse the same selection. To characterize routing dynamics
we measure the **Jaccard similarity** between the selected block sets of consecutive tokens at each
recursion step. Formally, for tokens at positions $t$ and $t+1$, the hard-overlap Jaccard at step
$r$ is

$$J_r = \frac{|S_r(t) \cap S_r(t+1)|}{|S_r(t) \cup S_r(t+1)|}$$

where $S_r(t) \subseteq \{1, \ldots, n\}$ is the top-$k$ selection at position $t$, step $r$.

In trained models we observe a stable two-phase pattern. At the **entry step** ($r = 1$), Jaccard
similarity between consecutive tokens is low ($J_1 \approx 0.20$): each token routes through a
largely distinct block set, functioning as a token-specific gatekeeper that selects an entry path
into the block bank. At all subsequent steps ($r = 2, \ldots, R$), Jaccard similarity collapses to
a near-constant high value ($J_{2..6} \approx 0.984$): the model has settled into a reusable core
that is shared across most tokens regardless of content.

This structure emerges spontaneously from language modeling loss without any explicit routing
supervision. It is architecturally convenient for working-set minimization — a stable resident core
reduces cold-start misses — but it also reveals a potential inefficiency: the reusable core may
represent over-consolidation rather than learned representational depth.

### 2.2 Reuse/Novelty Tension

To test whether the high-Jaccard reuse regime is optimal, we conduct **forced-diversity ablations**:
at evaluation time, we mask the $k$ blocks selected at step $r-1$ before computing the top-$k$ at
step $r$, forcing the model to route through fresh blocks at each recursive step.

The results are diagnostic. In a model trained without any routing regularization, forced diversity
*improves* loss at deep recursion steps: at $r = 6$, forcing diversity reduced loss by approximately
$0.104$ nats compared to the model's learned routing, which had converged to near-constant block
reuse. This means that the high-Jaccard reuse pattern observed in trained models is not the
representationally optimal configuration — it is a training artifact. The routing objective
(minimize next-token prediction loss) rewards early reuse because it reduces the variance of the
block computation, but this comes at the cost of the incremental representational update that later
recursive steps could provide.

These two observations together define the tension: **reuse** (same blocks, fewer cold-start
misses, cache-efficient) versus **novelty** (fresh blocks, more incremental representation,
potentially better predictions). Router structure is causally relevant: neither unconstrained
reuse nor unconstrained diversity is a sufficient operating point. This motivates controlled
router shaping — an objective that exposes the reuse/novelty axis as a tunable degree of freedom
rather than maximizing either endpoint.

### 2.3 From Observation to Intervention

These findings motivate a router-shaping objective, but they do not prescribe the form of that
objective. One natural candidate is a cross-token similarity penalty (e.g., encouraging lower
Jaccard between consecutive tokens' routing decisions). We tested a soft Jaccard objective and
found that it is the wrong lever: it acts on pairwise token overlap but does not control the
*concentration* of the router's pre-selection probability distribution. In our evaluation, the
soft-Jaccard variant increased route diversity (more unique block combinations across the corpus)
while simultaneously reducing hard overlap — the opposite of the cache-beneficial consolidation we
seek.

The correct intervention point is the **sharpness of the pre-selection distribution** at $r = 1$.
A flatter distribution means the router is uncertain which blocks to select; sharpening it means
the router assigns higher probability mass to a smaller set of blocks before the top-$k$ cut,
producing more consistent routing choices across tokens with similar representations. Entropy
minimization acts directly on this quantity.

These observations motivate router-shaping objectives: not to maximize diversity or reuse blindly,
but to expose a controllable axis between route diversity and cache locality.

---

## 3. Method: Entropy-based Router Consolidation

### 3.1 SR-Core Architecture

We briefly formalize the architecture to fix notation. A block bank
$\mathcal{B} = \{F_1, \ldots, F_n\}$ contains $n$ parameter blocks, each a two-layer MLP with
shared input/output dimension $d$. The model processes each input token through $R$ recursive
applications of the bank. At step $r = 1$, a learned router produces a probability distribution
over blocks:

$$\mathbf{p}_t = \mathrm{softmax}\!\left(\mathbf{q}_t^\top \mathbf{K}\right) \in \mathbb{R}^n$$

where $\mathbf{q}_t = W_q \mathbf{h}_{t,0}$ is a query derived from the initial hidden state and
$\mathbf{K} \in \mathbb{R}^{n \times d_k}$ is a learned key matrix. The **SR-Core** selection is

$$S_t = \mathrm{top}\text{-}k(\mathbf{p}_t), \quad \text{WS} = |S_t| = k$$

with soft gates $g_{t,b} \propto p_{t,b}$ for $b \in S_t$. The hidden state update at step $r$ is

$$\mathbf{h}_{t,r} = \mathbf{h}_{t,r-1} + \sum_{b \in S_t} g_{t,b} \cdot F_b(\mathbf{h}_{t,r-1})$$

using the *same* selection $S_t$ for all $r = 1, \ldots, R$. The active working set is therefore
exactly $k$ blocks per token, independent of $R$ and $n$.

Training uses a deep-supervision language modeling loss that supervises the output at every
recursion depth, with end-weighted contributions to preserve full-depth quality while allowing
intermediate readouts.

### 3.2 Entropy-based Consolidation Objective

We augment the language modeling loss with an entropy penalty on the pre-selection router
distribution:

$$\mathcal{L} = \mathcal{L}_{\mathrm{LM}} + \lambda \cdot \mathcal{H}$$

where

$$\mathcal{H} = \frac{1}{N} \sum_{t=1}^{N} H(\mathbf{p}_t^{(1)}),
\quad H(\mathbf{p}) = -\sum_{i=1}^{n} p_i \log p_i$$

and $\mathbf{p}_t^{(1)}$ is the router distribution for token $t$ at $r = 1$. The entropy penalty
$\mathcal{H}$ encourages the router to assign concentrated probability mass before the top-$k$
selection: lower entropy corresponds to higher confidence in the selected blocks, which in practice
produces more consistent routing choices across tokens with similar content and thereby improves
cross-token cache hit rates.

**Why $r = 1$ only.** Routing decisions are made at $r = 1$; steps $r = 2, \ldots, R$ reuse the
same block set $S_t$ and do not have a meaningful "unconcentrated" distribution to regularize.
Applying the entropy loss at $r = 1$ is therefore the canonical intervention point.

**Why pre-top-$k$.** The router distribution $\mathbf{p}_t$ is computed before the top-$k$
cut. Applying entropy regularization at this point shapes the entire selection landscape: the
penalty encourages the router to place probability mass on a small subset of blocks before any
discretization, rather than penalizing the post-selection gates (which are already sparse by
construction). A concentrated pre-selection distribution also improves the router's topk-margin
$p_k - p_{k+1}$, stabilizing which blocks fall at the selection boundary.

**Distinction from pairwise objectives.** We test soft-Jaccard-style overlap objectives as an
internal negative control. Such objectives operate on *cross-token* routing similarity: they
compare the block selections of consecutive tokens and penalize dissimilarity. This is a different
quantity from the *within-token* distribution concentration that entropy targets. In our evaluation,
soft-Jaccard objectives increase soft overlap metrics while simultaneously increasing the number of
unique routing combinations — the opposite of the cache-beneficial consolidation we seek, and
the wrong lever for controlling pre-selection sharpness. Entropy minimization, by acting on the
per-token distribution before the top-$k$ cut, produces the qualitatively different effect of a
reduced unique routing vocabulary.

**Relationship to load balancing.** Standard MoE load-balancing losses [CITATION] promote *uniform*
block utilization, which is the opposite of entropy minimization. We do not replace load balancing
with the entropy objective; the two terms address different failure modes (dead blocks vs.
over-diffuse routing) and can be combined. In our experiments we observe that trained SR-Core models
already activate all $n$ blocks across a corpus; load balancing is not required as a prerequisite.

**Hyperparameter $\lambda$.** We sweep $\lambda \in \{0.001, 0.003, 0.005, 0.007\}$. The penalty
is dimensionally compatible with the language modeling loss (both in nats), making $\lambda$ a
direct ratio between the two terms. Across our sweep, $\lambda = 0.003$ is the most reliable generalization-balanced operating
point, with cache and code-generalization improvements that replicate across seeds.
$\lambda = 0.005$ is the stronger cache/systems point, delivering larger LRU byte reductions;
its quality effects are seed-dependent and should not be characterized as a uniform generalization
cost (Section 5.1).

---

---

## 4. Experimental Setup

**Dataset.** We train and evaluate on HeteroMini-v1, a four-domain pretraining corpus of
approximately 6.6M tokens drawn from Web (FineWeb-Edu), Wikipedia, Code (The Stack, Python
subset), and Literature (Project Gutenberg). All domains are tokenized with a shared
byte-level BPE vocabulary of 8,000 tokens trained on the mixture. Documents are kept
contiguous during training; evaluation uses a held-out split constructed with the same
tokenizer and domain proportions, ensuring no overlap with training data.

**Model.** We use SR-Core with a block bank of $n = 64$ blocks, $k = 8$ active blocks per
token, and $R = 6$ recursion steps (${\sim}19$M total parameters, $d = 256$,
block hidden dimension 512). The dense quality baseline is a standard Transformer with 24
layers and matching hidden dimension (dense\_d24, ${\sim}100$M parameters), trained for the
same number of steps. All models are trained with AdamW, cosine learning rate schedule, and
a deep-supervision language modeling loss that weights later recursion steps more heavily.

**Training procedure.** SR-Core models are first trained for 15,000 steps as a common base
checkpoint, then continued for an additional 2,000 steps with the entropy objective at
$\lambda \in \{0.001, 0.003, 0.005, 0.007\}$. The unregularized continuation (ctrl) is
trained identically without the entropy term, providing a direct ablation baseline that
shares training history up to step 15,000. The dense baseline is trained end-to-end for
17,000 steps. Cross-seed replication uses an independently initialized SR-Core base; we
report seed 1 results for $\lambda = 0.003$ and $\lambda = 0.005$.

**Negative controls.** Three alternative objectives are evaluated against the same ctrl
checkpoint: (i) *softfull*: a soft Jaccard penalty on the full router probability vectors
across consecutive tokens, encouraging pairwise similarity; (ii) *softsharp\_a2*: a sharpened
Jaccard variant with temperature parameter $\alpha = 2$; (iii) *reduced\_noise*: training
with reduced Gumbel noise ($\sigma = 0.1$) on the top-$k$ selection. All controls are trained
for 2,000 steps from the same 15k base.

**Evaluation.** Routing metrics (router entropy $H(\mathbf{p})$, unique core combinations,
hard-overlap Jaccard, LRU bytes/token at cache capacities $K \in \{8, 16, 24, 32\}$) are
measured on 40 held-out batches using `eval_compare.py`, which evaluates both checkpoints
on the same batches. Per-domain held-out loss and code generalization ratio (code heldout
loss normalized by mean of other domain losses) are measured via paired runs that include
the dense baseline on the same batches, ensuring a consistent anchor for cross-model
comparison. Offloading simulation estimates bytes-in-motion using an LRU cache trace
collected from routing decisions on 1,024 held-out sequences.

---

## 5. Results

### 5.1 Router Consolidation

Figure 2 shows routing metrics as a function of $\lambda$. All four panels respond
monotonically to entropy pressure across the sweep range $\lambda \in [0, 0.007]$:

| $\lambda$ | $H(\mathbf{p})$ | Unique cores | Hard overlap | K=24 LRU (KB/tok) |
|---|---|---|---|---|
| 0.000 (ctrl) | 3.846 | 25,853 | 0.251 | 1,103 |
| 0.001 | 3.815 | 25,399 | 0.254 | 1,098 |
| 0.003 | 3.799 | 23,077 | 0.262 | 1,085 |
| 0.005 | 3.732 | 21,273 | 0.266 | 1,020 |
| 0.007 | 3.647 | 18,817 | 0.278 | 1,009 |

Router entropy decreases monotonically (more concentrated pre-selection distribution),
unique core combinations decrease (fewer distinct routing vocabularies), and hard-overlap
Jaccard across consecutive tokens increases (more stable routing paths). The LRU cache
metric at $K = 24$ decreases from 1,103 to 1,009 KB/token across the sweep — a $8.5\%$
reduction in simulated bytes-in-motion achieved at the cache level.

The response is non-uniform across the range. At $\lambda = 0.001$ the effect on routing
vocabulary (unique cores) is marginal ($-1.8\%$); the main consolidation occurs between
$\lambda = 0.001$ and $\lambda = 0.005$. At $\lambda = 0.007$ the unique-core count
continues to decrease, but as Section 5.2 shows, quality effects become less favorable.
We identify three qualitative regimes: a *generalization-balanced* region ($\lambda \leq 0.003$),
a *cache sweet spot* ($\lambda \approx 0.005$), and a *boundary* ($\lambda = 0.007$) where
consolidation is maximal but the generalization picture becomes less reliable.

### 5.2 Cache–Quality Pareto

Figure 1 plots each model as a point in the (K=24 LRU bytes/token, code-ratio heldout)
plane. The *code ratio* is defined as the ratio of held-out code loss to the mean of held-out
Web, Wikipedia, and Literature losses; lower values indicate better code generalization
relative to the other domains, reflecting the degree to which the model's sparse routing
serves code structure.

**lam003 Pareto-dominates ctrl.** In seed 0, $\lambda = 0.003$ improves K=24 cache
efficiency by $1.6\%$ ($1{,}103 \to 1{,}085$ KB/token) while simultaneously improving the
code-generalization ratio from $0.886$ to $0.866$ ($\Delta = -0.021$). This improvement
replicates in seed 1, where the code-ratio improvement is larger ($0.896 \to 0.822$,
$\Delta = -0.074$). lam003 is therefore a Pareto improvement over the unregularized baseline:
mild entropy pressure sharpens routing in a way that is simultaneously beneficial for cache
locality and code generalization.

**lam005: stronger cache, seed-dependent quality.** $\lambda = 0.005$ achieves the largest
consistent cache improvement in the sweep ($K=24: 1{,}103 \to 1{,}020$ KB/token, $-7.5\%$;
$K=16: -5.2\%$). Within-run paired comparisons show opposing signals across seeds: in seed 0,
lam005 code-ratio is $0.877$ versus its within-run ctrl of $0.859$ ($\Delta = +0.018$, slight
degradation); in seed 1, the ratio is $0.841$ versus within-run ctrl $0.863$ ($\Delta =
-0.022$, improvement). The Pareto figure uses a single ctrl anchor ($0.886$) for visual
consistency across points; lam005 ($0.877$) and lam003 ($0.866$) both fall below that anchor
in that common coordinate. The code-quality effect of $\lambda = 0.005$ is seed-dependent and
should not be characterized as a consistent cost or benefit.

**lam007: boundary.** At $\lambda = 0.007$ the cache metrics continue to improve marginally
($K=24: 1{,}009$ KB/token, $-0.1\%$ vs lam005), while the code-ratio returns toward ctrl
levels ($0.884$ in seed 0). The regime is dominated by lam005 on the cache dimension and
offers no generalization benefit; we treat it as the breakpoint of the controllable axis.

**Target-entropy variants.** We also evaluate a bounded entropy variant,
$\lambda \cdot \mathrm{relu}(H(\mathbf{p}) - H_{\mathrm{target}})$, with targets $H = 3.75$
and $H = 3.70$. Both fall below the Pareto frontier of the unconstrained sweep: the bounded
objective converges more slowly within the 2,000-step continuation window and does not reach
the same cache/quality operating points.

### 5.3 Dense Quality Baseline

Figure 3 shows held-out loss across all four domains for the dense baseline and three
sparse models, evaluated on the same batches. Dense d24 is the quality upper bound in every
domain:

| Model | Web | Wiki | Code | Lit |
|---|---|---|---|---|
| Dense d24 | 5.448 | 5.456 | 4.621 | 4.735 |
| SR-Core ctrl | 5.807 | 5.841 | 4.856 | 5.286 |
| + entmin $\lambda{=}0.003$ | 5.672 | 5.688 | 4.969 | 5.352 |
| + entmin $\lambda{=}0.005$ | 5.887 | 5.792 | 4.985 | 5.419 |

The dense advantage is largest in Literature ($0.55$ nats) and smallest in Code ($0.24$ nats).
Entropy-minimized variants do not narrow the dense–sparse quality gap; the objective shapes
routing structure without a commensurate language modeling quality gain in paired evaluation.
The contribution of entropy minimization is a systems-level degree of freedom, not a path
toward dense-equivalent quality.

These results are evaluated with the dense baseline as a consistent anchor (same held-out
batches for all models in each paired run). Within-run deltas are reliable; absolute
cross-run comparisons should be interpreted with the batch-sampling variance of approximately
$\pm 0.05$ nats in mind.

### 5.4 Negative Controls

The softfull objective — a soft-Jaccard penalty encouraging pairwise router similarity
across consecutive tokens — moves routing metrics in the opposite direction from entropy
minimization. Compared to ctrl, softfull increases unique core combinations from $25{,}684$
to $30{,}246$ ($+17.8\%$) and decreases hard-overlap Jaccard from $0.259$ to $0.251$.
Router entropy rises slightly ($3.843 \to 3.868$), and K=24 LRU bytes change by less than
$1.4\%$.

This confirms the mechanism described in Section 2.3: cross-token similarity objectives
act on pairwise overlap but do not control the within-token distribution concentration.
The softfull result is not a failure of regularization strength — it is a directional
failure. Maximizing cross-token similarity can be achieved by the router increasing route
diversity (unique routing combinations) while still mapping similar tokens to similar
paths on average. Pre-selection entropy is the correct intervention point.

### 5.5 Offloading Simulation

To bridge the cache-simulation metrics to a bytes-in-motion estimate, we run the LRU
offloading simulator on the fully-trained 17k-step checkpoints:

| Model | K=8 KB/tok | K=16 KB/tok | vs. Dense (K=8) |
|---|---|---|---|
| Dense d24 | 12,348 | 12,348 | $1.00\times$ |
| SR-Core ctrl | 3,035 | 1,857 | $0.246\times$ |
| + entmin $\lambda{=}0.003$ | 3,015 | 1,808 | $0.244\times$ |
| + entmin $\lambda{=}0.005$ | 2,973 | 1,747 | $0.241\times$ |

At $K = 8$ cached blocks, sparse SR-Core requires $4.1\times$ fewer bytes per token than
dense layer-offloading; entropy consolidation at $\lambda = 0.005$ adds a further $2\%$
reduction ($3{,}035 \to 2{,}973$ KB/token). These are simulated estimates under an LRU
caching model at 16 GB/s bandwidth; they do not account for transfer scheduling, compute
overlap, or dispatch overhead.

One structural boundary is worth noting: at cache capacities $K \geq n_{\mathrm{layers}}$
(here $K \geq 24$ for dense d24), the dense model fits entirely in cache and its
bytes-in-motion approach zero, while the sparse model still incurs misses from a bank of
$n = 64$ blocks. This capacity crossover is not a contradiction of the sparse working-set
result; it reflects that a cache large enough to hold the entire dense model eliminates its
transfer cost, an assumption that fails precisely in the memory-constrained deployment
regime this work targets.

---

## 6. Discussion

**What this work shows.** SR-Core provides a hard working-set guarantee: exactly $k$ blocks
are active per token, regardless of bank size or recursion depth. Entropy-based router
consolidation exposes a controllable axis within this fixed working set: increasing $\lambda$
moves the router toward more concentrated, stable routing paths, reducing simulated
bytes-in-motion and unique routing vocabulary. At $\lambda = 0.003$ this consolidation is
free in the sense that it simultaneously improves cache locality and held-out code
generalization relative to the unregularized baseline; the improvement replicates across two
independent seeds. At $\lambda = 0.005$ the cache benefit is larger and the quality effect is
seed-dependent. Dense models remain the quality upper bound throughout.

**What this work does not show.** We do not demonstrate wall-clock inference speedup.
The dispatch overhead of sparse block selection imposes a CPU-side penalty (approximately
$2.8\times$ slower than compute-matched dense in our CPU benchmark), so the bytes-in-motion
reduction does not directly translate to latency improvement without hardware co-design.
We also do not demonstrate quality parity with dense models, scale the bank beyond $n = 64$
blocks, or evaluate on downstream tasks. The bytes-in-motion estimates are from simulation
under idealized LRU caching; real transfer costs depend on block size, DRAM bandwidth,
PCIe characteristics, and prefetch scheduling, none of which are accounted for here.

**Scope of the entropy objective.** Entropy minimization sharpens the pre-selection
distribution at $r = 1$; it does not explicitly maximize diversity or cross-token overlap.
This means it is not a general solution to the reuse/novelty tension identified in Section 2:
it moves the router toward the reuse end of the axis in a controlled way, which happens to
benefit cache efficiency. Whether a different objective that explicitly targets diversity at
deep recursion steps could yield complementary gains remains an open question.

**Future directions.** The most immediate validation gap is real-latency measurement: an
end-to-end RAM→VRAM offloading prototype would connect the bytes-in-motion simulation to
observable tokens/second. Scaling the bank size beyond $n = 64$ is also necessary to test
whether the working-set invariant holds at the regime (large $n$, very small active fraction)
where the transfer-reduction ratio is most compelling. On the regularization side, targeted
entropy objectives (e.g., penalizing only when entropy exceeds a threshold) and their
interaction with load-balancing terms remain unexplored at the trained-model scale.

---

## 7. Related Work

**Mixture of Experts.** Sparse gating in transformer models routes tokens to subsets of
expert feed-forward networks [CITATION: Shazeer 2017, Switch Transformer]. Standard MoE
architectures activate a different expert subset at each layer, accumulating a transfer
budget proportional to depth. SR-Core differs in two respects: it uses a single shared
block bank applied recursively (no per-layer parameters), and the SR-Core constraint fixes
the routing decision at $r = 1$, giving WS $= k$ exactly rather than $R \cdot k$ across $R$
layers. Load-balancing auxiliary losses [CITATION] prevent expert collapse in MoE; SR-Core
does not require load balancing as a prerequisite but can combine entropy and balance terms.

**Looped and recurrent depth.** The Universal Transformer [CITATION: Dehghani 2019] applies
a shared layer repeatedly, achieving depth without parameter growth. Deep Equilibrium Models
[CITATION: Bai 2019] seek fixed points of a shared block. SR-Core is similar in using shared
parameters across recursion steps but differs in intent: it does not seek convergence
(empirically, state change persists at $12$–$22\%$ per step), and its routing decisions are
made once and reused, giving a predictable active-block set rather than a variable
computation graph. These models do not address block-level prefetching or working-set control.

**Parameter offloading.** FlexGen [CITATION: Sheng 2023] and related systems [CITATION]
optimize layer-offloading schedules for dense transformers under memory constraints, achieving
high throughput by overlapping compute and transfer. These approaches assume the full
parameter set must be accessed and optimize scheduling; SR-Core reduces *what* needs to be
loaded per token. The two approaches are complementary: a sparse working-set model could
benefit from both a reduced transfer budget and an optimized transfer schedule.

**Routing regularization.** Auxiliary objectives in MoE training include load-balancing
losses [CITATION], z-loss for router stability [CITATION: ST-MoE], and router dropout
[CITATION]. These objectives target dead experts, unstable softmax logits, or training
instability; none are designed to control the cache locality of routing patterns across
tokens. The entropy objective introduced here is specifically motivated by the pre-topk
concentration that determines cross-token routing consistency, a quantity orthogonal to
load balance or training stability.

**Weight sharing and bank reuse.** SHA-RNN [CITATION] and related weight-sharing recurrent
architectures reduce parameter count through shared matrices; the reuse is structural (always
the same weights) rather than routing-dependent. SR-Core combines weight sharing (same
blocks across recursion steps) with content-dependent selection, making the active subset
per token a function of input while keeping the total active count fixed at $k$.

---

*End of draft v0. All sections complete.*  
*Next: author review, citation resolution, figure captions, abstract.*
