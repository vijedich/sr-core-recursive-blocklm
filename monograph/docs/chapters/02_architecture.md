# Chapter 2: SR-Core Architecture and the Working-Set Guarantee

## 2.1 Background: Free-Routing Variants

Before SR-Core, earlier variants of the block-sparse recursive model selected blocks
*independently* at each recursion step r. Formally, each step maintained its own routing
decision:

```
S_r = top-k(Router(h_r))    for each r = 1, 2, …, R independently
```

Analysis of these free-routing variants revealed a **two-phase routing structure**:

- At r=1, routing is token-specific (mean inter-token Jaccard J₁ ≈ 0.20): the first
  step broadly explores the block bank.
- At r=2…R, routing collapses to near-identical selections across tokens
  (J₂₋₆ ≈ 0.984): the deeper steps overwhelmingly reuse the same blocks.

This empirical observation motivates a key design question: if steps r=2…R almost always
select the same blocks as r=1, why compute a separate routing decision for each?

Additionally, a forced-diversity experiment (entropy bonus at r=6, the deepest step)
reduced language modeling loss by 0.104 nats, indicating that routing collapse at deep
steps is a **training artifact** — the model learns to avoid the diversity penalty but
would benefit from more novel block selections at depth.

## 2.2 SR-Core: Shared Routing with Fixed Active Set

SR-Core replaces per-step routing with a single routing decision at r=1, reused for all
subsequent steps.

### 2.2.1 Block Bank

A shared bank of n parameter blocks:

```
B = {W₁, W₂, …, Wₙ}
```

Each block implements a feedforward transformation:

```
F_b(h) = W₂,b · σ(W₁,b · LayerNorm(h))
```

### 2.2.2 Router

At recursion step r=1, a noisy top-k router selects k blocks:

```
logits_b = q · k_b + ε_b,     ε_b ~ Gumbel(0, 1)
S = top-k({logits_b : b ∈ B}, k)
```

where q = Wq · h₁ is the routing query and k_b is the learned key for block b.

### 2.2.3 Reuse Rule (SR-Core)

The selection S from r=1 is reused unchanged for r=2, 3, …, R:

```
S_r = S₁    for all r = 1, …, R
```

### 2.2.4 State Update

At each step r, the state is updated by the selected blocks:

```
h_{r+1} = h_r + Σ_{b ∈ S} α_{r,b} · F_b(h_r)
```

where α_{r,b} are normalized gates over the selected k blocks.

### 2.2.5 Deep-Supervision Output

A readout head is applied after each step:

```
p_r(x_{t+1}) = softmax(W_out · h_r)
L = Σ_r w_r · CrossEntropy(p_r, x_{t+1})
```

where w_r are step weights (end-heavy weighting is used to preserve the anytime property
without flattening the depth curve).

## 2.3 The Working-Set Guarantee

**Theorem (WS = k).** Under the SR-Core reuse rule, the working set — the number of
distinct blocks accessed during the processing of a single token across all R recursion
steps — equals exactly k, independent of n and R.

*Proof sketch.* Since S_r = S₁ for all r, the union of active blocks is:
```
Union(S_1, S_2, …, S_R) = S_1
|S_1| = k
```
regardless of R or n. □

This is the fundamental architectural property that enables predictable weight caching:
the complete set of weights needed for a given token is known after the first routing
decision and does not grow with recursion depth.

### Comparison with Layer-Offloading

| Architecture | Bytes loaded per token | Grows with depth? |
|---|---|---|
| Dense layer offloading | All weights | Yes (one pass per layer) |
| Standard MoE (per-layer) | k × block_size × num_layers | Yes (independent per layer) |
| **SR-Core** | **k × block_size** | **No** |

## 2.4 Why Route Only at r=1?

The empirical motivation from free-routing analysis (J₂₋₆ ≈ 0.984) shows that routing
decisions at deep steps are nearly identical across tokens. Fixing them to r=1 trades
marginal routing adaptability (at steps where it barely exists) for a hard WS guarantee.

The pre-topk application of the entropy objective (described in Chapter 6) operates on
the routing distribution before the discrete selection, preserving the differentiable
path while controlling the concentration of the pre-selection distribution.

## 2.5 Implementation Details

**Dispatch:** True sparse dispatch — each token is processed only by its k selected
blocks, not by all n blocks with masking. This is essential for the WS guarantee to
translate into actual memory footprint reduction.

**Load balancing:** A soft load-balancing auxiliary loss encourages uniform block
utilization across the training batch. Without it, a small subset of blocks dominates
all routing decisions (collapse).

**Model configuration (this work):** n=64 blocks, k=8, R=6, d=256, block hidden
dimension 512, ~19M parameters total.

## 2.6 Relationship to Prior Work

SR-Core shares structural similarities with Universal Transformers and Deep Equilibrium
Models (shared weights applied repeatedly) but differs in two key properties:

1. Routing is content-dependent (different tokens activate different blocks)
2. The active set is fixed at r=1 (static across recursion depth)

Neither Universal Transformers nor DEQ address the predictability of the active weight
set at inference time, which is the property required for efficient offloading.

SR-Core also differs from standard MoE in that the routing is shared across depth:
in a 6-layer MoE, a token loads k experts at each of 6 layers = 6k distinct expert
loads. In SR-Core with R=6, a token loads exactly k blocks total.

A full literature review — covering MoE load-balancing, looped depth (Universal Transformer,
DEQ), parameter offloading (FlexGen), and routing regularization — is in Chapter 8.
