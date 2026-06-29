# Chapter 8: Related Work

SR-Core sits at the intersection of four lines of work: sparse mixture-of-experts routing,
looped/recurrent depth, parameter offloading under memory constraints, and routing
regularization. This chapter situates the contributions of this monograph against each.

---

## 8.1 Mixture of Experts

Sparse gating in transformer models routes tokens to subsets of expert feed-forward networks
(Shazeer et al., 2017; Fedus et al., 2022). Standard MoE architectures activate a different
expert subset at each layer, so a model with depth D and per-layer budget k touches up to
D × k distinct parameter blocks per token — the transfer budget grows with depth.

SR-Core differs in two respects. First, it uses a **single shared block bank** applied
recursively: there are no per-layer parameters, only one bank of n blocks reused R times.
Second, the SR-Core constraint fixes the routing decision at r = 1 and reuses the same
selection S_t for all subsequent steps r = 2 … R. The result is WS = k exactly, independent
of R — depth and transfer budget are decoupled.

Load-balancing auxiliary losses (Lepikhin et al., 2021; Fedus et al., 2022) prevent expert
collapse in standard MoE. SR-Core does not require load balancing as a prerequisite (the
entropy objective addresses a different property — cross-token routing consistency, not
within-batch utilization), but the two terms can be combined.

## 8.2 Looped and Recurrent Depth

The Universal Transformer (Dehghani et al., 2019) applies a single shared layer repeatedly,
achieving depth without parameter growth — structurally the closest prior to SR-Core's
recursive reuse. Deep Equilibrium Models (Bai et al., 2019) seek fixed points of a shared
transformation, iterating until hidden state converges.

SR-Core is similar in using shared parameters across recursion steps but differs in intent
and measurement. It does not seek convergence: state change at each step is empirically
12–22% of the hidden norm, and this persists through training (Section 5a). The goal is not
a fixed point but a sequence of refinements within a bounded active set. More importantly,
neither the Universal Transformer nor DEQ addresses **block-level prefetching or working-set
control**: in both, the full shared layer is loaded at every step, so the transfer budget is
identical to a dense model of the same depth. The WS = k guarantee is specific to SR-Core's
top-k bank selection.

## 8.3 Parameter Offloading

FlexGen (Sheng et al., 2023) and related systems (Aminabadi et al., 2022; HuggingFace
Accelerate) optimize layer-offloading schedules for dense transformers under memory
constraints, overlapping compute and transfer to maximize throughput. These approaches assume
the **full parameter set must be accessed** — scheduling optimizes *when* each layer is
transferred, not *whether*. For a model that does not fit in VRAM, every layer must cross
the bus at some point per forward pass.

SR-Core reduces *what* needs to be loaded per token. The transfer budget per token is k
blocks rather than all n (or equivalently, all D layers in a dense model). The two approaches
are **complementary**: an SR-Core model could apply an optimized transfer schedule (FlexGen-style)
on top of an already-reduced per-token budget. The streaming prototype in Chapter 5b.4
demonstrates the combined effect at small scale without a scheduler; adding a Leiterbahn-based
prefetch plan (Chapter 7.5.3) would be the natural next integration.

## 8.4 Routing Regularization

Auxiliary objectives in MoE training include load-balancing losses (Fedus et al., 2022),
z-loss for router stability (Zoph et al., 2022), and router noise regularization
(Shazeer et al., 2017). These objectives target **different failure modes**: dead experts
(utilization collapse), unstable softmax logits (training divergence), or inadequate
exploration during early training.

The entropy objective introduced in Chapter 6 targets a different quantity entirely:
**pre-topk concentration**, which determines how consistent the routing decision is across
tokens. High pre-topk entropy means the router's probability mass is spread evenly before
the top-k cut, producing random-looking selections that change token to token — good for
coverage, bad for cache locality. Low entropy means the router's mass concentrates on a
small preferred set, producing stable, reusable routing paths. This is orthogonal to load
balance (a per-batch statistic over which blocks get used) and to training stability.

No prior routing regularizer is motivated by cross-token cache locality or by the
working-set geometry of a RAM→VRAM streaming scenario.

## 8.5 Weight Sharing and Bank Reuse

SHA-RNN (Merity, 2019) and related weight-sharing recurrent architectures reduce parameter
count through shared weight matrices. The reuse is **structural** — the same weights are
always applied, with no content-dependent selection. SR-Core combines weight sharing
(same blocks across recursion steps) with content-dependent top-k selection: the active
subset per token is a function of input, while the total active count is fixed at k.
This makes SR-Core's working set simultaneously predictable (always k blocks, never more)
and adaptive (which k blocks depends on the token).

---

## 8.6 Summary of Positioning

| Property | Dense | MoE | Universal Transformer | SR-Core |
|---|---|---|---|---|
| Working set per token | All params | k per layer | All (shared) | k total (WS=k) |
| Routing per step | — | Independent | — | Fixed at r=1, reused |
| Transfer budget | D × full | D × k | D × full | k (independent of R) |
| Depth without param growth | No | No | Yes | Yes |
| Cache locality control | — | Load balance | — | Entropy regularization |
| Offloading addressed | Scheduling | Scheduling | No | What to load |

The WS=k guarantee is the property that makes all the others tractable: it bounds the
transfer budget, enables the entropy objective to have a well-defined target (stable
active-block identity across tokens), and provides the architectural basis for the
prefetch-planning ideas in Chapter 7.5.3.

---

*References cited: Shazeer et al. (2017) "Outrageously Large Neural Networks: The Sparsely-Gated
Mixture-of-Experts Layer"; Fedus et al. (2022) "Switch Transformers"; Lepikhin et al. (2021)
"GShard"; Dehghani et al. (2019) "Universal Transformers"; Bai et al. (2019) "Deep Equilibrium
Models"; Sheng et al. (2023) "FlexGen"; Aminabadi et al. (2022) "DeepSpeed Inference";
Zoph et al. (2022) "ST-MoE"; Merity (2019) "Single Headed Attention RNN".*
