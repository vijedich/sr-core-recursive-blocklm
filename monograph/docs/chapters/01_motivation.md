# Chapter 1: Motivation and Problem Statement

## How to Read This Monograph

This document is a self-published research monograph, not a conference paper and not a
thesis. It records a small-scale but end-to-end investigation of one architectural question:

> *Can a language model be structured so that, for each token, only a small and predictable
> subset of its parameters must be resident on the GPU?*

The work was conducted under consumer-hardware constraints — primarily an RTX 2060 (6 GB
VRAM). Its purpose is not to claim deployment-scale performance, dense-quality parity, or a
production-ready system. Instead, it documents what could be built, trained, measured,
falsified, and bounded within those constraints.

**What was simulated vs. what was measured:**

| Claim | Status |
|---|---|
| WS=k guarantee (architectural) | Proved and empirically verified (Ch. 2–3) |
| 4.1× transfer reduction vs. dense | LRU simulation on real trained models (Ch. 5b.0) |
| Real wall-clock throughput advantage | Measured on RTX 2060 under forced offloading (Ch. 5b.4) |
| ~0.5 nats quality gap behind Dense d24 | Measured, param- and compute-matched (Ch. 5a.5) |
| Advantage at deployment scale | Not tested — requires larger models and VRAM budget |
| Quality gap at convergence | Not tested — both models undertrained at 15k steps |

**The story arc in one paragraph:**

The initial question was whether fewer bytes in motion would mean more tokens per second.
Simulation said yes. The first naive RAM→VRAM prototype said no — per-block kernel launches
dominated and made SR-Core slower than dense. Fusing those launches into a grouped block
matmul revealed the structural advantage: 297 vs. 205 tok/s. Asynchronous two-stream overlap
added another 5–9% in the transfer-limited regime. The mechanism is real and measured on this
hardware. The open question is whether it survives at deployment scale — larger blocks, models
that genuinely exceed VRAM, and batched serving. That test requires hardware beyond what is
available here.

**Open questions** are structured as continuation points in Section 7.5 — not as limitations to
apologize for, but as discrete, well-scoped experiments for anyone with the right hardware or
the right curiosity to pick up.

---

## 1.1 The Inference Bottleneck on Consumer Hardware

Large language models with 30B, 70B, or 100B+ parameters exceed the VRAM capacity of
typical consumer and prosumer GPUs. The dominant solution in practice is
*layer-by-layer offloading*: weights reside in CPU RAM or on SSD, and each layer is
transferred to VRAM before computation, then evicted afterward.

On a typical consumer GPU with 6–8 GB VRAM and PCIe bandwidth of ~16 GB/s:

```
7B model in fp16  ≈  14 GB weights
Per-token cost    =  14 GB ÷ 16 GB/s  ≈  0.9 seconds
→ approximately 1 token/second
```

This is not a software optimization problem. It is a structural problem: a densely
executed model requires *all* its weights at every token, so all of them must be
transferred.

## 1.2 The Core Hypothesis

A language model can be trained such that for each token, only a small, predictable
fraction of its weights is needed.

If this active fraction is:
- **small** (1–5% of total weights per token),
- **predictable** (the next required block is known before it is needed),
- **reproducible** (similar inputs activate similar weights),

then a model whose total size exceeds available VRAM can run on small hardware — not
because it is faster, but because it never needs the full model simultaneously.

**The goal is enablement, not optimization.** The baseline is layer-by-layer offloading,
not dense in-VRAM execution.

## 1.3 Relationship to Existing Approaches

| Approach | Mechanism | Limitation |
|---|---|---|
| Layer offloading (llama.cpp) | Load each layer sequentially | 100% of weights per token |
| Quantization (GPTQ, AWQ) | Compress weights | Fewer bytes, but still all layers |
| Standard MoE (Mixtral) | Top-k of N experts per layer | Per-layer, no cross-step predictability |
| **This work** | Shared block bank, fixed active set, fixed routing | Predictable WS=k independent of depth |

Standard MoE reduces the *number* of active parameters per layer but does not bound the
active set across recursion steps or enable routing reuse. Each layer still selects
independently.

## 1.4 Research Questions

This monograph investigates four questions:

1. **Feasibility:** Can a block-sparse recursive model learn language modeling tasks at
   all, and does its routing remain stable (no collapse to a few blocks)?

2. **Working-set guarantee:** Does fixing routing at the first recursion step and reusing
   it for subsequent steps (SR-Core) preserve language quality while guaranteeing WS=k
   independent of bank size n and recursion depth R?

3. **Cache efficiency:** Can router consolidation (entropy-based regularization) improve
   the predictability and locality of active block sets without degrading language quality?

4. **Offloading relevance:** How does the simulated bytes-in-motion of SR-Core compare
   to dense layer-offloading, and what is the effect of consolidation on this metric?

## 1.5 Scope and Limitations

This work operates at small scale: a ~19M parameter model trained on a ~6.6M token
four-domain corpus. Results are not transferable to large-scale models without further
validation. A small-scale RAM→VRAM prototype demonstrates a measured throughput advantage
under forced offloading (Chapter 5b.4); deployment-scale throughput, batched serving, and
large-block scaling remain untested. The CPU-side dispatch overhead (~2.8× relative to
compute-matched dense, Chapter 4) limits throughput in the compute-bound regime; it
becomes sub-dominant only in the transfer-bound regime after sparse dispatch has been fused
to remove per-block kernel-launch overhead.

## 1.6 Chapter Overview

- **Chapter 2** formalizes SR-Core and proves the WS=k guarantee.
- **Chapter 3** validates the guarantee empirically across bank sizes n ∈ {16, 32, 64,
  128, 256} on TinyStories.
- **Chapter 4** measures the dispatch overhead of sparse block selection.
- **Chapter 5** evaluates SR-Core on HeteroMini-v1, a four-domain corpus, including
  offloading simulation and cross-seed robustness.
- **Chapter 6** introduces entropy-based router consolidation and characterizes its
  effect on cache efficiency and language quality.
- **Chapter 7** synthesizes the findings and outlines directions for future work.
