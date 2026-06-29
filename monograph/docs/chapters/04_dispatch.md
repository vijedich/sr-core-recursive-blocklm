# Chapter 4: Dispatch Overhead — The CPU-Side Tax of Sparse Routing

## 4.1 Motivation

The working-set guarantee (WS=k) reduces the number of blocks that must be loaded per
token. However, loading fewer blocks is only beneficial if the cost of *selecting* which
blocks to load (dispatch overhead) does not dominate the savings.

This chapter characterizes the dispatch overhead of SR-Core relative to a
compute-matched dense baseline on CPU, to establish a realistic picture of where the
current prototype stands and what hardware properties are required for the sparse
approach to yield net inference speedup.

## 4.2 Benchmark Setup

**Hardware:** Intel Core i7-10700 (8 cores / 16 threads, 2.90 GHz), 16 GB RAM; PyTorch CPU backend  
**Config:** batch size 8, sequence length 128 (1,024 tokens/forward), median over 15 runs  
**Note:** Core time excludes deep-supervision readout heads; same overhead for all models.

Four measurement modes for the sparse model:
- **A_dense_all:** All n=64 blocks × R=6 steps = 384 block-applications (upper bound)
- **B_sparse_route:** True sparse dispatch, full routing at each step r
- **C_sparse_pre:** True sparse dispatch, routing pre-computed once at r=1 (= SR-Core)
- **D_fixed_core:** True sparse dispatch, completely fixed block set (no routing)

## 4.3 Results

| Model | Block-apps/token | L_final | Params | Core time | tok/s |
|---|---|---|---|---|---|
| Sparse b64 R=6 (C_sparse_pre) | 24 | 3.720 | 19M | **158 ms** | 6,471 |
| Sparse b64 R=2 | 8 | ~3.72 | 19M | 57 ms | 18,057 |
| **Dense d24 (compute-matched)** | **24** | **3.481** | **8.7M** | **56 ms** | **18,185** |
| Dense d8 | 8 | 3.582 | 4.5M | 18 ms | 58,016 |

Additional breakdown at R=6:
- A_dense_all (64 blocks × 6 = 384 apps): 823 ms
- D_fixed_core (~7 fixed blocks × 6 = ~42 apps): 92 ms  
- Router overhead (B − C): **7 ms** — only 5% of sparse total; routing is not the bottleneck
- Pure dispatch cost: 151 ms for 24 block-apps vs. 56 ms dense → **2.7× dispatch tax**

**Result:** Sparse R=6 is **2.8× slower** than compute-matched Dense d24 at *identical*
24 block-applications per token (158 ms vs. 56 ms), and qualitatively worse (3.720 vs.
3.481) despite more parameters (19M vs. 8.7M).

This is a measured result, not a theoretical estimate. The overhead is confirmed.

## 4.4 Sources of Dispatch Overhead

Three main contributors:

1. **Routing computation:** The router must evaluate all n block keys against the query
   vector and apply top-k selection. This is O(n) per token.

2. **Gather/scatter:** After routing, the selected k blocks must be gathered from the
   block bank (non-contiguous memory access). Dense execution is a single contiguous
   matrix multiply.

3. **Kernel launch overhead:** Each block is a separate operation. At k=8 and R=6, this
   is 48 small matrix multiplications per token vs. a single large one for dense.

## 4.5 Implications

The 2.8× dispatch overhead establishes a clear engineering requirement: the
bytes-in-motion reduction must exceed 2.8× *on the transfer-bound path* before sparse
routing yields net benefit.

From the offloading simulation (Chapter 5b):
- At K=8 cached blocks, SR-Core requires 4.1× fewer bytes/token than dense
  layer-offloading (3,035 vs 12,348 KB/token)

In the RAM→VRAM transfer-bound regime, the 4.1× transfer reduction can in principle
offset the 2.8× dispatch overhead — but this depends on the ratio of transfer latency
to compute latency, which varies by hardware. On memory-bandwidth-limited hardware
(where transfers are the bottleneck), the sparse approach can win; on
compute-bandwidth-limited hardware, it cannot.

**The dispatch overhead does not invalidate the sparse approach; it specifies the target
deployment regime:** hardware where the RAM→VRAM transfer cost dominates, and the
overhead of dynamic dispatch is amortized by large block sizes and high bandwidth.

## 4.6 Mitigation Directions

1. **Larger blocks:** Higher arithmetic intensity (FLOPs/byte) amortizes dispatch cost
   over more computation. Attention blocks have ~128× higher intensity than MLP blocks
   at typical sequence lengths.

2. **Hardware support:** GPU-native sparse dispatch (e.g., CUDA Streams, structured
   sparsity) eliminates the Python-level dispatch overhead.

3. **Batch processing:** At inference batch size >1, routing decisions can be
   vectorized. The overhead is amortized across the batch.

4. **Leiterbahn (future):** If the active block set is known from a routing index before
   execution begins, dispatch reduces to a table lookup with prefetched weights.

## 4.7 Summary

Sparse dispatch imposes a ~2.8× CPU overhead relative to compute-matched dense
execution. This overhead is *not* disqualifying in the target deployment scenario
(RAM→VRAM transfer-bound inference), where the 4.1× reduction in bytes-in-motion
exceeds it. However, a real-latency demonstration requires hardware where transfer
costs dominate, which the current CPU prototype does not test.

Data source: `data/eval/phase1/cpu_benchmark.json`
