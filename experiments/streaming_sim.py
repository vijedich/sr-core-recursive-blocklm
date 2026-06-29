"""Experiment 2 — cache/streaming simulation on real routing traces.

Honest framing: the toy bank has 24 blocks (~12 MB at fp16) and fits entirely in
VRAM, so this CANNOT prove byte savings at deployment scale. What it CAN do, and
what governs a real large-bank system, is characterise LOCALITY:

  * working-set / miss-rate vs cache capacity (the shape extrapolates to scale)
  * cross-token reuse distance (does a loaded block serve many tokens?)
  * entry-phase vs steady-state load share (where transfer cost concentrates)

Decisive test: compare LEARNED routing against a RANDOM-routing control with the
same load. If learned routing has a much lower miss-rate under cache pressure,
the locality is real and streamable IN PRINCIPLE. If not, Jaccard overlap was a
mirage and the plan's streaming hypothesis needs rethinking before scaling.
"""
from __future__ import annotations
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

# ---- physical model (full-spec block, not the tiny demo block) ----
D, H = 256, 512
BLOCK_PARAMS = 2 * D * H + H + D + 2 * D          # two matmuls + biases + LayerNorm
DTYPE_BYTES = {"fp32": 4, "fp16": 2, "int8": 1}
FLOPS_PER_BLOCK_TOKEN = 4 * D * H                 # ~0.52 MFLOP / token / block
GPU_THROUGHPUT = 50e12                            # effective FLOP/s for batched small GEMMs
SETUP_S = 5e-6                                    # per-transfer DMA launch latency


def load_traces():
    z = np.load(os.path.join(RESULTS, "traces_s0.npz"))
    meta = json.loads(str(z["meta"]))
    return z["traces"], z["regimes"], meta            # (nb,R,B,T,k)


# ---------- build request-set streams ----------
def per_sequence_streams(tr, rng=None):
    """List over sequences of list over micro-steps (t outer, r inner) of block-id arrays.
    If rng given, replace each set with k random distinct blocks (control)."""
    nb, R, B, T, k = tr.shape
    streams = []
    nblocks = int(tr.max()) + 1
    for bi in range(nb):
        for s in range(B):
            steps = []
            for t in range(T):
                for r in range(R):
                    if rng is None:
                        steps.append(np.unique(tr[bi, r, s, t]))
                    else:
                        steps.append(rng.choice(nblocks, size=k, replace=False))
            streams.append(steps)
    return streams, nblocks


def batched_stream(tr, rng=None):
    """Single stream: union over the B sequences at each (t,r) micro-step."""
    nb, R, B, T, k = tr.shape
    nblocks = int(tr.max()) + 1
    steps = []
    for bi in range(nb):
        for t in range(T):
            for r in range(R):
                if rng is None:
                    steps.append(np.unique(tr[bi, :, :, t].reshape(-1))) if False else \
                        steps.append(np.unique(tr[bi, r, :, t].reshape(-1)))
                else:
                    u = np.unique(np.concatenate(
                        [rng.choice(nblocks, size=k, replace=False) for _ in range(B)]))
                    steps.append(u)
    return steps, nblocks, (R, B, T, k)


# ---------- cache simulation ----------
def simulate(streams, capacity, nblocks, policy="lru", pinned=None):
    """streams: list of step lists (per-sequence) OR a single step list (batched).
    Returns aggregate stats. Cache persists within each stream; reset per stream."""
    if streams and isinstance(streams[0], np.ndarray):
        streams = [streams]                       # wrap single batched stream
    pinned = set(pinned or [])
    total_req = total_miss = 0
    loads_by_inner0 = 0; loads_total = 0
    reuse_dist = []
    R_inner = None
    for steps in streams:
        cache = list(pinned)                      # MRU at end
        last_seen = {}
        step_idx = 0
        for si, req in enumerate(steps):
            for b in req:
                b = int(b)
                total_req += 1
                if b in last_seen:
                    reuse_dist.append(step_idx - last_seen[b])
                last_seen[b] = step_idx
                if b in cache or b in pinned:
                    if b in cache:                # LRU bump
                        cache.remove(b); cache.append(b)
                else:
                    total_miss += 1; loads_total += 1
                    # inner-iteration index of this micro-step (r = si % R)
                    cache.append(b)
                # evict
                cap_free = capacity - len(pinned)
                while len(cache) > cap_free:
                    cache.pop(0)
            step_idx += 1
    return dict(requests=total_req, misses=total_miss,
                miss_rate=total_miss / max(1, total_req),
                hit_rate=1 - total_miss / max(1, total_req),
                reuse_p50=float(np.median(reuse_dist)) if reuse_dist else 0.0,
                reuse_p90=float(np.percentile(reuse_dist, 90)) if reuse_dist else 0.0,
                loads=loads_total)


def inner_phase_load_share(streams, capacity, nblocks, R):
    """Fraction of total block-LOADS that occur at the entry iteration r=0."""
    if streams and isinstance(streams[0], np.ndarray):
        streams = [streams]
    load0 = loadrest = 0
    for steps in streams:
        cache = []; 
        for si, req in enumerate(steps):
            r = si % R
            for b in req:
                b = int(b)
                if b in cache:
                    cache.remove(b); cache.append(b)
                else:
                    if r == 0: load0 += 1
                    else: loadrest += 1
                    cache.append(b)
                    while len(cache) > capacity:
                        cache.pop(0)
    tot = load0 + loadrest
    return load0 / max(1, tot), loadrest / max(1, tot)


# ---------- driver ----------
def run():
    tr, reg, meta = load_traces()
    R, k, nblocks = meta["R"], meta["k"], meta["n_blocks"]
    rng = np.random.default_rng(0)

    learned_seq, _ = per_sequence_streams(tr)
    random_seq, _ = per_sequence_streams(tr, rng=rng)
    learned_batch, _, dims = batched_stream(tr)
    random_batch, _, _ = batched_stream(tr, rng=np.random.default_rng(1))

    caps = [4, 6, 8, 10, 12, 16, 20, 24]
    out = {"meta": meta, "caps": caps,
           "block_bytes": {dt: BLOCK_PARAMS * b for dt, b in DTYPE_BYTES.items()},
           "seq": {"learned": [], "random": []},
           "batch": {"learned": [], "random": []}}
    for cap in caps:
        out["seq"]["learned"].append(simulate(learned_seq, cap, nblocks))
        out["seq"]["random"].append(simulate(random_seq, cap, nblocks))
        out["batch"]["learned"].append(simulate(learned_batch, cap, nblocks))
        out["batch"]["random"].append(simulate(random_batch, cap, nblocks))

    # entry-phase load share (per-sequence, modest cache)
    s0, srest = inner_phase_load_share(learned_seq, capacity=8, nblocks=nblocks, R=R)
    out["entry_phase"] = {"r0_load_share": s0, "rrest_load_share": srest}

    # hub-pinning: pin top-H most frequent blocks (learned vs random), batched stress
    freq = np.bincount(tr.reshape(-1), minlength=nblocks)
    hub_order = np.argsort(-freq).tolist()
    out["hub"] = {"freq_sorted": freq[hub_order][:8].tolist()}
    hub_res = {"learned": [], "random": []}
    Hs = [0, 2, 4, 6, 8]
    for Hh in Hs:
        pin = hub_order[:Hh]
        hub_res["learned"].append(
            simulate(learned_batch, capacity=10, nblocks=nblocks, pinned=pin)["miss_rate"])
        # random control: pinning arbitrary blocks
        hub_res["random"].append(
            simulate(random_batch, capacity=10, nblocks=nblocks,
                     pinned=list(range(Hh)))["miss_rate"])
    out["hub"]["H"] = Hs; out["hub"]["miss_rate"] = hub_res

    # ---- latency / amortization headline ----
    # Single-stream (one user) is pessimistic; batched serving amortizes one block
    # load over B concurrent tokens. Report BOTH.
    cap_fit = 12
    learned_1 = simulate(learned_seq, cap_fit, nblocks)
    learned_B = simulate(learned_batch, cap_fit, nblocks)
    R_, B_, T_, k_ = dims
    tokens = B_ * T_ * tr.shape[0]
    out["latency"] = {
        "single_stream": _latency_table(learned_1["loads"], tokens, R_, k_, concurrency=1),
        "batched_serving": _latency_table(learned_B["loads"], tokens, R_, k_, concurrency=B_),
    }

    # ---- break-even: how many uses per loaded block to hide transfer? ----
    # need uses * flops_per_use / throughput >= block_bytes / BW
    out["break_even"] = {}
    for dt, by in DTYPE_BYTES.items():
        bb = BLOCK_PARAMS * by
        out["break_even"][dt] = {
            bw: bb * GPU_THROUGHPUT / (FLOPS_PER_BLOCK_TOKEN * bw * 1e9)
            for bw in [4, 8, 16, 32]}

    with open(os.path.join(RESULTS, "streaming_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    _figures(out)
    _print_summary(out)
    return out


def _latency_table(loads, tokens, R_, k_, concurrency=1):
    block_bytes_fp16 = BLOCK_PARAMS * DTYPE_BYTES["fp16"]
    total_loaded = loads * block_bytes_fp16
    amort_bytes_per_token = total_loaded / tokens
    total_block_apps = tokens * R_ * k_
    compute_s = total_block_apps * FLOPS_PER_BLOCK_TOKEN / GPU_THROUGHPUT
    rows = {}
    for bw_gbs in [4, 8, 16, 32]:
        bw = bw_gbs * 1e9
        transfer_s = loads * SETUP_S + total_loaded / bw
        stall_prefetch = max(0.0, transfer_s - compute_s)
        rows[bw_gbs] = dict(transfer_ms=transfer_s * 1e3, compute_ms=compute_s * 1e3,
                            stall_ms_prefetch=stall_prefetch * 1e3,
                            transfer_bound=transfer_s > compute_s)
    return dict(concurrency=concurrency, loads=loads,
                amort_bytes_per_token=amort_bytes_per_token,
                total_loaded_MB=total_loaded / 1e6,
                compute_ms=compute_s * 1e3, by_bandwidth=rows)


# ---------- figures ----------
def _figures(out):
    caps = out["caps"]
    # fig5: miss-rate vs capacity, learned vs random, per-seq + batched
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(caps, [s["miss_rate"] for s in out["seq"]["learned"]], "o-",
            color="#2563eb", label="learned, single-stream")
    ax.plot(caps, [s["miss_rate"] for s in out["seq"]["random"]], "o--",
            color="#93c5fd", label="random ctrl, single-stream")
    ax.plot(caps, [s["miss_rate"] for s in out["batch"]["learned"]], "s-",
            color="#b91c1c", label="learned, 48-stream batch")
    ax.plot(caps, [s["miss_rate"] for s in out["batch"]["random"]], "s--",
            color="#fca5a5", label="random ctrl, 48-stream batch")
    ax.set_xlabel("VRAM cache capacity (blocks of 24)")
    ax.set_ylabel("cache miss rate")
    ax.set_title("Exp2: is the locality real?\nlearned routing vs random-routing control")
    ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig5_missrate.png"), dpi=130); plt.close(fig)

    # fig6: hub pinning + entry-phase
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    Hs = out["hub"]["H"]
    axes[0].plot(Hs, out["hub"]["miss_rate"]["learned"], "o-", color="#2563eb",
                 label="learned")
    axes[0].plot(Hs, out["hub"]["miss_rate"]["random"], "o--", color="#fca5a5",
                 label="random ctrl")
    axes[0].set_xlabel("# pinned hub blocks (cap=10)")
    axes[0].set_ylabel("miss rate"); axes[0].legend(fontsize=8)
    axes[0].set_title("hub pinning helps only\nif hub structure is real")
    axes[0].grid(alpha=.3)
    e = out["entry_phase"]
    axes[1].bar(["entry r=0", "steady r>=1"],
                [e["r0_load_share"], e["rrest_load_share"]],
                color=["#dc2626", "#2563eb"])
    axes[1].set_ylabel("share of all block loads")
    axes[1].set_title("transfer cost concentrates\nin the entry iteration")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig6_hub_entry.png"), dpi=130); plt.close(fig)


def _print_summary(out):
    print("\n" + "=" * 68)
    print("EXP2 STREAMING SIMULATION (toy 24-block bank; locality characterisation)")
    print("=" * 68)
    print(f"block size fp16 = {out['block_bytes']['fp16']/1e3:.0f} KB "
          f"(fp32 {out['block_bytes']['fp32']/1e3:.0f} KB, int8 {out['block_bytes']['int8']/1e3:.0f} KB)")
    print("\nmiss-rate vs capacity (single-stream):")
    for cap, l, r in zip(out["caps"], out["seq"]["learned"], out["seq"]["random"]):
        print(f"  cap={cap:2d}: learned={l['miss_rate']:.3f}  random={r['miss_rate']:.3f}")
    l8 = out["seq"]["learned"][2]
    print(f"\nreuse distance (learned, micro-steps): p50={l8['reuse_p50']:.0f} "
          f"p90={l8['reuse_p90']:.0f}")
    e = out["entry_phase"]
    print(f"entry-phase load share: r=0 -> {e['r0_load_share']:.2f}, "
          f"r>=1 -> {e['rrest_load_share']:.2f}")
    for label in ["single_stream", "batched_serving"]:
        lat = out["latency"][label]
        print(f"\n[{label}] concurrency={lat['concurrency']}  "
              f"amort bytes/token={lat['amort_bytes_per_token']:.0f}  "
              f"loaded={lat['total_loaded_MB']:.1f}MB  compute={lat['compute_ms']:.2f}ms")
        for bw, row in lat["by_bandwidth"].items():
            tag = "TRANSFER-BOUND" if row["transfer_bound"] else "compute-hideable"
            print(f"   {bw:2d} GB/s: transfer={row['transfer_ms']:.2f}ms "
                  f"stall={row['stall_ms_prefetch']:.2f}ms  [{tag}]")


    print("\nBREAK-EVEN: uses per block-load needed to hide transfer behind compute")
    for dt in ["fp16", "int8"]:
        be = out["break_even"][dt]
        print(f"  {dt}: " + "  ".join(f"{bw}GB/s->{v:,.0f}" for bw, v in be.items()))


if __name__ == "__main__":
    run()
