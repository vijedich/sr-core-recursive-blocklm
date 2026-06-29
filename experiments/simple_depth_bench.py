"""CPU-runnable depth benchmark using SimpleWalk.

Uses the same harness as depth_bench.py but with SimpleWalk (base=3, d_max=3,
no distractors) and a smaller model — designed to converge on a single CPU core
in ~500-1000 steps.

This fills the gap between:
  * Exp 1a: loss-variant result (real, but on non-depth-controlled data)
  * Exp 1b: full harness (correct architecture, but needs GPU to converge)

SimpleWalk keeps the non-commutativity that makes PermutationWalk a genuine
depth probe, while being fast enough to get a meaningful L[d,r] result on CPU.

Run:
    python -m experiments.simple_depth_bench
    python -m experiments.simple_depth_bench --steps 1000 --weighting end

Expected outcome at 800-1000 steps (end-weighting):
  * Loss well below random baseline (ln(9) ≈ 2.20)
  * L[d=3, r=1] > L[d=3, r=3]  (deeper tasks gain from more iterations)
  * corr(r*, d) > 0  (model uses more iterations for harder depths)
  * state_reset raises loss  (accumulated state matters)
"""
from __future__ import annotations
import argparse, os, json, math, time, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# make sure project root is on path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rblm.config import ModelConfig
from rblm.models import build_model, iteration_losses, weighting
from rblm.depth_tasks import SimpleWalk
from experiments.depth_bench import loss_depth_iter, oracle_loss, rstar, spearman

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
D_MAX = 3   # SimpleWalk has d_max=3
SEQ   = 16  # generous padding for d=3, no distractors (max needed: 11)


def make_small_model(data, R=4, k=4, n_blocks=16, read=True):
    """Smaller model than the GPU harness — fits comfortably in CPU memory/time."""
    cfg = ModelConfig(
        vocab_size=data.vocab_size,   # 9
        d_model=64, block_hidden=128,
        n_heads=4, context_layers=1,
        max_len=SEQ, variant="C",
        n_blocks=n_blocks, k_active=k,
        routed_iters=R, key_dim=32,
        router_noise_std=0.4,
        recurrent_read=read,
    )
    return build_model(cfg)


def train(model, data, steps, weighting_kind, seed, lr=3e-3, bs=128):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    R = model.n_iters
    w = weighting(R, weighting_kind)
    for step in range(steps):
        t, tg, mask, dep = data.batch(bs, SEQ)
        logits, aux = model(t)
        per = iteration_losses(logits, tg, mask)
        loss = (w * per).sum() + 0.01 * sum(a["lb_loss"] for a in aux["iters"]) / R
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"  step {step:4d}  loss={loss.item():.3f}  "
                  f"[{', '.join(f'{p.item():.3f}' for p in per)}]")


def run(steps=800, weighting_kind="end", seed=0):
    data = SimpleWalk(seed=1)
    print(f"\n=== SimpleWalk CPU bench | steps={steps} weighting={weighting_kind} seed={seed} ===")
    print(f"    vocab={data.vocab_size}  D_max={D_MAX}  "
          f"random_baseline={math.log(data.vocab_size):.3f}")

    model = make_small_model(data, R=4, k=4, n_blocks=16, read=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    params: {n_params:,}")

    t0 = time.time()
    train(model, data, steps, weighting_kind, seed)
    elapsed = time.time() - t0
    print(f"  Training: {elapsed:.0f}s")

    # --- core measurement: L[d, r] ---
    Ldr = loss_depth_iter(model, data, D_MAX, batches=10, bs=128, seq=SEQ)
    rs = [rstar(Ldr[d]) for d in range(D_MAX)]
    depths = list(range(1, D_MAX + 1))
    G = [float(Ldr[d, 0] - Ldr[d, -1]) for d in range(D_MAX)]
    corr = spearman(np.array(rs), np.array(depths))

    # --- controls ---
    oracle = oracle_loss(model, data, D_MAX, batches=10, bs=128, seq=SEQ)
    reset_L  = loss_depth_iter(model, data, D_MAX, batches=10, bs=128, seq=SEQ,
                               state_reset=True)[:, -1].mean()
    shuffle_L = loss_depth_iter(model, data, D_MAX, batches=10, bs=128, seq=SEQ,
                                random_route=True)[:, -1].mean()

    # --- no-read control (architecture without in-loop attention) ---
    print("\n  Training no-read control (same steps, no in-loop attention)...")
    model_noread = make_small_model(data, R=4, k=4, n_blocks=16, read=False)
    train(model_noread, data, steps, weighting_kind, seed + 100)
    Ldr_noread = loss_depth_iter(model_noread, data, D_MAX, batches=10, bs=128, seq=SEQ)
    noread_final = float(Ldr_noread[:, -1].mean())

    out = {
        "task": "SimpleWalk", "steps": steps, "weighting": weighting_kind,
        "seed": seed, "n_params": n_params, "train_s": elapsed,
        "random_baseline": math.log(data.vocab_size),
        "L_depth_iter": Ldr.tolist(),
        "rstar": rs, "G_depth": G,
        "corr_rstar_depth_spearman": corr,
        "control_oracle": oracle,
        "control_final_loss": float(Ldr[:, -1].mean()),
        "control_state_reset": float(reset_L),
        "control_shuffle_route": float(shuffle_L),
        "control_noread_final": noread_final,
    }

    # --- print ---
    print(f"\nL[d, r]  (random baseline = {out['random_baseline']:.2f})")
    for d, row in enumerate(Ldr, 1):
        print(f"  d={d}: {[round(x, 3) for x in row]}  "
              f"r*={rs[d-1]}  G={G[d-1]:+.3f}")
    print(f"corr(r*, d) Spearman = {corr:.2f}")
    print(f"oracle={oracle:.3f}  final={out['control_final_loss']:.3f}  "
          f"state_reset={float(reset_L):.3f}  "
          f"shuffle={float(shuffle_L):.3f}  noread={noread_final:.3f}")

    # sanity interpretation
    print("\n--- Interpretation ---")
    converged = out["control_final_loss"] < out["random_baseline"] * 0.7
    print(f"  Converged (<70% baseline): {'YES' if converged else 'NO — increase --steps'}")
    depth_signal = G[D_MAX - 1] > 0.02 and corr > 0
    print(f"  Depth signal (G_d>0.02 and corr>0): {'YES' if depth_signal else 'NO'}")
    reset_hurts = float(reset_L) > out["control_final_loss"] + 0.05
    print(f"  State-reset hurts (>+0.05): {'YES' if reset_hurts else 'NO'}")
    read_helps = noread_final > out["control_final_loss"] + 0.05
    print(f"  In-loop read helps (>+0.05): {'YES' if read_helps else 'NO'}")

    # --- save ---
    path = os.path.join(RESULTS, f"simple_depth_{weighting_kind}_s{seed}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    _figure(out, weighting_kind, seed)
    print(f"\nSaved: {path}")
    return out


def _figure(out, w, seed):
    Ldr = np.array(out["L_depth_iter"])
    D, R = Ldr.shape
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    im = axes[0].imshow(Ldr, aspect="auto", cmap="viridis_r", origin="lower",
                        extent=[0.5, R + 0.5, 0.5, D + 0.5],
                        vmin=0, vmax=out["random_baseline"])
    axes[0].set_xlabel("model iteration r")
    axes[0].set_ylabel("true task depth d")
    axes[0].set_title(f"SimpleWalk L[d,r]  ({w})")
    fig.colorbar(im, ax=axes[0], fraction=0.046, label="val loss")
    rs = out["rstar"]
    axes[0].plot(rs, range(1, D + 1), "w.-", lw=1.5, label="r*(d)")
    axes[0].axhline(y=0.5, color="w", lw=0.3, alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")

    axes[1].bar(range(1, D + 1), out["G_depth"], color="#2563eb", alpha=0.8)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xlabel("true task depth d")
    axes[1].set_ylabel("depth gain G_d = L(r=1) - L(r=R)")
    axes[1].set_title(f"Depth gain per task depth  (Spearman corr={out['corr_rstar_depth_spearman']:.2f})")
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(RESULTS, f"fig8b_simpledepth_{w}_s{seed}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Figure: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--weighting", default="end", choices=["equal", "linear", "end"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(steps=a.steps, weighting_kind=a.weighting, seed=a.seed)
