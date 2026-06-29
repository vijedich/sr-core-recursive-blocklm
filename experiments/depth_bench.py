"""Experiment 1 — does recursive depth do useful sequential computation?

Hard task with KNOWN true depth d_q (rblm/depth_tasks.py). Trains Model C with an
in-loop attention read so depth is architecturally possible. Logs L[d, r] (true
depth x model iteration) and runs the four controls from the plan:

  * oracle depth   : eval each example at iteration = its true depth (upper bound)
  * state reset    : reset state each iteration -> kills build-up
  * shuffle route  : random block selection -> tests learned routing/order
  * width vs depth : 2x8 / 4x4 / 8x2 (iters x active blocks) at matched compute

Central outputs: heatmap L[d,r], r*_d vs d with corr(r*_d, d), depth gain G_d.

NOTE ON COMPUTE: convergence on this task needs a GPU / many thousands of steps;
on a 1-CPU sandbox it stays near chance. Run with --steps small only to validate
execution; use --steps >= 4000 on GPU for scientific results. Default protocol:
3 weightings x 3 seeds, R=8, D_max=8.
"""
from __future__ import annotations
import argparse, os, json, math, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rblm.config import ModelConfig
from rblm.models import build_model, iteration_losses, weighting
from rblm.depth_tasks import DepthWalk

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def make_model(data, R, k, n_blocks, read, d_model=128, hidden=256, device="cpu"):
    cfg = ModelConfig(vocab_size=data.vocab_size, d_model=d_model, block_hidden=hidden,
                      n_heads=4, context_layers=1, max_len=48, variant="C",
                      n_blocks=n_blocks, k_active=k, routed_iters=R, key_dim=48,
                      router_noise_std=0.4, recurrent_read=read)
    return build_model(cfg).to(device)


def train(model, data, steps, weighting_kind, seed, lr=3e-3, bs=64, seq=48, device="cpu"):
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    R = model.n_iters
    w = weighting(R, weighting_kind).to(device)
    for step in range(steps):
        t, tg, mask, dep = data.batch(bs, seq, device=device)
        logits, aux = model(t)
        per = iteration_losses(logits, tg, mask)
        loss = (w * per).sum() + 0.01 * sum(a["lb_loss"] for a in aux["iters"]) / R
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


@torch.no_grad()
def loss_depth_iter(model, data, D_max, batches=6, bs=64, seq=48, device="cpu", **fwd):
    """Returns L[d, r] matrix (true depth x iteration)."""
    R = model.n_iters
    L = np.zeros((D_max, R)); n = np.zeros(D_max)
    model.eval()
    for d in range(1, D_max + 1):
        for _ in range(batches):
            t, tg, mask, dep = data.batch(bs, seq, device=device, depths=[d] * bs)
            logits, _ = model(t, **fwd) if fwd else model(t)
            per = iteration_losses(logits, tg, mask)
            m = int(mask.sum()); L[d - 1] += per.cpu().numpy() * m; n[d - 1] += m
    return L / n[:, None]


@torch.no_grad()
def oracle_loss(model, data, D_max, batches=6, bs=64, seq=48, device="cpu"):
    """Eval each example at iteration index = min(true_depth, R)."""
    R = model.n_iters; tot = 0.0; cnt = 0
    model.eval()
    for d in range(1, D_max + 1):
        r_idx = min(d, R) - 1
        for _ in range(batches):
            t, tg, mask, dep = data.batch(bs, seq, device=device, depths=[d] * bs)
            logits, _ = model(t)
            per = iteration_losses([logits[r_idx]], tg, mask)
            m = int(mask.sum()); tot += float(per[0]) * m; cnt += m
    return tot / cnt


def rstar(row, eps=0.03):
    final = row[-1]
    for r, v in enumerate(row):
        if v <= final + eps:
            return r + 1
    return len(row)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra * rb).sum() / (np.sqrt((ra**2).sum() * (rb**2).sum()) + 1e-9))


def run(steps=200, weighting_kind="end", seed=0, D_max=6, R=6, smoke=False, device="cpu",
        max_distract=4, k=4):
    data = DepthWalk(base=8, d_max=D_max, max_distract=max_distract, seed=1)
    out = {"steps": steps, "weighting": weighting_kind, "seed": seed,
           "D_max": D_max, "R": R, "device": device, "max_distract": max_distract, "k": k}

    # main model (in-loop read)
    model = make_model(data, R=R, k=k, n_blocks=24, read=True, device=device)
    t0 = time.time()
    train(model, data, steps, weighting_kind, seed, device=device)
    out["train_s"] = time.time() - t0

    Ldr = loss_depth_iter(model, data, D_max, device=device)
    out["L_depth_iter"] = Ldr.tolist()
    rs = [rstar(Ldr[d]) for d in range(D_max)]
    depths = list(range(1, D_max + 1))
    out["rstar"] = rs
    out["G_depth"] = [float(Ldr[d, 0] - Ldr[d, -1]) for d in range(D_max)]
    out["corr_rstar_depth_spearman"] = spearman(np.array(rs), np.array(depths))

    # controls
    out["control_oracle_loss"] = oracle_loss(model, data, D_max, device=device)
    out["control_fixedR_loss"] = float(loss_depth_iter(model, data, D_max, device=device)[:, -1].mean())
    out["control_state_reset_L"] = loss_depth_iter(
        model, data, D_max, device=device, state_reset=True)[:, -1].mean().item()
    out["control_shuffle_route_L"] = loss_depth_iter(
        model, data, D_max, device=device, random_route=True)[:, -1].mean().item()

    if not smoke:
        # width-vs-depth grid at matched compute (apps = iters*k = 16)
        grid = {}
        for R_, k_ in [(2, 8), (4, 4), (8, 2)]:
            m = make_model(data, R=R_, k=k_, n_blocks=24, read=True, device=device)
            train(m, data, steps, weighting_kind, seed, device=device)
            Lg = loss_depth_iter(m, data, D_max, device=device)
            grid[f"{R_}x{k_}"] = float(Lg[:, -1].mean())
        out["width_depth_grid"] = grid

    tag = f"depth_{weighting_kind}_d{D_max}dist{max_distract}_s{seed}"
    with open(os.path.join(RESULTS, f"{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    out["_tag"] = tag
    _figure(out)
    _print(out)
    return out


def _figure(out):
    Ldr = np.array(out["L_depth_iter"]); D_max, R = Ldr.shape
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    im = axes[0].imshow(Ldr, aspect="auto", cmap="viridis_r", origin="lower",
                        extent=[0.5, R + 0.5, 0.5, D_max + 0.5])
    axes[0].set_xlabel("model iteration r"); axes[0].set_ylabel("true task depth d")
    axes[0].set_title("Exp1 central plot: L[d, r]\n(diagonal = depth-gated computation)")
    fig.colorbar(im, ax=axes[0], fraction=0.046, label="val loss")
    axes[0].plot(out["rstar"], range(1, D_max + 1), "w.-", lw=1.5, label="r*(d)")
    axes[0].legend(fontsize=8, loc="upper right")

    axes[1].plot(range(1, D_max + 1), out["rstar"], "o-", color="#2563eb")
    axes[1].plot([1, D_max], [1, D_max], "k:", alpha=.5, label="ideal r*=d")
    axes[1].set_xlabel("true task depth d"); axes[1].set_ylabel("optimal stop depth r*(d)")
    axes[1].set_title(f"r*(d) vs d  (Spearman = {out['corr_rstar_depth_spearman']:.2f})")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
    fig.tight_layout()
    tag = out.get("_tag", f"fig8_depth_{out['weighting']}")
    fig.savefig(os.path.join(RESULTS, f"{tag}.png"), dpi=130)
    plt.close(fig)


def _print(out):
    print(f"\n[Exp1 {out['weighting']} seed{out['seed']} steps={out['steps']} "
          f"({out['train_s']:.0f}s)]")
    print("L[d,r]:")
    for d, row in enumerate(out["L_depth_iter"], 1):
        print(f"  d={d}: {[round(x,2) for x in row]}  r*={out['rstar'][d-1]} G={out['G_depth'][d-1]:+.2f}")
    print(f"corr(r*,d) Spearman = {out['corr_rstar_depth_spearman']:.2f}")
    print(f"controls: oracle={out['control_oracle_loss']:.3f} fixedR={out['control_fixedR_loss']:.3f} "
          f"state_reset={out['control_state_reset_L']:.3f} shuffle_route={out['control_shuffle_route_L']:.3f}")
    if "width_depth_grid" in out:
        print("width-depth (matched compute, final loss):", out["width_depth_grid"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--weighting", default="end", choices=["equal", "linear", "end"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--D_max", type=int, default=6)
    ap.add_argument("--R", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_distract", type=int, default=4)
    ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args()
    run(steps=a.steps, weighting_kind=a.weighting, seed=a.seed,
        D_max=a.D_max, R=a.R, smoke=a.smoke, device=a.device,
        max_distract=a.max_distract, k=a.k)
