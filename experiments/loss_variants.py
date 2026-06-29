"""Exp1a: does the loss weighting (not the data) cause the flat anytime curve?
Compares equal / linear / end iteration-weighting for Model C on the synthetic
task. Produces a figure and prints the depth-gain table."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
REGS = ["REPEAT", "INCR", "FIB", "ALT"]
TAGS = {"equal": "C_routed_s0", "linear": "C_routed_s0_linear", "end": "C_routed_s0_end"}
COL = {"equal": "#6b7280", "linear": "#2563eb", "end": "#dc2626"}


def main():
    data = {k: json.load(open(os.path.join(RESULTS, f"{t}.json"))) for k, t in TAGS.items()}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    ax = axes[0]
    for k, d in data.items():
        pi = d["final"]["loss_per_iter"]
        ax.plot(range(1, len(pi) + 1), pi, "o-", color=COL[k], lw=2,
                label=f"{k} (gap {pi[0]-pi[-1]:+.3f})")
    ax.set_xlabel("iteration r"); ax.set_ylabel("aggregate val loss")
    ax.set_title("Anytime curve vs loss weighting\n(flat under equal = loss artifact)")
    ax.grid(alpha=.3); ax.legend(fontsize=8)

    ax = axes[1]
    x = np.arange(len(REGS)); w = 0.25
    for i, (k, d) in enumerate(data.items()):
        m = np.array(d["per_regime_base"])
        g = m[:, 0] - m[:, -1]
        ax.bar(x + (i - 1) * w, g, w, color=COL[k], label=k)
    ax.axhline(0, color="k", lw=.6)
    ax.set_xticks(x); ax.set_xticklabels(REGS, fontsize=8)
    ax.set_ylabel("depth gain  G = L(iter1) - L(iter_final)")
    ax.set_title("Per-regime depth gain\n(REPEAT uses depth; trivial regimes don't)")
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig7_loss_variants.png"), dpi=130)
    print("wrote fig7_loss_variants.png")

    print("\nanytime gap (L1 - Lfinal):")
    for k, d in data.items():
        pi = d["final"]["loss_per_iter"]
        print(f"  {k:7s}: {pi[0]-pi[-1]:+.3f}   curve={[round(x,3) for x in pi]}")


if __name__ == "__main__":
    main()
