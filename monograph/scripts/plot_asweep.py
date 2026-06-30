"""A-sweep visualization: useful recursion depth as a function of active brain A=k.

Reads all anytime_inference_srcore_b64*_s*.json sweep runs (@15000) and plots:
  (1) Cumulative anytime gain   L(r=1)-L(r) vs. r   — one line per k
  (2) Marginal gain             L(r-1)-L(r) vs. r   — where useful depth runs out
  (3) Final quality             min_r L(r)  vs. A=k — quality above the A-floor

New points (other k, seeds s1/s2) are picked up automatically.
Input:  data/eval/heteromini/anytime_inference_srcore_b64*_s*.json
Output: figures/fig_asweep_depth.png
Usage:  python monograph/scripts/plot_asweep.py
"""
from __future__ import annotations
import glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "eval", "heteromini")
FIG  = os.path.join(ROOT, "figures")
OUT  = os.path.join(FIG, "fig_asweep_depth.png")

STYLE = {"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
         "axes.spines.top": False, "axes.spines.right": False}
plt.rcParams.update(STYLE)

MARGINAL_THRESH = 0.005  # threshold for a "useful" step (nats)


def load_runs():
    """Read sweep evals. Returns list of dicts {k, seed, r[], cum[], marg[], best_L}."""
    runs = []
    for path in sorted(glob.glob(os.path.join(DATA, "anytime_inference_srcore_b64*_s*.json"))):
        d = json.load(open(path, encoding="utf-8"))
        if d.get("step") != 15000:
            continue
        rows = sorted(d["rows"], key=lambda x: x["r"])
        k = rows[0]["block_apps"]                      # block_apps at r=1 == k
        seed_m = re.search(r"_s(\d+)\.json$", path)
        seed = int(seed_m.group(1)) if seed_m else 0
        r   = [x["r"] for x in rows]
        L   = [x["Lfin_seen"] for x in rows]
        cum = [x["anytime_seen"] for x in rows]        # L(1)-L(r), consistent metric
        # Derive marginal from cum (NOT from Lfin_seen — that is a different, noisier measurement)
        marg = [0.0] + [cum[i] - cum[i-1] for i in range(1, len(cum))]  # gain step r-1 -> r
        runs.append({"k": k, "seed": seed, "r": r, "L": L,
                     "cum": cum, "marg": marg, "best_L": min(L)})
    return sorted(runs, key=lambda x: (x["k"], x["seed"]))


def main():
    runs = load_runs()
    if not runs:
        print("No sweep evals found (data/eval/heteromini/anytime_inference_srcore_b64*_s*.json).")
        return
    print(f"{len(runs)} sweep run(s) loaded: " +
          ", ".join(f"k{r['k']}s{r['seed']}" for r in runs))

    cmap = plt.get_cmap("viridis")
    ks = sorted({r["k"] for r in runs})
    kcol = {k: cmap(i / max(len(ks) - 1, 1)) for i, k in enumerate(ks)}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (1) cumulative gain
    ax = axes[0]
    for run in runs:
        ax.plot(run["r"], run["cum"], "o-", color=kcol[run["k"]],
                label=f"k={run['k']} (A={run['k']}·blk)" + (f" s{run['seed']}" if run['seed'] else ""))
    ax.set_xlabel("recursion step r"); ax.set_ylabel("cumulative gain  L(1)−L(r)  [nats]")
    ax.set_title("(1) Anytime gain — more is better")
    ax.legend(fontsize=8)

    # (2) marginal gain per step + threshold
    ax = axes[1]
    for run in runs:
        ax.plot(run["r"], run["marg"], "o-", color=kcol[run["k"]])
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(MARGINAL_THRESH, color="crimson", ls="--", lw=0.8,
               label=f"threshold {MARGINAL_THRESH}")
    ax.set_xlabel("recursion step r"); ax.set_ylabel("marginal  L(r−1)−L(r)  [nats]")
    ax.set_title("(2) Where useful depth runs out")
    ax.legend(fontsize=8)

    # (3) final quality vs A
    ax = axes[2]
    for run in runs:
        ax.plot(run["k"], run["best_L"], "o", color=kcol[run["k"]], markersize=10)
        ax.annotate(f"k{run['k']}" + (f"s{run['seed']}" if run['seed'] else ""),
                    (run["k"], run["best_L"]), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("active brain  A = k  (log2)"); ax.set_ylabel("best final quality  min_r L(r)  [nats]")
    ax.set_title("(3) Quality vs. A — smaller is better")

    fig.suptitle("A-sweep: recursion depth vs. active brain (b64, R6, 15k, HeteroMini)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT}")

    # Console summary: useful depth per run
    print(f"\nUseful depth (last r with marginal > {MARGINAL_THRESH}):")
    for run in runs:
        useful = [run["r"][i] for i in range(1, len(run["r"])) if run["marg"][i] > MARGINAL_THRESH]
        depth = max(useful) if useful else 1
        print(f"  k={run['k']:<3} s{run['seed']}:  depth r~{depth}   "
              f"total gain={run['cum'][-1]:+.4f}   best_L={run['best_L']:.4f}")


if __name__ == "__main__":
    main()
