"""Build paper figures for entropy-minimization sweep.

Outputs (all in results/):
  fig_entropy_pareto.png        -- cache vs code-generalization Pareto
  fig_router_consolidation.png  -- lambda → routing metrics
  fig_dense_vs_sparse_quality.png -- dense quality upper bound
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

STYLE = {
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 150,
}
plt.rcParams.update(STYLE)

COLORS = {
    "ctrl":  "#607D8B",
    "lam001":"#90CAF9",
    "lam003":"#2196F3",
    "lam005":"#FF9800",
    "lam007":"#F44336",
    "H375":  "#AB47BC",
    "H370":  "#CE93D8",
    "dense": "#2E7D32",
}


def load(name: str) -> dict:
    return json.load(open(os.path.join(RESULTS, name)))


def code_ratio_ho(metrics: dict) -> float:
    lf = metrics["Lfin_heldout"]
    others = [v for k, v in lf.items() if k != "code"]
    return lf["code"] / (sum(others) / len(others))


# ─── Figure 1: Pareto ────────────────────────────────────────────────────────

def fig_pareto() -> None:
    points = [
        # (label, K24, code_ratio_HO, color, marker, zorder, ann_offset)
        ("ctrl",
         load("eval_compare_entmin_lam003.json")["metrics_a"]["lb"]["24"]["lru"],
         code_ratio_ho(load("eval_quality_lam003.json")["metrics_a"]),
         COLORS["ctrl"], "o", 3, (6, 4)),
        ("λ=0.003",
         load("eval_compare_entmin_lam003.json")["metrics_b"]["lb"]["24"]["lru"],
         code_ratio_ho(load("eval_quality_lam003.json")["metrics_b"]),
         COLORS["lam003"], "^", 5, (6, 4)),
        ("λ=0.005",
         load("eval_compare_entmin_lam005.json")["metrics_b"]["lb"]["24"]["lru"],
         code_ratio_ho(load("eval_quality_lam005.json")["metrics_b"]),
         COLORS["lam005"], "s", 5, (6, -14)),
        ("λ=0.007",
         load("eval_compare_entmin_lam007.json")["metrics_b"]["lb"]["24"]["lru"],
         code_ratio_ho(load("eval_quality_lam007.json")["metrics_b"]),
         COLORS["lam007"], "D", 5, (6, 8)),
        ("H=3.75",
         load("eval_compare_target_H375.json")["metrics_b"]["lb"]["24"]["lru"],
         code_ratio_ho(load("eval_quality_H375.json")["metrics_b"]),
         COLORS["H375"], "v", 3, (6, 6)),
        ("H=3.70",
         load("eval_compare_target_H370.json")["metrics_b"]["lb"]["24"]["lru"],
         code_ratio_ho(load("eval_quality_H370.json")["metrics_b"]),
         COLORS["H370"], "P", 3, (8, 6)),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    # draw Pareto-frontier line (ctrl → lam003 → lam005 → lam007)
    sweep = [p for p in points if p[0] in ("ctrl", "λ=0.003", "λ=0.005", "λ=0.007")]
    xs = [p[1] for p in sweep]
    ys = [p[2] for p in sweep]
    ax.plot(xs, ys, "--", color="#BDBDBD", lw=1.4, zorder=2)

    # regime annotations (before scatter so they land behind points)
    regime_labels = {
        "λ=0.003": ("balanced", (-4, -20)),
        "λ=0.005": ("cache\nsweet spot", (6, -32)),
        "λ=0.007": ("boundary", (6, -20)),
    }

    for label, k24, cr, color, marker, zo, off in points:
        hollow = label in ("H=3.75", "H=3.70")
        fc = "white" if hollow else color
        ax.scatter(k24, cr, s=130, marker=marker, color=fc,
                   edgecolors=color, linewidth=2.0 if hollow else 1.4, zorder=zo)
        # primary label
        ax.annotate(label, (k24, cr), xytext=off,
                    textcoords="offset points", fontsize=9, color=color,
                    fontweight="bold" if not hollow else "normal",
                    alpha=0.65 if hollow else 1.0)
        # regime sub-label
        if label in regime_labels:
            sub, sub_off = regime_labels[label]
            ax.annotate(sub, (k24, cr), xytext=sub_off,
                        textcoords="offset points", fontsize=7.5,
                        color=color, fontstyle="italic", alpha=0.85)

    # shaded "lam003 dominates ctrl" region
    ctrl_k24 = points[0][1]; ctrl_cr = points[0][2]
    lam3_k24 = points[1][1]; lam3_cr = points[1][2]
    ax.fill_between([min(ctrl_k24, lam3_k24) - 5, max(ctrl_k24, lam3_k24) + 5],
                    [min(ctrl_cr,  lam3_cr)  - 0.002, min(ctrl_cr,  lam3_cr)  - 0.002],
                    [max(ctrl_cr,  lam3_cr)  + 0.002, max(ctrl_cr,  lam3_cr)  + 0.002],
                    color=COLORS["lam003"], alpha=0.10, zorder=1)

    ax.set_xlabel("K=24 LRU bytes/token (KB)  [lower = better cache →]", fontsize=11)
    ax.set_ylabel("Code-Ratio held-out  [lower = better generalization ↓]", fontsize=11)
    ax.set_title("Entropy Minimization Pareto: Cache vs. Code Generalization", fontsize=12)

    ax.set_xlim(1110, 980)  # inverted: high KB on left, better cache on right

    note = "Code-ratio estimates use 40-batch evals; interpret small vertical differences cautiously."
    ax.text(0.02, 0.015, note, transform=ax.transAxes,
            fontsize=7.5, color="#999", va="bottom", style="italic")

    fig.tight_layout()
    out = os.path.join(RESULTS, "fig_entropy_pareto.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ─── Figure 2: Router consolidation ──────────────────────────────────────────

def fig_router_consolidation() -> None:
    lam_configs = [
        (0.000, "eval_compare_entmin.json",       "a"),
        (0.001, "eval_compare_entmin.json",       "b"),
        (0.003, "eval_compare_entmin_lam003.json","b"),
        (0.005, "eval_compare_entmin_lam005.json","b"),
        (0.007, "eval_compare_entmin_lam007.json","b"),
    ]

    lams, entropies, u_cores, hard_ovs, k24s = [], [], [], [], []
    for lam, fname, side in lam_configs:
        m = load(fname)["metrics_" + side]
        lams.append(lam)
        entropies.append(m["router_entropy"])
        u_cores.append(m["unique_cores"])
        hard_ovs.append(m["hard_overlap_eval"])
        k24s.append(m["lb"]["24"]["lru"])

    lams = np.array(lams)
    regime_colors = [(0.003, COLORS["lam003"]), (0.005, COLORS["lam005"]), (0.007, COLORS["lam007"])]

    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    fig.suptitle(
        "Entropy pressure consolidates routing and improves cache behavior",
        fontsize=12, y=1.01
    )

    panels = [
        (axes[0, 0], entropies, "Router entropy H(p)", "↓ more concentrated"),
        (axes[0, 1], u_cores,   "Unique core combinations", "↓ better LRU locality"),
        (axes[1, 0], k24s,      "K=24 LRU KB/token (KB)", "↓ better cache"),
        (axes[1, 1], hard_ovs,  "Hard core overlap", "↑ more cross-token reuse"),
    ]

    for ax, vals, ylabel, subtitle in panels:
        ax.plot(lams * 1000, vals, "o-", color="#1565C0",
                lw=2.2, ms=8, mec="white", mew=1.8, zorder=4)
        ax.set_xlabel("λ × 1000", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(subtitle, fontsize=9, color="#555")
        ax.set_xticks(lams * 1000)
        ax.set_xticklabels(["0", "1", "3", "5", "7"], fontsize=9)

        for lam_mark, col in regime_colors:
            ax.axvline(lam_mark * 1000, ls="--", lw=1.1, color=col, alpha=0.65, zorder=2)

    # regime labels on bottom-right panel
    ax_lr = axes[1, 1]
    y_lo = min(hard_ovs) - 0.0003
    for lam_mark, col, label in [
        (0.003, COLORS["lam003"], "gen.\nbalanced"),
        (0.005, COLORS["lam005"], "cache\nsweet\nspot"),
        (0.007, COLORS["lam007"], "break-\npoint"),
    ]:
        ax_lr.text(lam_mark * 1000 + 0.05, y_lo, label,
                   fontsize=7, color=col, ha="left", va="bottom")

    fig.tight_layout()
    out = os.path.join(RESULTS, "fig_router_consolidation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ─── Figure 3: Dense vs Sparse quality ───────────────────────────────────────

def fig_dense_vs_sparse() -> None:
    # dense+ctrl share the same eval batches → valid delta; entmin models from own paired runs
    d_dc   = load("eval_quality_dense_vs_ctrl.json")
    d_lam3 = load("eval_quality_dense_vs_lam003.json")
    d_lam5 = load("eval_quality_dense_vs_lam005.json")

    # All four models, heldout only
    model_data = [
        ("Dense d24 (quality upper bound)", d_dc["metrics_a"],  COLORS["dense"]),
        ("SR-Core ctrl",                    d_dc["metrics_b"],  COLORS["ctrl"]),
        ("+ entmin λ=0.003",                d_lam3["metrics_b"], COLORS["lam003"]),
        ("+ entmin λ=0.005",                d_lam5["metrics_b"], COLORS["lam005"]),
    ]

    domains    = ["web", "wiki", "code", "lit"]
    dom_labels = ["Web", "Wiki", "Code", "Lit"]
    x  = np.arange(len(domains))
    n  = len(model_data)
    w  = 0.19
    offsets = np.linspace(-(n - 1) / 2 * w, (n - 1) / 2 * w, n)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        "Dense Baseline vs. SR-Core Sparse — Held-out Loss (b64 k8 R6, @17k steps)",
        fontsize=12
    )

    for i, (label, m, color) in enumerate(model_data):
        vals = [m["Lfin_heldout"][d] for d in domains]
        ax.bar(x + offsets[i], vals, w, label=label,
               color=color, alpha=0.87, edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(dom_labels, fontsize=12)
    ax.set_ylabel("Held-out loss  (↓ better)", fontsize=11)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.set_ylim(bottom=3.8)

    # Annotate dense advantage on code (paired run, same batches → tight estimate)
    code_dense = d_dc["metrics_a"]["Lfin_heldout"]["code"]
    code_ctrl  = d_dc["metrics_b"]["Lfin_heldout"]["code"]
    code_x     = 2 + offsets[0]
    ax.annotate(
        f"Δ={code_ctrl - code_dense:.2f} nats\n(ctrl − dense)",
        xy=(code_x, code_dense + 0.03),
        xytext=(code_x + 0.55, code_dense + 0.28),
        fontsize=8, color=COLORS["dense"],
        arrowprops=dict(arrowstyle="->", color=COLORS["dense"], lw=1.0),
    )

    note = ("Dense remains the quality upper bound across all held-out domains.\n"
            "Entropy-minimized sparse variants are evaluated for systems behavior "
            "(cache/routing), not dense-level quality.")
    ax.text(0.01, 0.01, note, transform=ax.transAxes,
            fontsize=7.5, color="#999", va="bottom", style="italic")

    fig.tight_layout()
    out = os.path.join(RESULTS, "fig_dense_vs_sparse_quality.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building figures...")
    fig_pareto()
    fig_router_consolidation()
    fig_dense_vs_sparse()
    print("Done — all figures in results/")
