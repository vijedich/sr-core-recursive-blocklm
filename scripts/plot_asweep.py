"""A-Sweep-Visualisierung: nutzbare Rekursionstiefe als Funktion des Aktiv-Brains A=k.

Liest alle anytime_inference_srcore_b64*_s*.json (Sweep-Läufe, @15000) und plottet:
  (1) Kumulativer Anytime-Gewinn  L(r=1)-L(r) vs. r   — eine Linie pro k
  (2) Marginal-Gewinn  L(r-1)-L(r) vs. r              — wo die nutzbare Tiefe versiegt
  (3) Endqualität  min_r L(r)  vs. A=k                 — Qualitaet ueber dem A-Floor

Greift automatisch spaeter dazukommende Punkte (k32, Seeds s1/s2) mit.
Output: results/fig_asweep_depth.png
Nutzung:  python scripts/plot_asweep.py
"""
from __future__ import annotations
import glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
OUT     = os.path.join(RESULTS, "fig_asweep_depth.png")

STYLE = {"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
         "axes.spines.top": False, "axes.spines.right": False}
plt.rcParams.update(STYLE)

MARGINAL_THRESH = 0.005  # Schwelle "nutzbarer" Schritt (nats)


def load_runs():
    """Liest Sweep-Evals. Rueckgabe: Liste dicts {k, seed, r[], cum[], marg[], best_L}."""
    runs = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "anytime_inference_srcore_b64*_s*.json"))):
        d = json.load(open(path, encoding="utf-8"))
        if d.get("step") != 15000:
            continue
        rows = sorted(d["rows"], key=lambda x: x["r"])
        k = rows[0]["block_apps"]                      # block_apps bei r=1 == k
        seed_m = re.search(r"_s(\d+)\.json$", path)
        seed = int(seed_m.group(1)) if seed_m else 0
        r   = [x["r"] for x in rows]
        L   = [x["Lfin_seen"] for x in rows]
        cum = [x["anytime_seen"] for x in rows]        # L(1)-L(r), konsistente Metrik
        # Marginal aus cum ableiten (NICHT aus Lfin_seen — das ist eine andere, verrauschte Messung)
        marg = [0.0] + [cum[i] - cum[i-1] for i in range(1, len(cum))]  # Gewinn Schritt r-1 -> r
        runs.append({"k": k, "seed": seed, "r": r, "L": L,
                     "cum": cum, "marg": marg, "best_L": min(L)})
    return sorted(runs, key=lambda x: (x["k"], x["seed"]))


def main():
    runs = load_runs()
    if not runs:
        print("Keine Sweep-Evals gefunden (results/anytime_inference_srcore_b64*_s*.json).")
        return
    print(f"{len(runs)} Sweep-Lauf/Laeufe geladen: " +
          ", ".join(f"k{r['k']}s{r['seed']}" for r in runs))

    cmap = plt.get_cmap("viridis")
    ks = sorted({r["k"] for r in runs})
    kcol = {k: cmap(i / max(len(ks) - 1, 1)) for i, k in enumerate(ks)}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (1) kumulativer Gewinn
    ax = axes[0]
    for run in runs:
        ax.plot(run["r"], run["cum"], "o-", color=kcol[run["k"]],
                label=f"k={run['k']} (A={run['k']}·blk)" + (f" s{run['seed']}" if run['seed'] else ""))
    ax.set_xlabel("Rekursionsschritt r"); ax.set_ylabel("kumul. Gewinn  L(1)−L(r)  [nats]")
    ax.set_title("(1) Anytime-Gewinn — mehr ist besser")
    ax.legend(fontsize=8)

    # (2) marginaler Gewinn pro Schritt + Schwelle
    ax = axes[1]
    for run in runs:
        ax.plot(run["r"], run["marg"], "o-", color=kcol[run["k"]])
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(MARGINAL_THRESH, color="crimson", ls="--", lw=0.8,
               label=f"Schwelle {MARGINAL_THRESH}")
    ax.set_xlabel("Rekursionsschritt r"); ax.set_ylabel("Marginal  L(r−1)−L(r)  [nats]")
    ax.set_title("(2) Wo die nutzbare Tiefe versiegt")
    ax.legend(fontsize=8)

    # (3) Endqualitaet vs A
    ax = axes[2]
    for run in runs:
        ax.plot(run["k"], run["best_L"], "o", color=kcol[run["k"]], markersize=10)
        ax.annotate(f"k{run['k']}" + (f"s{run['seed']}" if run['seed'] else ""),
                    (run["k"], run["best_L"]), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Aktiv-Brain  A = k  (log2)"); ax.set_ylabel("beste Endqualitaet  min_r L(r)  [nats]")
    ax.set_title("(3) Qualitaet vs. A — kleiner=besser")

    fig.suptitle("A-Sweep: Rekursionstiefe vs. Aktiv-Brain (b64, R6, 15k, HeteroMini)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gespeichert: {OUT}")

    # Konsolen-Zusammenfassung: nutzbare Tiefe pro Lauf
    print("\nNutzbare Tiefe (letztes r mit Marginal > "
          f"{MARGINAL_THRESH}):")
    for run in runs:
        useful = [run["r"][i] for i in range(1, len(run["r"])) if run["marg"][i] > MARGINAL_THRESH]
        depth = max(useful) if useful else 1
        print(f"  k={run['k']:<3} s{run['seed']}:  Tiefe r~{depth}   "
              f"Gesamt-Gewinn={run['cum'][-1]:+.4f}   best_L={run['best_L']:.4f}")


if __name__ == "__main__":
    main()
