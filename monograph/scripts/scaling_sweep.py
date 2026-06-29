"""H1-Skalierungstest (losgeloest, GPU-seriell, idempotent).

Kernfrage (die eigentliche VRAM-Hypothese): Bleibt das aktive Working Set pro Token
klein (~7-8), wenn die GESAMTE Blockbank waechst? Oder waechst es mit der Bank mit?

Design: IDENTISCHE Hyperparameter, nur n_blocks variiert. From-scratch (kein Warmstart,
keine Diversity), damit der Bankgroessen-Effekt nicht von anderen Faktoren konfundiert ist.
  n_blocks in {64, 128, 256},  k=4, R=6,  3000 Schritte,  seed 0.

Gemessen pro Lauf (aus der eingebauten Auswertung): unique_blocks_per_token (Working Set),
hub_gini, dead_blocks, cache_sim (Miss-Rate gelernt vs. zufall), final_loss_per_iter.

Start (losgeloest):
  Start-Process python -ArgumentList '-u','scripts/scaling_sweep.py'
    -WorkingDirectory <root> -RedirectStandardOutput results/_scaling.out
    -RedirectStandardError results/_scaling.err -WindowStyle Hidden
"""
import os, sys, time, shutil, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PY = sys.executable
LOG = os.path.join(ROOT, "results", "_scaling.log")
BANKS = [64, 128, 256]
K, R, STEPS, SEED = 4, 6, 3000, 0


def log(m):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + "  " + m
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_bank(n):
    exp = f"tinystories_scale_b{n}"
    final = os.path.join(ROOT, "checkpoints", exp, f"seed_{SEED}", "step_3000", "model.pt")
    if os.path.exists(final):
        log(f"SKIP n_blocks={n} (finaler Checkpoint existiert)")
        return
    # Kopien-Regel: NUR die exakt ueberschriebenen Artefakte sichern (Tag ohne
    # Phase-/Curriculum-/Warm-Suffix, da from-scratch+plain). Praezise, nicht via "*".
    tag = f"tinystories_b{n}k{K}R{R}_s{SEED}"
    overwrite = [f"{tag}.json", f"{tag}_model.pt",
                 f"fig_exp3_{tag}.png", f"fig_exp3_ablation_{tag}.png"]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for fn in overwrite:
        h = os.path.join(ROOT, "results", fn)
        if os.path.exists(h):
            shutil.copy2(h, h + ".bak_" + stamp)
            log(f"Kopien-Regel: gesichert {fn}")
    args = ["-m", "experiments.tinystories_exp",
            "--n_blocks", str(n), "--k", str(K), "--R", str(R),
            "--steps", str(STEPS), "--seed", str(SEED),
            "--exp_name", exp, "--device", "cuda"]
    log(f"=== Skalierung n_blocks={n} START ===")
    r = subprocess.run([PY] + args, cwd=ROOT)
    log(f"=== n_blocks={n} exit={r.returncode} ===")


def main():
    open(LOG, "w", encoding="utf-8").close()
    log(f"Skalierungs-Sweep gestartet (PID {os.getpid()}) — Baenke {BANKS}, k={K}, R={R}")
    for n in BANKS:
        run_bank(n)
    log("SKALIERUNGS-SWEEP FERTIG.")


if __name__ == "__main__":
    main()
