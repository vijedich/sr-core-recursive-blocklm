"""Overnight-Forschungspipeline (losgeloest startbar, GPU-seriell, idempotent).

Reihenfolge (RTX 2060, 6 GB -> nur EIN GPU-Job gleichzeitig):
  0. Warte bis Seed-3-Training fertig ist (extern gestartet)
  1. Kompetenzanalyse Seed 2   (curriculum_fromIter2)
  2. Kompetenzanalyse Seed 3   (curriculum_fromIter2)
  3. Seed-0-Regen              (curriculum_fromIter2; Kopien-Regel: Backup vorher)
  4. No-Diversity-Warmstart    (Training seed 0, 3000 Schritte ab phase2)
  5. Kompetenzanalyse No-Div   (tinystories_nodiv_warmstart)

Idempotent: jeder Schritt prueft sein Zielartefakt und ueberspringt, wenn vorhanden.
Training nutzt Zwischen-Checkpoints + Resume, falls ein Lauf doch gekillt wird.

Start (losgeloest):
  Start-Process python -ArgumentList '-u','scripts/run_pipeline.py'
    -WorkingDirectory <root> -RedirectStandardOutput results/_pipeline.out
    -RedirectStandardError results/_pipeline.err -WindowStyle Hidden
"""
import os, sys, time, glob, shutil, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PY = sys.executable
LOG = os.path.join(ROOT, "results", "_pipeline.log")
CURR = "tinystories_curriculum_fromIter2"
NODIV = "tinystories_nodiv_warmstart"


def log(m):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + "  " + m
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_cmd(args):
    log("RUN: " + " ".join(args))
    r = subprocess.run([PY] + args, cwd=ROOT)
    log(f"  -> exit={r.returncode}")
    return r.returncode


def wait_for(path, label):
    log(f"Warte auf {label}")
    w = 0
    while not os.path.exists(path):
        time.sleep(30)
        w += 30
        if w % 300 == 0:
            log(f"  ... warte weiter ({w//60} min) auf {label}")
    log(f"{label} vorhanden.")
    time.sleep(20)  # GPU-Freigabe-Puffer


def competence(seed, exp):
    out = os.path.join(ROOT, "results",
                       f"competence_b64k4R6_s{seed}_analysis_{exp}.json")
    if os.path.exists(out):
        log(f"SKIP Kompetenzanalyse s{seed}/{exp} (Ziel existiert)")
        return
    ckpt = f"checkpoints/{exp}/seed_{seed}/step_3000"
    if not os.path.exists(os.path.join(ROOT, ckpt, "model.pt")):
        log(f"!! Checkpoint fehlt fuer s{seed}/{exp} ({ckpt}) — uebersprungen")
        return
    log(f"=== Kompetenzanalyse Seed {seed} / {exp} ===")
    run_cmd(["-m", "experiments.competence_centers_exp",
             "--ckpt", ckpt, "--analysis", "--device", "cuda"])


def backup_existing(seed, exp):
    """Kopien-Regel: bestehende Zielartefakte sichern, bevor sie ueberschrieben werden."""
    pat = os.path.join(ROOT, "results",
                       f"competence_b64k4R6_s{seed}_analysis_{exp}*")
    hits = glob.glob(pat)
    if hits:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for h in hits:
            dst = h + f".bak_{stamp}"
            shutil.copy2(h, dst)
        log(f"Kopien-Regel: {len(hits)} Datei(en) gesichert (.bak_{stamp})")


def train_nodiv():
    final = os.path.join(ROOT, "checkpoints", NODIV, "seed_0", "step_3000", "model.pt")
    if os.path.exists(final):
        log("SKIP No-Div-Training (finaler Checkpoint existiert)")
        return
    args = ["-m", "experiments.tinystories_exp",
            "--pretrained_ckpt", "checkpoints/tinystories_phase2/seed_0/step_3000",
            "--steps", "3000", "--seed", "0", "--exp_name", NODIV, "--device", "cuda"]
    log("=== No-Diversity-Warmstart Training (seed 0, 3000 Schritte) ===")
    run_cmd(args)


def main():
    open(LOG, "w", encoding="utf-8").close()
    log(f"Pipeline gestartet (PID {os.getpid()})")

    # 0. Auf externes Seed-3-Training warten
    wait_for(os.path.join(ROOT, "checkpoints", CURR, "seed_3", "step_3000", "model.pt"),
             "Seed-3-Training (step_3000)")

    # 1-2. Kompetenzanalysen der Diversity-Seeds
    competence(2, CURR)
    competence(3, CURR)

    # 3. Seed-0-Regen (vom Overwrite-Bug getroffen) — mit Backup-Regel
    backup_existing(0, CURR)
    competence(0, CURR)

    # 4-5. No-Diversity-Warmstart: Training + Analyse (Kontrollbedingung)
    train_nodiv()
    competence(0, NODIV)

    log("PIPELINE FERTIG.")


if __name__ == "__main__":
    main()
