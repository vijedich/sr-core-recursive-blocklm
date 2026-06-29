"""Langer HeteroMini-Lauf, reduzierte Matrix (losgeloest, GPU-seriell, idempotent).

Frage: Haelt die SR-Core-Geometrie nach ernsthaftem Training? Pro Config: bis max_steps
trainieren, an Meilensteinen 5k/10k/20k/50k inline auswerten (Verlaufstabelle).

Reihenfolge so, dass die primaere Vergleichsgruppe zuerst fertig ist:
  srcore_b32_R6 (primär) -> naked_b32_R6 -> dense_d24 -> srcore_b64_R6 -> srcore_b32_R2 (optional)

Idempotent: ueberspringt Config, wenn finaler Snapshot existiert. Trajectory-JSON wird je
Meilenstein inkrementell geschrieben (Teilfortschritt bleibt sichtbar).

Start (losgeloest):
  Start-Process python -ArgumentList '-u','scripts/heteromini_long_matrix.py'
    -WorkingDirectory <root> -RedirectStandardOutput results/_hm_long.out
    -RedirectStandardError results/_hm_long.err -WindowStyle Hidden
"""
import os, sys, time, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PY = sys.executable
LOG = os.path.join(ROOT, "results", "_hm_long.log")
MAX_STEPS = 10000  # Segment 5k->10k; heteromini_long setzt automatisch vom 5k-Snapshot fort.

# Reduziert auf die drei relevanten Modelle (b64/R2 fallengelassen: b64 ohne Nutzen,
# R2 beantwortet die Rekursionsfrage nicht).  (exp, cli-args)
JOBS = [
    ("hm_srcore_b32_R6", ["--variant", "sparse", "--n_blocks", "32", "--R", "6", "--core_mode", "per_token"]),
    ("hm_naked_b32_R6",  ["--variant", "sparse", "--n_blocks", "32", "--R", "6"]),
    ("hm_dense_d24",     ["--variant", "dense", "--depth", "24"]),
]
# Modelle fuer die Rekursionsgewinn-Analyse (per Domaene/Beispiel) nach dem Training.
GAIN_EXPS = ["hm_srcore_b32_R6", "hm_naked_b32_R6", "hm_dense_d24"]


def log(m):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + "  " + m
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    open(LOG, "w", encoding="utf-8").close()
    log(f"HeteroMini-LONG gestartet (PID {os.getpid()}), max_steps={MAX_STEPS}")
    # heteromini_long setzt vom Continuation-Snapshot fort bzw. ueberspringt, wenn schon am Ziel.
    for exp, args in JOBS:
        log(f"=== {exp} START ===")
        r = subprocess.run([PY, "-m", "experiments.heteromini_long"] + args
                           + ["--max_steps", str(MAX_STEPS), "--seed", "0", "--device", "cuda"],
                           cwd=ROOT)
        log(f"=== {exp} exit={r.returncode} ===")
    # Rekursionsgewinn-Analyse (per Domaene/Beispiel) auf den 10k-Snapshots
    for exp in GAIN_EXPS:
        log(f"=== gain_analysis {exp} START ===")
        r = subprocess.run([PY, "-m", "experiments.gain_analysis", "--exp", exp,
                            "--device", "cuda", "--n_batches", "60"], cwd=ROOT)
        log(f"=== gain_analysis {exp} exit={r.returncode} ===")
    log("HETEROMINI-LONG FERTIG.")


if __name__ == "__main__":
    main()
