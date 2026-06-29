"""Dense-Baselines fuer den CPU-Benchmark (losgeloest, GPU-seriell, idempotent).

Traint ModelA auf TinyStories in zwei Tiefen:
  depth=24  -> compute-matched zu Sparse R=6 (k*R=24 Blockanwendungen/Token)
  depth=8   -> Effektivbreite (~ eindeutiges aktives Set der Sparse-Modelle)

Idempotent (Skip wenn finaler Checkpoint da), Resume aus Zwischen-Checkpoint.

Start (losgeloest):
  Start-Process python -ArgumentList '-u','scripts/dense_baselines.py'
    -WorkingDirectory <root> -RedirectStandardOutput results/_dense.out
    -RedirectStandardError results/_dense.err -WindowStyle Hidden
"""
import os, sys, time, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PY = sys.executable
LOG = os.path.join(ROOT, "results", "_dense.log")
DEPTHS = [24, 8]
SEED = 0


def log(m):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + "  " + m
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_depth(depth):
    final = os.path.join(ROOT, "checkpoints", f"dense_d{depth}", f"seed_{SEED}",
                         "step_3000", "model.pt")
    if os.path.exists(final):
        log(f"SKIP dense depth={depth} (finaler Checkpoint existiert)")
        return
    args = ["-m", "experiments.train_dense", "--dense_depth", str(depth),
            "--seed", str(SEED), "--device", "cuda"]
    log(f"=== Dense depth={depth} START ===")
    r = subprocess.run([PY] + args, cwd=ROOT)
    log(f"=== dense depth={depth} exit={r.returncode} ===")


def main():
    open(LOG, "w", encoding="utf-8").close()
    log(f"Dense-Baselines gestartet (PID {os.getpid()}) — Tiefen {DEPTHS}")
    for d in DEPTHS:
        run_depth(d)
    log("DENSE-BASELINES FERTIG.")


if __name__ == "__main__":
    main()
