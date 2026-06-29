"""Faire Modellmatrix (losgeloest, GPU-seriell, idempotent).

Frage: Kann ein kleiner rekursiv wiederverwendeter Kern (SR-Fixed-Core) bei konstanter
aktiver Breite die Tiefe eines dichten Modells teilweise ersetzen? Bankgroesse klein genug,
dass die Parameterzahl fair bleibt. R=2/4/6 DIREKT trainiert (nicht nachtraeglich gekuerzt).

Matrix (alle: TinyStories, 3000 Schritte, k=4, seed 0, End-Gewichtung, gleiche Pipeline):
  Dense        : d4, d12          (d8, d24 bereits vorhanden)
  Naked Sparse : b16 R2/R4/R6, b32 R2/R4/R6     (core_mode=None, keine Diversity)
  SR-Core      : b32 R2/R4/R6     (core_mode=per_token)
  Core+Satellit: b32 R2/R4/R6     (core_mode=core_satellite)

Idempotent (Skip bei finalem Checkpoint), Resume aus Zwischen-Checkpoint.

Start (losgeloest):
  Start-Process python -ArgumentList '-u','scripts/matrix_sweep.py'
    -WorkingDirectory <root> -RedirectStandardOutput results/_matrix.out
    -RedirectStandardError results/_matrix.err -WindowStyle Hidden
"""
import os, sys, time, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PY = sys.executable
LOG = os.path.join(ROOT, "results", "_matrix.log")
SEED = 0


def log(m):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + "  " + m
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def done(exp):
    return os.path.exists(os.path.join(ROOT, "checkpoints", exp, f"seed_{SEED}",
                                        "step_3000", "model.pt"))


def run(exp, args):
    if done(exp):
        log(f"SKIP {exp} (finaler Checkpoint existiert)")
        return
    log(f"=== {exp} START ===")
    r = subprocess.run([PY] + args, cwd=ROOT)
    log(f"=== {exp} exit={r.returncode} ===")


def dense(depth):
    exp = f"dense_d{depth}"
    run(exp, ["-m", "experiments.train_dense", "--dense_depth", str(depth),
              "--seed", str(SEED), "--device", "cuda"])


def sparse(n, R, mode=None):
    tag = {None: "naked", "per_token": "srcore", "core_satellite": "srsat"}[mode]
    exp = f"{tag}_b{n}_R{R}"
    args = ["-m", "experiments.tinystories_exp", "--n_blocks", str(n), "--k", "4",
            "--R", str(R), "--steps", "3000", "--seed", str(SEED), "--exp_name", exp,
            "--device", "cuda"]
    if mode:
        args += ["--core_mode", mode]
    run(exp, args)


def main():
    open(LOG, "w", encoding="utf-8").close()
    log(f"Matrix-Sweep gestartet (PID {os.getpid()})")

    # Dense-Ergaenzung (d8, d24 schon da)
    for d in (4, 12):
        dense(d)
    # Naked Sparse (Kontrolle)
    for n in (16, 32):
        for R in (2, 4, 6):
            sparse(n, R, None)
    # SR-Core (Kern-Reuse) + Core+Satellit
    for R in (2, 4, 6):
        sparse(32, R, "per_token")
    for R in (2, 4, 6):
        sparse(32, R, "core_satellite")

    log("MATRIX-SWEEP FERTIG.")


if __name__ == "__main__":
    main()
