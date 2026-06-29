"""Cross-Seed-Stabilitaet der Schluesselvarianten (losgeloest, GPU-seriell, idempotent).

Ziel (Viktors Sofort-Schritt): Ist SR-Core b32 R2 robust besser als Naked b32 R2, und bleibt
das Within-Run-anytime-Signal stabil? Nur die wichtigsten Configs, nicht die ganze Matrix.

Seeds 1 und 2 (Seed 0 liegt schon aus der Matrix vor -> wird idempotent uebersprungen):
  Dense d4, Dense d8
  Naked b32 R2/R6   (core_mode=None)
  SR-Core b32 R2/R6 (core_mode=per_token)

Start (losgeloest):
  Start-Process python -ArgumentList '-u','scripts/crossseed_sweep.py'
    -WorkingDirectory <root> -RedirectStandardOutput results/_crossseed.out
    -RedirectStandardError results/_crossseed.err -WindowStyle Hidden
"""
import os, sys, time, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PY = sys.executable
LOG = os.path.join(ROOT, "results", "_crossseed.log")
SEEDS = [1, 2]


def log(m):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + "  " + m
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def done(exp, seed):
    return os.path.exists(os.path.join(ROOT, "checkpoints", exp, f"seed_{seed}",
                                        "step_3000", "model.pt"))


def run(exp, seed, args):
    if done(exp, seed):
        log(f"SKIP {exp} seed={seed} (finaler Checkpoint existiert)")
        return
    log(f"=== {exp} seed={seed} START ===")
    r = subprocess.run([PY] + args, cwd=ROOT)
    log(f"=== {exp} seed={seed} exit={r.returncode} ===")


def dense(depth, seed):
    run(f"dense_d{depth}", seed,
        ["-m", "experiments.train_dense", "--dense_depth", str(depth),
         "--seed", str(seed), "--device", "cuda"])


def sparse(n, R, mode, seed):
    tag = {None: "naked", "per_token": "srcore"}[mode]
    exp = f"{tag}_b{n}_R{R}"
    args = ["-m", "experiments.tinystories_exp", "--n_blocks", str(n), "--k", "4",
            "--R", str(R), "--steps", "3000", "--seed", str(seed), "--exp_name", exp,
            "--device", "cuda"]
    if mode:
        args += ["--core_mode", mode]
    run(exp, seed, args)


def main():
    open(LOG, "w", encoding="utf-8").close()
    log(f"Cross-Seed-Sweep gestartet (PID {os.getpid()}) — Seeds {SEEDS}")
    # Nur die UMSTRITTENEN Sparse-Configs cross-seeden. Dense-Front (seed 0) ist sauber
    # und unstrittig -> kein Dense cross-seed noetig.
    for seed in SEEDS:
        for R in (2, 6):
            sparse(32, R, None, seed)        # Naked b32
        for R in (2, 6):
            sparse(32, R, "per_token", seed)  # SR-Core b32
    log("CROSS-SEED-SWEEP FERTIG.")


if __name__ == "__main__":
    main()
