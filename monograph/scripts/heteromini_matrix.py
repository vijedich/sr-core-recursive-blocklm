"""HeteroMini-Smoke-Matrix (losgeloest, GPU-seriell, idempotent).

Reduzierte Matrix auf HeteroMini-v1, gleiches Budget (steps), k=4, End-Gewichtung,
R direkt trainiert, seed 0 (Smoke):
  Dense   : d8, d24
  Naked   : b32 R2/R6
  SR-Core : b32 R2/R6, b64 R2/R6  (per_token)

Idempotent (Skip wenn Ergebnis-JSON existiert). Start losgeloest:
  Start-Process python -ArgumentList '-u','scripts/heteromini_matrix.py'
    -WorkingDirectory <root> -RedirectStandardOutput results/_hm_matrix.out
    -RedirectStandardError results/_hm_matrix.err -WindowStyle Hidden
"""
import os, sys, time, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PY = sys.executable
LOG = os.path.join(ROOT, "results", "_hm_matrix.log")
STEPS = 2000
SEED = 0


def log(m):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + "  " + m
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def done(exp):
    return os.path.exists(os.path.join(ROOT, "results", f"heteromini_{exp}_s{SEED}.json"))


def run(exp, args):
    if done(exp):
        log(f"SKIP {exp} (Ergebnis existiert)")
        return
    log(f"=== {exp} START ===")
    r = subprocess.run([PY, "-m", "experiments.heteromini_train"] + args
                       + ["--steps", str(STEPS), "--seed", str(SEED), "--device", "cuda"],
                       cwd=ROOT)
    log(f"=== {exp} exit={r.returncode} ===")


def main():
    open(LOG, "w", encoding="utf-8").close()
    log(f"HeteroMini-Matrix gestartet (PID {os.getpid()}), steps={STEPS}")
    run("hm_dense_d8",      ["--variant", "dense", "--depth", "8"])
    run("hm_dense_d24",     ["--variant", "dense", "--depth", "24"])
    run("hm_naked_b32_R2",  ["--variant", "sparse", "--n_blocks", "32", "--R", "2"])
    run("hm_naked_b32_R6",  ["--variant", "sparse", "--n_blocks", "32", "--R", "6"])
    run("hm_srcore_b32_R2", ["--variant", "sparse", "--n_blocks", "32", "--R", "2", "--core_mode", "per_token"])
    run("hm_srcore_b32_R6", ["--variant", "sparse", "--n_blocks", "32", "--R", "6", "--core_mode", "per_token"])
    run("hm_srcore_b64_R2", ["--variant", "sparse", "--n_blocks", "64", "--R", "2", "--core_mode", "per_token"])
    run("hm_srcore_b64_R6", ["--variant", "sparse", "--n_blocks", "64", "--R", "6", "--core_mode", "per_token"])
    log("HETEROMINI-MATRIX FERTIG.")


if __name__ == "__main__":
    main()
