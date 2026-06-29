"""Selbsttest fuer run_queue.py — python scripts/_test_run_queue.py"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_queue import run_queue

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

ok = failed = 0

def check(label, cond):
    global ok, failed
    if cond:
        print(f"  OK  {label}")
        ok += 1
    else:
        print(f"  FAIL {label}")
        failed += 1

# ── Test 1: Dry-run — kein tatsächlicher Start ──────────────────────────────
print("=== Test 1: Dry-run ===")
jobs = [
    {"name": "job_a", "args": ["-c", "print(1)"], "log": "_rq_t1_a"},
    {"name": "job_b", "args": ["-c", "print(2)"], "log": "_rq_t1_b"},
]
r = run_queue(jobs, dry_run=True)
check("ok=0 bei dry-run",  r["ok"] == 0)
check("failed=0",          r["failed"] == 0)
check("2 dry-run entries", sum(x["status"]=="dry-run" for x in r["results"]) == 2)
check("kein Log erstellt", not (os.path.exists("results/_rq_t1_a.log") and
                                 os.path.getsize("results/_rq_t1_a.log") > 0))

# ── Test 2: Echter Lauf ──────────────────────────────────────────────────────
print("=== Test 2: Echter Lauf ===")
jobs2 = [
    {"name": "job_a", "args": ["-c", "print('A done')"], "log": "_rq_t2_a"},
    {"name": "job_b", "args": ["-c", "print('B done')"], "log": "_rq_t2_b"},
]
r2 = run_queue(jobs2)
check("ok=2",          r2["ok"] == 2)
check("failed=0",      r2["failed"] == 0)
log_a = open("results/_rq_t2_a.log", encoding="utf-8").read().strip()
log_b = open("results/_rq_t2_b.log", encoding="utf-8").read().strip()
check("Log A hat Output", "A done" in log_a)
check("Log B hat Output", "B done" in log_b)

# ── Test 3: Kollisions-Guard ─────────────────────────────────────────────────
print("=== Test 3: Kollisions-Guard ===")
try:
    run_queue(jobs2)  # same logs already have content
    check("Guard abgefangen", False)
except FileExistsError:
    check("Guard abgefangen", True)

# ── Test 4: overwrite=True erlaubt Kollision ─────────────────────────────────
print("=== Test 4: overwrite=True ===")
jobs4 = [
    {"name": "job_a", "args": ["-c", "print('A v2')"], "log": "_rq_t2_a", "overwrite": True},
]
r4 = run_queue(jobs4)
check("overwrite ok=1",    r4["ok"] == 1)
check("Log ueberschrieben", "A v2" in open("results/_rq_t2_a.log", encoding="utf-8").read())

# ── Test 5: skip_if ──────────────────────────────────────────────────────────
print("=== Test 5: skip_if ===")
jobs5 = [
    {"name": "skip_job", "args": ["-c", "import sys; sys.exit('SHOULD NOT RUN')"],
     "log": "_rq_t5_skip",
     "skip_if": "results/_rq_t2_b.log"},   # existiert -> skip
    {"name": "run_job",  "args": ["-c", "print('ran')"], "log": "_rq_t5_run"},
]
r5 = run_queue(jobs5)
check("skipped=1",  r5["skipped"] == 1)
check("ok=1",       r5["ok"] == 1)
check("skip_job Log nicht erstellt",
      not (os.path.exists("results/_rq_t5_skip.log") and
           os.path.getsize("results/_rq_t5_skip.log") > 0))
check("run_job ran", "ran" in open("results/_rq_t5_run.log", encoding="utf-8").read())

# ── Test 6: Validierungsfehler (name fehlt) ──────────────────────────────────
print("=== Test 6: Validierungsfehler ===")
try:
    run_queue([{"args": ["-c", "pass"], "log": "_rq_t6"}])
    check("ValueError geworfen", False)
except ValueError:
    check("ValueError geworfen", True)

# ── Test 7: exit!=0, Abbruch ─────────────────────────────────────────────────
print("=== Test 7: Job-Fehler bricht ab ===")
jobs7 = [
    {"name": "fail_job", "args": ["-c", "import sys; sys.exit(1)"], "log": "_rq_t7_fail"},
    {"name": "after",    "args": ["-c", "print('after')"],           "log": "_rq_t7_after"},
]
r7 = run_queue(jobs7, continue_on_error=False)
check("failed=1",  r7["failed"] == 1)
check("ok=0",      r7["ok"] == 0)
check("'after' nicht gestartet",
      not (os.path.exists("results/_rq_t7_after.log") and
           os.path.getsize("results/_rq_t7_after.log") > 0))

# ── Test 8: continue_on_error ────────────────────────────────────────────────
print("=== Test 8: continue_on_error ===")
jobs8 = [
    {"name": "fail_job", "args": ["-c", "import sys; sys.exit(1)"],
     "log": "_rq_t8_fail", "overwrite": True},
    {"name": "after",    "args": ["-c", "print('after')"],
     "log": "_rq_t8_after"},
]
r8 = run_queue(jobs8, continue_on_error=True)
check("failed=1", r8["failed"] == 1)
check("ok=1",     r8["ok"] == 1)
check("'after' lief durch",
      "after" in open("results/_rq_t8_after.log", encoding="utf-8").read())

# ── Test 9: CLI --dry-run ────────────────────────────────────────────────────
print("=== Test 9: CLI --dry-run ===")
jf = "results/_rq_test_jobs.json"
json.dump([{"name":"cli_test","args":["-c","print(99)"],"log":"_rq_t9"}], open(jf,"w"))
p = subprocess.run([sys.executable, "scripts/run_queue.py", jf, "--dry-run"],
                   capture_output=True, text=True, cwd=ROOT)
os.remove(jf)
check("CLI exit=0",        p.returncode == 0)
check("CLI zeigt dry-run", "dry-run" in p.stdout)

# ── Ergebnis ─────────────────────────────────────────────────────────────────
print()
print(f"GESAMT: {ok} OK, {failed} FAIL")
if failed:
    sys.exit(1)
