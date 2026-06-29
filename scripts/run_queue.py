"""Universeller sequenzieller Job-Runner.

Führt GPU-Jobs strikt nacheinander aus. Sicher gegen:
  - Log-Überschreiben  (Fehler wenn Log existiert und Inhalt hat)
  - Checkpoint-Kollisionen  (skip_if: überspringen wenn Pfad existiert)
  - Gleichzeitige GPU-Nutzung  (immer sequenziell, blockierend)

──────────────────────────────────────────────────────────────────────────
Job-Spec (dict):
  name      str        Anzeigename / Identifier (PFLICHT)
  args      list[str]  argv ohne 'python', z.B. ["-m","experiments.foo","--n_blocks","64"]
  log       str        Log-Prefix in results/ (PFLICHT) → results/{log}.log + .err
  skip_if   str        Überspringen wenn dieser Pfad existiert  (optional)
  overwrite bool       Log überschreiben falls vorhanden — Default: False  (optional)
  cwd       str        Arbeitsverzeichnis — Default: Projekt-Root  (optional)
──────────────────────────────────────────────────────────────────────────

Nutzung als Bibliothek:
    from scripts.run_queue import run_queue
    run_queue([
        {"name": "ctrl",
         "args": ["-m", "experiments.heteromini_long", "--n_blocks", "64",
                  "--max_steps", "17000", "--exp_tag", "_ctrl"],
         "log": "_b64_ctrl",
         "skip_if": "results/hm_cont_hm_srcore_b64_k8_R6_ctrl_s0.pt"},
        {"name": "softfull",
         "args": ["-m", "experiments.heteromini_long", "--n_blocks", "64",
                  "--max_steps", "17000", "--lambda_core", "0.01",
                  "--exp_tag", "_softfull"],
         "log": "_b64_softfull"},
    ])

Nutzung als CLI:
    python scripts/run_queue.py jobs.json
    python scripts/run_queue.py jobs.json --dry-run
    python scripts/run_queue.py jobs.json --force          # erlaubt Log-Überschreiben
    python scripts/run_queue.py jobs.json --continue-on-error
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from typing import Optional

# Windows-Terminal oft cp1252 -> alles auf UTF-8 stellen
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
PY      = sys.executable


# ─── Validation ──────────────────────────────────────────────────────────────

def _validate_job(job: dict, idx: int) -> list[str]:
    errors = []
    if not isinstance(job.get("name"), str) or not job["name"].strip():
        errors.append(f"Job[{idx}]: 'name' fehlt oder leer")
    if not isinstance(job.get("args"), list) or not job["args"]:
        errors.append(f"Job[{idx}] '{job.get('name','?')}': 'args' fehlt oder leer")
    if not isinstance(job.get("log"), str) or not job["log"].strip():
        errors.append(f"Job[{idx}] '{job.get('name','?')}': 'log' fehlt oder leer")
    return errors


def _log_path(log_prefix: str, ext: str) -> str:
    prefix = log_prefix if log_prefix.startswith(os.sep) else os.path.join(RESULTS, log_prefix)
    return prefix + ext


def _check_log_collision(job: dict, force: bool) -> Optional[str]:
    """Gibt Fehlermeldung zurück falls Log existiert und nicht überschrieben werden darf."""
    lp = _log_path(job["log"], ".log")
    if os.path.exists(lp) and os.path.getsize(lp) > 0:
        if force or job.get("overwrite", False):
            return None  # explizit erlaubt
        return (f"Log existiert bereits und hat Inhalt: {lp}\n"
                f"  → job 'overwrite: true' setzen oder --force verwenden")
    return None


def _would_skip(job: dict) -> bool:
    skip_if = job.get("skip_if")
    if not skip_if:
        return False
    path = skip_if if os.path.isabs(skip_if) else os.path.join(ROOT, skip_if)
    return os.path.exists(path)


# ─── Queue-Log ───────────────────────────────────────────────────────────────

class QueueLog:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# run_queue gestartet {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    def write(self, msg: str):
        line = f"{time.strftime('%H:%M:%S')}  {msg}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ─── Core ────────────────────────────────────────────────────────────────────

def run_queue(
    jobs: list[dict],
    *,
    dry_run: bool = False,
    force: bool = False,
    continue_on_error: bool = False,
    queue_log_name: str = "_queue",
) -> dict:
    """Führt Jobs sequenziell aus.

    Args:
        jobs              : Liste von Job-Dicts (siehe Modul-Docstring)
        dry_run           : Nur Vorschau, kein tatsächlicher Start
        force             : Log-Überschreiben erlauben
        continue_on_error : Bei Job-Fehler (exit != 0) weitermachen statt abbrechen
        queue_log_name    : Prefix für den Queue-eigenen Log (in results/)

    Returns:
        dict mit "ok", "skipped", "failed", "results" (pro Job)
    """
    # ── Validierung aller Jobs VOR dem ersten Start ──────────────────────────
    all_errors = []
    for i, job in enumerate(jobs):
        all_errors.extend(_validate_job(job, i))
    if all_errors:
        for e in all_errors:
            print(f"FEHLER: {e}", file=sys.stderr)
        raise ValueError(f"{len(all_errors)} Validierungsfehler — kein Job gestartet")

    # Log-Kollisions-Check (alle Jobs, bevor irgendeiner startet)
    collision_errors = []
    for job in jobs:
        if not _would_skip(job):
            err = _check_log_collision(job, force)
            if err:
                collision_errors.append(f"Job '{job['name']}': {err}")
    if collision_errors:
        for e in collision_errors:
            print(f"FEHLER: {e}", file=sys.stderr)
        raise FileExistsError(
            f"{len(collision_errors)} Log-Kollision(en) — kein Job gestartet\n"
            "Tipp: anderen 'log'-Namen wählen, 'overwrite: true' setzen, oder --force")

    ts   = time.strftime("%Y%m%d_%H%M%S")
    qlog = QueueLog(_log_path(f"{queue_log_name}_{ts}", ".log"))
    qlog.write(f"run_queue: {len(jobs)} Job(s), dry_run={dry_run}, force={force}")

    summary = {"ok": 0, "skipped": 0, "failed": 0, "results": []}

    for i, job in enumerate(jobs):
        name   = job["name"]
        args   = job["args"]
        log_pf = job["log"]
        cwd    = job.get("cwd") or ROOT
        stdout = _log_path(log_pf, ".log")
        stderr = _log_path(log_pf, ".err")

        # ── skip_if ──────────────────────────────────────────────────────────
        if _would_skip(job):
            qlog.write(f"[{i+1}/{len(jobs)}] SKIP  {name}  (skip_if existiert: {job['skip_if']})")
            summary["skipped"] += 1
            summary["results"].append({"name": name, "status": "skipped"})
            continue

        cmd_str = f"python {' '.join(args)}"
        qlog.write(f"[{i+1}/{len(jobs)}] START {name}")
        qlog.write(f"  cmd:    {cmd_str}")
        qlog.write(f"  stdout: {stdout}")
        qlog.write(f"  stderr: {stderr}")
        qlog.write(f"  cwd:    {cwd}")

        if dry_run:
            qlog.write(f"  [dry-run] nicht gestartet")
            summary["results"].append({"name": name, "status": "dry-run", "cmd": cmd_str})
            continue

        # ── Tatsächlicher Lauf ────────────────────────────────────────────────
        t0 = time.time()
        os.makedirs(os.path.dirname(stdout), exist_ok=True)
        with open(stdout, "w", encoding="utf-8") as fout, \
             open(stderr, "w", encoding="utf-8") as ferr:
            proc = subprocess.Popen(
                [PY] + args,
                cwd=cwd,
                stdout=fout,
                stderr=ferr,
            )
            qlog.write(f"  pid:    {proc.pid}")
            rc = proc.wait()
        elapsed = time.time() - t0
        status = "ok" if rc == 0 else "failed"
        qlog.write(f"[{i+1}/{len(jobs)}] {status.upper()}  {name}  exit={rc}  ({elapsed:.0f}s)")

        summary[status] += 1
        summary["results"].append({"name": name, "status": status,
                                   "exit": rc, "elapsed_s": round(elapsed)})

        if rc != 0 and not continue_on_error:
            qlog.write(f"ABBRUCH nach Fehler in '{name}' (--continue-on-error zum Weitermachen)")
            break

    qlog.write(f"FERTIG — ok={summary['ok']} skipped={summary['skipped']} "
               f"failed={summary['failed']}")
    return summary


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Sequenzieller Job-Runner. jobs.json = Liste von Job-Dicts.")
    ap.add_argument("jobs_file",
                    help="JSON-Datei mit Job-Liste, oder '-' für stdin")
    ap.add_argument("--dry-run",  action="store_true",
                    help="Vorschau ohne tatsächlichen Start")
    ap.add_argument("--force",    action="store_true",
                    help="Log-Überschreiben erlauben")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="Bei Job-Fehler weitermachen statt abbrechen")
    ap.add_argument("--queue-log", default="_queue",
                    help="Prefix fuer Queue-eigenen Log in results/ (default: _queue)")
    a = ap.parse_args()

    if a.jobs_file == "-":
        jobs = json.load(sys.stdin)
    else:
        with open(a.jobs_file, encoding="utf-8") as f:
            jobs = json.load(f)

    if not isinstance(jobs, list):
        sys.exit("jobs.json muss eine JSON-Liste sein")

    try:
        summary = run_queue(
            jobs,
            dry_run=a.dry_run,
            force=a.force,
            continue_on_error=a.continue_on_error,
            queue_log_name=a.queue_log,
        )
    except (ValueError, FileExistsError) as e:
        sys.exit(str(e))

    if summary["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
