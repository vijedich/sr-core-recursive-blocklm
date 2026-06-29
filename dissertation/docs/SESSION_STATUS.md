# Session-Statusbericht — 2026-06-17

## UPDATE (Folge-Session): Crash-Fixes angewendet + Seed 2 gerettet

**Drei Crash-Ursachen behoben** (Details in `memory/technical_constraints.md`):
1. `rblm/router.py`: Logits werden via `nan_to_num`+`clamp(-30,30)` abgesichert
   (verhindert NaN→topk/Dispatch-Crash). **Beweis:** Seed 2 lief diesmal sauber
   durch die alte Crash-Zone step 1500 (vorher 2× still gecrasht).
2. `experiments/tinystories_exp.py train()`: `check_finite` STANDARD AN +
   überspringt schlechte Steps statt Abbruch (`--strict_finite`/`--no_check_finite`).
3. `run()`: Modell + versionierter Checkpoint werden SOFORT nach Training gesichert,
   Auswertung läuft best-effort in try/except → kein Modellverlust mehr durch Eval-Hang.
4. **NEU: Resume-Pfad** (`--resume_from <ckpt.pt>`): lädt Modell+Optimizer+Scheduler+RNG
   und setzt Training fort. Zwischen-Checkpoints werden nach Erfolg automatisch gelöscht.

**Seed 2 — gelöst:** Lief bis step 2700 (alte Zone 1500 überstanden!), dann ein
umgebungsbedingter Spät-Kill (kein TDR/WHEA/App-Crash im Event-Log → vermutl. CUDA-
Illegal-Access oder VRAM-OOM nach ~40 Min Dauerlast, NICHT der alte NaN-Bug). Per
`--resume_from step_2700.pt` fortgesetzt und abgeschlossen:
**val_loss=3.158, step_3000-Checkpoint gespeichert.** (Seed 0: 3.127, Seed 1: 3.069)

**Noch offen:** Seed 3 (Checkpoint war verloren → Neulauf nötig, jetzt safe);
Seed-0-Analyse-Regen; Seed-3-Kompetenzanalyse; No-Div-Warmstart.

---

## Prozess läuft noch (Stand: Ende Session) — VERALTET, siehe UPDATE oben

Seed-3-Training (`b1svhz4z0`) ist **nicht gecrasht**. Der Prozess lebt, zeigt kurze
GPU-Spitzen wenn der Monitor aufwacht (sichtbar im Task-Manager). Das Muster passt zu
`full_eval()` oder `iteration_diagnostics()` nach dem Training: CPU-lastig mit kurzen
GPU-Bursts, kein kontinuierlicher 30%-Betrieb wie beim Training. Letzter Output war:

```
step  2000  L1=3.172  Lfin=3.173  Jacc=0.993  Div=0.078  1182s
```

Das Training selbst lief durch (step 2000 nach 1182s = ~20 min). Die Auswertungsphase
danach kann länger dauern als erwartet. **Nicht abbrechen** — wenn es fertig ist kommt
die Checkpoint-Benachrichtigung.

---

## Was in dieser Session erledigt wurde

| Task | Status | Ergebnis |
|------|--------|----------|
| Curriculum C Seed 0 Training | ✓ | val_loss=3.127, SHA256=482fa9f8... |
| Eval-Suite Seed 0 (Tests 3+4+5+8) | ✓ | Depth flach, State-Reset +0.068 Nats, Cache 3.7–4.6× |
| Seed 1 Training | ✓ | val_loss=3.069 |
| Seed 1 Kompetenzanalyse | ✓ | Jaccard=0.336, CLF=0.401 [0.381,0.429], group_r1=+5.5%, frac_r1=0.009 |
| MILESTONE_CURRICULUM_C.md erstellt | ✓ | Eingefroren, vollständig |
| TESTPLAN.md erstellt | ✓ | 10 Tests dokumentiert |
| eval_suite.py erstellt | ✓ | Tests 3/4/5/8 auf beliebigem Checkpoint |
| Seed 3 Training | laufend | Letzter Output step 2000, dann Auswertungsphase |

---

## Fehler in dieser Session

### 1. Tag-Bug in competence_centers_exp.py (behoben)

**Problem:** `tag = f"competence_b64k4R6_s{seed}"` nutzte immer den Default-Parameter
`seed=0`, nicht den Seed aus dem geladenen Checkpoint. Alle Analysen speicherten als
`_s0_...` unabhängig vom tatsächlichen Seed.

**Folge:** Seed-1-Analyse überschrieb Seed-0-Ergebnis-JSONs und -PNGs.

**Fix:** `competence_centers_exp.py` liest Seed jetzt aus dem Checkpoint-Pfad:
```python
_seed_dir = os.path.basename(os.path.dirname(ckpt))
if _seed_dir.startswith("seed_"):
    seed = int(_seed_dir.split("_")[1])
```

**Nacharbeit nötig:** Seed-0-Analysedateien wurden überschrieben und müssen
neu generiert werden (Task #9):
```bash
python -m experiments.competence_centers_exp \
  --ckpt checkpoints/tinystories_curriculum_fromIter2/seed_0/step_3000 \
  --analysis --device cuda
```

Seed-1-Dateien wurden korrekt auf `_s1_` umbenannt (8 Dateien).

---

### 2. Seed-2-Crash — deterministisch, Ursache offen

**Beobachtung:** Seed 2 crashte zweimal identisch zwischen step 1500–2000.
Kein Python-Traceback, kein sichtbarer GPU-Prozess danach → stiller CUDA/System-Kill.

| Versuch | bs | Letzter Step | Timing |
|---------|----|-------------|--------|
| 1 | 32 | 1500 (bei 879s) | Crash irgendwo in step 1501–1999 |
| 2 | 16 | 1500 (bei 792s) | Crash irgendwo in step 1501–1999 |

**Hypothese:** Deterministischer Crash durch batch-spezifische NaN-Propagation mit
seed=2, die dann zu einem CUDA-SEGFAULT in `argsort()` führt (NaN-Cast zu int64
in topk_idx → ungültige Block-IDs).

**Kein einfaches "Seed ist schlecht"** — das ist ein reproduzierbarer Stabilitätsfall
der einzugrenzen ist.

**Warum Seeds 0+1 liefen:** Andere Batch-Sequenzen, kein NaN-auslösender Batch
im kritischen Bereich.

---

### 3. Seed-3-Hang nach step 2000 — kein Crash

**Update:** Prozess läuft noch (GPU-Spikes beim Monitor-Aufwachen). Kein Crash,
nur sehr langsame Auswertungsphase nach dem Training. Abwarten.

---

## Code-Änderungen in dieser Session

### rblm/models.py
Diversity-Sicherheitsfallback in `ModelC.core()`:
```python
# Neu nach iter_ablate[top_blocks] = True:
n_valid = int((~iter_ablate).sum())
if n_valid < self.cfg.k_active:
    release_n = self.cfg.k_active - n_valid
    least_used = counts[top_blocks].argsort()[:release_n]
    iter_ablate[top_blocks[least_used]] = False
```
Garantiert ≥ k_active gültige Kandidaten nach Diversity-Maskierung.

### experiments/tinystories_exp.py
Neue Parameter in `train()` und `run()`:
- `--check_finite`: NaN/Inf-Prüfung bei Loss und Gradienten pro Schritt.
  Produziert Python-Exception statt stiller CUDA-SEGFAULT mit exakter Step-Nummer.
- `--intermediate_ckpt_every N --intermediate_ckpt_from M`: Speichert alle N
  Schritte ab Schritt M ein Snapshot (Modell + Optimizer + Scheduler + RNG-State)
  nach `checkpoints/.intermediate/seed_N/step_X.pt`
- `|grad|=N.NNN` jetzt immer in jedem eval-Print sichtbar

### experiments/competence_centers_exp.py
- Tag-Bug gefixt (Seed aus Checkpoint-Pfad)

### scripts/eval_suite.py (neu)
Standalone-Script für Tests 3/4/5/8 auf beliebigem Checkpoint ohne Training.

---

## Ausstehende Tasks

### Priorität 1 — Sofort nach Seed-3-Abschluss
```bash
# Task 6: Seed-3-Kompetenzanalyse
# ZUERST prüfen ob Zieldateien existieren (Kopien-Regel!):
Get-ChildItem results/competence_b64k4R6_s3_analysis_* 
# Dann:
python -m experiments.competence_centers_exp \
  --ckpt checkpoints/tinystories_curriculum_fromIter2/seed_3/step_3000 \
  --analysis --device cuda
```

### Priorität 2 — Seed-2-Stabilitätstest (Task 10)
```bash
python -m experiments.tinystories_exp \
  --diverse --diverse_from_iter 2 \
  --pretrained_ckpt checkpoints/tinystories_phase2/seed_0/step_3000 \
  --steps 3000 --seed 2 --device cuda \
  --check_finite \
  --intermediate_ckpt_every 25 --intermediate_ckpt_from 1400
```
Ziel: Python-Exception mit exaktem Step + betroffenen Parametern statt stiller Crash.
Danach: Befund in TESTPLAN.md und MILESTONE dokumentieren.

### Priorität 3 — Seed-0-Analyse regenerieren (Task 9)
```bash
# Zuerst prüfen:
Get-ChildItem results/competence_b64k4R6_s0_analysis_*
# Dann:
python -m experiments.competence_centers_exp \
  --ckpt checkpoints/tinystories_curriculum_fromIter2/seed_0/step_3000 \
  --analysis --device cuda
```

### Priorität 4 — No-Diversity Warmstart (Tasks 7+8)
```bash
# Training:
python -m experiments.tinystories_exp \
  --pretrained_ckpt checkpoints/tinystories_phase2/seed_0/step_3000 \
  --steps 3000 --seed 0 --exp_name tinystories_nodiv_warmstart --device cuda

# Analyse (danach):
python -m experiments.competence_centers_exp \
  --ckpt checkpoints/tinystories_nodiv_warmstart/seed_0/step_3000 \
  --analysis --device cuda
```

---

## Wichtige Regel — IMMER vor Analyse prüfen

```powershell
# Vor JEDEM Analyse-Lauf zuerst:
Get-ChildItem "C:\_temp\dev\3D KI Modell\recursive-blocklm\results\competence_b64k4R6_s*_analysis_*"
# Wenn Dateien existieren: umbenennen (z.B. _bak Suffix) BEVOR der Lauf startet
```

---

## Checkpoint-Übersicht (Stand Session-Ende)

| Experiment | Seed | Step | val_loss | SHA256 |
|-----------|------|------|---------|--------|
| tinystories_phase2 | 0 | 3000 | 3.658 | 65b5c3ee... |
| tinystories_curriculum_fromIter2 | 0 | 3000 | 3.127 | 482fa9f8... |
| tinystories_curriculum_fromIter2 | 1 | 3000 | 3.069 | 9d6de2e3... |
| tinystories_curriculum_fromIter2 | 3 | 3000 | — | (laufend) |

---

## Offene Stabilitätsfrage — Auswertungsphase

`full_eval()` und `iteration_diagnostics()` nach Training laufen länger als erwartet
(CPU-lastig, kurze GPU-Bursts). Prüfen ob `iteration_diagnostics` in `run()` mit
zu vielen Batches aufgerufen wird. Eventuell n_batches reduzieren oder in separates
`eval_suite.py` auslagern (bereits vorhanden).

---

*Generiert: 2026-06-17, Session c0e25176*
