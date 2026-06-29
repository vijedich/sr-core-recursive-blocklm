# Testplan: Curriculum C — Validierung und Erweiterung

**Erstellt:** 2026-06-17  
**Ausgangsstand:** MILESTONE_CURRICULUM_C.md (Seed 0, vollständig dokumentiert)  
**Prioritätsreihenfolge:** Test 1 → Test 2 → Test 3+4 parallel → Tests 5–10 nach Bedarf

---

## Implementierungsstand-Übersicht

| Test | Name | GPU-Training | Status | Blockiert durch |
|------|------|-------------|--------|-----------------|
| 1 | Seeds 1+2 | ~60 Min (2×30) | **Sofort lauffähig** | — |
| 2 | No-Diversity-Kontrolle | ~30 Min | **Sofort lauffähig** | — |
| 3 | Depth-Truncation | 0 Min (Eval) | **Sofort lauffähig** | — |
| 4 | State-Reset-Test | 0 Min (Eval) | **Sofort lauffähig** | — |
| 5 | Inference ohne Diversity | 0 Min (Eval) | **Sofort lauffähig** | — |
| 6 | Usage-gematchte Ablation | 0 Min (Eval) | Neuer Code nötig | — |
| 7 | Iterationsweise Ablation | 0 Min (Eval) | Teils vorhanden | — |
| 8 | Cache/Working-Set | 0 Min (Eval) | **Sofort lauffähig** | — |
| 9 | Fest-verdrahtete Baseline | ~30 Min | Neues Modell nötig | Tests 1–3 |
| 10 | Skalierung (256 Blöcke) | ~60–90 Min | Sofort lauffähig | Tests 1–4 |

Tests 3, 4, 5, 7, 8 laufen auf dem bestehenden Curriculum-C-Checkpoint — kein Training.

---

## Test 1 — Reproduzierbarkeit: Seeds 1 und 2

### Ziel

Alle Befunde aus MILESTONE_CURRICULUM_C.md sind Seed-0-Ergebnisse. Ohne Seed-Validierung
bleibt unklar, ob Gini=0.13, Jaccard off-diag=0.282 und CLF=0.421 reproduzierbar sind.

### Kommandos

```bash
# Seed 1 — Training (~30 Min)
python -m experiments.tinystories_exp \
  --diverse --diverse_from_iter 2 \
  --pretrained_ckpt checkpoints/tinystories_phase2/seed_0/step_3000 \
  --steps 3000 --seed 1 --device cuda

# Seed 1 — Analyse
python -m experiments.competence_centers_exp \
  --ckpt checkpoints/tinystories_curriculum_fromIter2/seed_1/step_3000 \
  --analysis --device cuda

# Seed 2 — Training (~30 Min)
python -m experiments.tinystories_exp \
  --diverse --diverse_from_iter 2 \
  --pretrained_ckpt checkpoints/tinystories_phase2/seed_0/step_3000 \
  --steps 3000 --seed 2 --device cuda

# Seed 2 — Analyse
python -m experiments.competence_centers_exp \
  --ckpt checkpoints/tinystories_curriculum_fromIter2/seed_2/step_3000 \
  --analysis --device cuda
```

### Pass-Kriterien

| Metrik | Seed 0 (Referenz) | Toleranz |
|--------|-------------------|---------|
| val_loss | 3.127 | ±0.05 |
| Hub-Gini | 0.13 | ±0.05 |
| Jaccard off-diag | 0.282 | ±0.10 |
| CLF bootstrap-mean | 0.421 | > 0.35 (klar über Zufall 0.167) |
| CLF 95%-KI | [0.391, 0.451] | unteres Ende > 0.30 |
| Tote Blöcke | 0 | = 0 |

### Interpretation

- **Alle 3 Seeds innerhalb Toleranz:** Befunde aus Seed 0 sind reproduzierbar → Architekturbeleg belastbar.
- **val_loss stabil, aber Gini/Jaccard nicht:** Diversity-Effekt ist Seed-abhängig → Muster real, aber Stärke variiert.
- **Große Streuung in CLF:** Kategorieinformation im Routing ist nicht stabil → Befund G4 vorläufig.

---

## Test 2 — No-Diversity-Kontrolle (Fairness-Test)

### Ziel

Klärt, ob val_loss=3.127 aus der Diversity-Regularisierung kommt oder schlicht aus
3000 zusätzlichen Trainingsschritten auf dem Phase-2-Checkpoint.

Ohne diese Kontrolle ist nicht entscheidbar, ob Curriculum C *besser wegen Diversity*
oder *besser wegen mehr Training* ist.

### Kommando

```bash
python -m experiments.tinystories_exp \
  --pretrained_ckpt checkpoints/tinystories_phase2/seed_0/step_3000 \
  --steps 3000 --seed 0 --device cuda
# (kein --diverse, kein --diverse_from_iter)
# Checkpoint-Name: tinystories_nodiv_warmstart
```

Anschließend identische Analyse-Pipeline:

```bash
python -m experiments.competence_centers_exp \
  --ckpt checkpoints/tinystories_nodiv_warmstart/seed_0/step_3000 \
  --analysis --device cuda
```

### Vergleichstabelle (auszufüllen nach Run)

| Metrik | Phase 2 (Start) | No-Diversity WS | Curriculum C | Curriculum A |
|--------|-----------------|-----------------|--------------|--------------|
| val_loss | 3.658 | ? | 3.127 | 3.127 |
| Hub-Gini | 0.62 | ? | **0.13** | 0.52 |
| Jaccard off-diag | 0.759 | ? | **0.282** | 0.899 |
| CLF bootstrap | 0.380 | ? | **0.421** | 0.374 |

### Interpretation

- **No-Div val_loss ≈ 3.127:** Loss-Vorteil kommt aus mehr Training, nicht aus Diversity.
  Diversity-spezifischer Beitrag muss dann über Gini/Jaccard/CLF nachgewiesen werden.
- **No-Div val_loss > 3.15:** Diversity hilft der Qualität. Überraschender Befund.
- **No-Div Gini ≈ 0.5, Jaccard ≈ 0.9:** Diversity ist allein verantwortlich für Blockverteilung.
  Das wäre der erwartete Fall.

---

## Test 3 — Depth-Truncation (Anytime-Kurve)

### Ziel

Wie viel bringt jede zusätzliche Iteration? Ab wann sättigt die Qualität?
Ist der Kern r=2–r=6 wirklich iterativ nützlich — oder redundant?

### Implementierungsstand

**Bereits vorhanden:** `loss_per_iter` wird bei jedem Eval-Schritt und in der `metrics.json`
gespeichert. Für Curriculum C (Seed 0): `[3.126, 3.126, 3.126, 3.126, 3.126, 3.127]`
— fast flach.

Für eine gründlichere Messung mit mehr Batches und allen Checkpoints:

```bash
python scripts/eval_suite.py \
  --ckpt checkpoints/tinystories_curriculum_fromIter2/seed_0/step_3000 \
  --tests depth_truncation \
  --device cuda
```

### Gesuchte Kurve

```
L[r=1], L[r=2], ..., L[r=6]
```

**Pass-Kriterium:** Sichtbarer Abfall L[1] > L[r_min] mit r_min > 1 (Tiefennutzen vorhanden).

**Aktuelle Antwort (Seed 0):** Δ(L1−L6) = 0.001 Nats — minimaler Tiefennutzen.
Direkter Vergleich mit No-Diversity-Kontrolle (Test 2) zeigt, ob Diversity die Anytime-Kurve beeinflusst.

---

## Test 4 — State-Reset-Test

### Ziel

Wenn der Hidden State zwischen Iterationen zurückgesetzt wird (h → h0),
verschwindet dann der Tiefengewinn? Das wäre der stärkste Nachweis, dass
spätere Iterationen auf Zwischenergebnissen aufbauen.

### Implementierungsstand

**Bereits implementiert in `rblm/models.py`:**

```python
# ModelC.core(), Zeile 143:
base = h0 if state_reset else h
```

Der Parameter `state_reset=True` wird über `model(toks, state_reset=True)` übergeben
(`**kw` in `forward()` → `core()`). **Kein neuer Code nötig.**

```bash
python scripts/eval_suite.py \
  --ckpt checkpoints/tinystories_curriculum_fromIter2/seed_0/step_3000 \
  --tests state_reset \
  --device cuda
```

### Erwartete Ausgabe

```
Normal:       L[r=1..6] = [3.126, 3.126, 3.126, 3.126, 3.126, 3.127]
State-Reset:  L[r=1..6] = [?, ?, ?, ?, ?, ?]
Delta:        [?, ?, ?, ?, ?, ?]
```

**Interpretation:**
- Δ klein → Iterationen bauen nicht auf Zwischenstand auf (Tiefe ist redundant).
- Δ groß → echter akkumulierter Zustand (stärkerer Architekturbeleg).

---

## Test 5 — Inference mit und ohne Diversity-Zwang

### Ziel

Ist die gelernte Routing-Struktur (Gini=0.13, Jaccard off-diag=0.282) eine echte
Modell-Eigenschaft — oder ein Scheduler-Effekt, der nur bei aktiver Diversity-Maske sichtbar ist?

### Implementierungsstand

**Forced-Diversity-Eval:** `iteration_diagnostics(..., run_forced=True)` — bereits in
`tinystories_exp.py` vorhanden. Erzwingt bei Iteration r, dass die Top-k Blöcke
aus r−1 gesperrt werden — identisch zum Training-Signal.

**Normal-Eval:** Standard-Eval ohne Maske (bereits Grundzustand).

```bash
python scripts/eval_suite.py \
  --ckpt checkpoints/tinystories_curriculum_fromIter2/seed_0/step_3000 \
  --tests forced_diversity \
  --device cuda
```

**Hinweis:** `run_forced=True` könnte bei Phase-3-ähnlichen Modellen (uniform Gini)
NaN-Werte liefern, wenn kumulative Ablation mehr als k Blöcke sperrt. Für Curriculum C
(Gini=0.13, aber nicht gleichmäßig genug für totalen Sperr-Overflow) voraussichtlich stabil.

### Interpretation

| Bedingung | Jaccard r2→r3 | Gini | Bedeutung |
|-----------|--------------|------|-----------|
| Normal-Eval | 0.99 (gemessen) | 0.13 (gemessen) | Lernter Zustand |
| Forced-Diversity | ? | ? | Was wenn Maske aktiv bleibt? |

- **Forced ≈ Normal:** Modell ignoriert Forced-Diversity (Routing resistent → echte gelernte Struktur).
- **Forced deutlich besser (Loss sinkt):** Diversity hilft auch bei Eval → Modell hat
  Diversity-Potential nicht vollständig internalisiert.

---

## Test 6 — Usage-gematchte Ablationen

### Ziel

Sauberster Test für funktionale Kompetenzzentren: Nicht nur Top-Lift vs. Random,
sondern explizit kontrollierten Vergleich:

1. **Eigene Kategoriegruppe** (Top-Lift-Blöcke der Ziel-Kategorie)
2. **Fremde Kategoriegruppe** (Top-Lift-Blöcke einer anderen Kategorie, gleiche n)
3. **Zufällige Gruppe** (n zufällige Blöcke)
4. **Nutzungsgematchte Gruppe** (n Blöcke mit ähnlicher globaler Nutzungsfrequenz,
   aber NICHT in der Top-Lift-Liste der Ziel-Kategorie)

Jeweils auf der Ziel-Kategorie evaluieren:

```
ΔL_eigene_Kategorie
ΔL_fremde_Kategorie     ← wenn ΔL_eigen >> ΔL_fremd: Kompetenzbeleg
ΔL_zufällig
ΔL_nutzungsgemacht
```

### Implementierungsstand

**Neuer Code nötig** in `experiments/competence_centers_exp.py`:

```python
def usage_matched_ablation_test(model, routing, cat_wins, cats, n_top=5, device="cpu"):
    """Ablation mit usage-gematchter Kontrollgruppe pro Kategorie."""
    # Für jede Kategorie:
    #   1. Top-n_top Lift-Blöcke (eigene Gruppe)
    #   2. Top-n_top Lift-Blöcke einer zufälligen anderen Kategorie (fremde Gruppe)
    #   3. n_top zufällige Blöcke (zufällig)
    #   4. n_top Blöcke mit ähnlicher globaler Frequenz wie die eigenen
    #      aber niedrigem Lift (<1.05) — nutzungsgemacht
    ...
```

Bootstrap über mehrere zufällige Auswahlen für Gruppen 3 und 4.

---

## Test 7 — Iterationsweise Ablation (alle Kategorien)

### Ziel

Ablation von Top-N-Lift-Blöcken der Ziel-Kategorie, aber getrennt pro Iteration:
`nur r=1`, `nur r=2`, ..., `nur r=6`. Über alle Kategorien, nicht nur Kausalität.

### Implementierungsstand

**Teils vorhanden:** `scripts/abl_sanity.py` macht dies für Kausalität.
`group_r1_ablation_test()` in `competence_centers_exp.py` macht r1 vs. r2–r6,
aber nicht einzelne Iterationen r=2, r=3, r=4, r=5, r=6.

**Ergänzung nötig:** 6 separate Konditionen statt 2 (r1 / r2–r6):

```python
# Für jede Kategorie, jede Iteration r=1..6:
masks = [None] * R
masks[r] = ablate_mask_for_category
loss = per_iter_fn(model, toks, tgt, masks, device)
```

Das Pattern existiert bereits in `abl_sanity.py` — muss nur in `competence_centers_exp.py`
als vollständige Funktion für alle Kategorien generalisiert werden.

---

## Test 8 — Cache- und Working-Set-Test auf Curriculum C

### Ziel

Curriculum C muss nicht nur besser spezialisiert sein — es muss auch streambar bleiben.
Messung auf denselben Routing-Traces:

- Einzigartige Blöcke pro Token (Working Set)
- Reuse-Distanz über Token hinweg
- Cache-Miss-Kurve (gelernt vs. Random)
- Hot-Block-Pinning-Effekt
- Vergleich mit Phase 2

### Implementierungsstand

**Vollständig vorhanden:** `full_eval()` + `cache_sim()` in `tinystories_exp.py`.
`full_eval()` liefert `routing`, `unique_stats`, `traces_arr`.

```bash
python scripts/eval_suite.py \
  --ckpt checkpoints/tinystories_curriculum_fromIter2/seed_0/step_3000 \
  --tests cache \
  --device cuda
```

Vergleichs-Checkpoint: Phase 2 (`checkpoints/tinystories_phase2/seed_0/step_3000`).

---

## Test 9 — Fest-verdrahtete Routing-Baseline

### Ziel

Modell mit:
- Festem r=1-Einstiegsblock (keine Routing-Entscheidung, immer derselbe Block)
- Festem r=2–r=6-Kern (die Top-4 genutzten Blöcke aus Curriculum C, immer dieselben)
- Gleiche aktive Breite (4 Blöcke/Iteration)
- Gleiche Gesamt-Parameter

Wenn Curriculum C klar besser ist → dynamisches Routing trägt wirklich etwas bei.
Wenn nicht → der Hub-Effekt erklärt alles.

### Implementierungsstand

**Neuer Code nötig.** Entweder:
- Neues Modell `ModelC_FixedRoute` in `rblm/models.py`
- Oder: Eval-Wrapper der die Router-Entscheidung durch feste Indizes ersetzt

Realistisch als Eval-Wrapper ohne Retraining implementierbar:

```python
# Feste Routing-Indizes aus Curriculum-C-Analyse ableiten:
r1_block = routing["causality"]["freq"].mean(axis=0).argmax()  # häufigster in r=1
kern_blocks = top_4_global_blocks  # aus full_eval()
```

Dann Eval mit erzwungenem Routing (ähnlich State-Reset aber für Block-Auswahl).

---

## Test 10 — Skalierung: 64 → 128 → 256 Blöcke

### Ziel

Bleibt das Working Set (~7 Blöcke/Token) konstant wenn n_blocks wächst?
Skaliert die Spezialisierung (Gini, Jaccard off-diag) mit der Bankgröße?

### Kommandos

```bash
# 128 Blöcke (~45 Min, Scratch oder Warm Start)
python -m experiments.tinystories_exp \
  --n_blocks 128 --steps 3000 --seed 0 --device cuda

# 256 Blöcke (~60–90 Min)
python -m experiments.tinystories_exp \
  --n_blocks 256 --steps 3000 --seed 0 --device cuda
```

### Schlüsselmetrik

```
n_blocks  | Working Set (mean) | Reduktion vs. Layer-Offloading
64        | ~7                 | 9.2×
128       | ?                  | ?
256       | ?                  | ?
```

Wenn Working Set ≈ konstant → Streaming-Vorteil wächst mit Modellgröße (Kernthese bestätigt).

---

## Prioritätsreihenfolge und Abhängigkeiten

```
Test 1 (Seeds 1+2)    ─────────────────────────────────→ Architekturbeleg belastbar
        │
Test 2 (No-Div)       ─────────────────────────────────→ Qualitäts-Ursache klar
        │
Test 3+4+5 (Eval)     → kein Training, sofort auf Seed 0 laufbar → Tiefenstruktur klar
        │
Test 6+7 (Ablation)   → neuer Code, kein Training nötig
        │
Test 8 (Cache)        → kein Training, Streaming-Bild vervollständigt
        │
Test 9 (Fixed)        → erst sinnvoll wenn Test 1 abgeschlossen
        │
Test 10 (Skalierung)  → erst sinnvoll wenn Tests 1–4 stabil
```

**GPU-Queue (eine Instanz gleichzeitig, RTX 2060):**

```
1. Curriculum C Seed 1  (~30 Min)
2. Curriculum C Seed 2  (~30 Min)
3. No-Diversity WS      (~30 Min)
4. (optional) 128-Block-Skalierung
```

Während GPU läuft: Tests 3, 4, 5, 8 auf bestehendem Seed-0-Checkpoint.

---

## Eval-Suite für sofortige Tests (3, 4, 5, 8)

```bash
# Alle Eval-Tests auf einem Checkpoint:
python scripts/eval_suite.py \
  --ckpt checkpoints/tinystories_curriculum_fromIter2/seed_0/step_3000 \
  --device cuda

# Einzelne Tests:
python scripts/eval_suite.py --ckpt ... --tests depth_truncation
python scripts/eval_suite.py --ckpt ... --tests state_reset
python scripts/eval_suite.py --ckpt ... --tests forced_diversity
python scripts/eval_suite.py --ckpt ... --tests cache
```

---

## Ergebnis-Tracking

*(wird nach jedem Test ausgefüllt)*

| Test | Seed | Status | val_loss | Gini | Jaccard | CLF | Ergebnis |
|------|------|--------|----------|------|---------|-----|---------|
| 1 Seed 1 | 1 | ausstehend | — | — | — | — | — |
| 1 Seed 2 | 2 | ausstehend | — | — | — | — | — |
| 2 No-Div | 0 | ausstehend | — | — | — | — | — |
| 3 Depth | 0 | **vorhanden** | — | — | — | — | Δ=0.001 (flach) |
| 4 State-Reset | 0 | ausstehend | — | — | — | — | — |
| 5 Forced | 0 | ausstehend | — | — | — | — | — |
| 8 Cache | 0 | ausstehend | — | — | — | — | — |
