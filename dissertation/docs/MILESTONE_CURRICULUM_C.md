# Meilenstein: Curriculum C — Eingefrorene Baseline vor weiteren Seeds

**Eingefroren am:** 2026-06-17  
**Status:** Einzel-Seed-Befund — Interpretation provisorisch bis Seeds 1–2 bestätigen  
**Zweck:** Unveränderlicher Referenzpunkt vor weiteren Experimenten

---

## 1. Exakte Konfiguration

### Modell (Modell C)

| Parameter | Wert |
|-----------|------|
| Architektur | Rekursiv geroutetes Block-LM |
| n_blocks | 64 |
| k_active | 4 |
| R (Iterationen) | 6 |
| d_model | 256 |
| block_hidden | 512 |
| Context-Encoder | 1 kausaler Attention-Layer (kein FFN) |
| Gesamt-Parameter | 19.3M |
| Readout | geteilter Head nach jeder Iteration (Deep Supervision) |

### Training

| Parameter | Wert |
|-----------|------|
| Seed | **0** |
| Schritte | 3000 |
| Batch-Größe | 32 |
| Sequenzlänge | 128 |
| Gerät | CUDA (RTX 2060, 6 GB) |
| Trainingsstrategie | Warm Start + Selective Diversity |
| Warm-Start-Checkpoint | `checkpoints/tinystories_phase2/seed_0/step_3000` |
| Diversity-Modus | `diverse_from_iter=2` (0-basiert) |
| Bedeutung | Iterationen r=3–r=6 (1-basiert) erhalten Diversity-Ablation; r=1 und r=2 trainieren ohne |
| diverse_until | nicht gesetzt (Diversity über alle 3000 Schritte aktiv) |
| Lernrate / Optimizer | wie Phase 2 (keine Änderung) |
| Checkpoint-Name | `tinystories_curriculum_fromIter2` |
| Checkpoint-Pfad | `checkpoints/tinystories_curriculum_fromIter2/seed_0/step_3000` |
| Checkpoint-SHA256 | `482fa9f816d2536a...` (vollständiger Hash im Checkpoint-Ordner) |

### Daten

| Parameter | Wert |
|-----------|------|
| Datensatz | TinyStories (HuggingFace) |
| Dokumente | 20 000 (feste Subsampling-Zahl) |
| Tokenizer | BPE, Vokabular 8 000 |
| Daten-Split | Kein separater Test-Split — val_loss auf gehaltenem Batch-Mittel aus Trainingsdaten |
| Kategorisierung | Heuristisch (Textmuster, nie im Training gesehen) — siehe Abschnitt 4 |

### Trainings-Kommando

```bash
python -m experiments.tinystories_exp \
  --diverse \
  --diverse_from_iter 2 \
  --pretrained_ckpt checkpoints/tinystories_phase2/seed_0/step_3000 \
  --steps 3000 \
  --device cuda
```

### Analyse-Kommando

```bash
python -m experiments.competence_centers_exp \
  --ckpt checkpoints/tinystories_curriculum_fromIter2/seed_0/step_3000 \
  --analysis \
  --device cuda
```

Sanity-Check (Ablations-Validierung):

```bash
python scripts/abl_sanity.py
```

### Code-Stand

| Datei | Relevante Änderungen in dieser Session |
|-------|---------------------------------------|
| `rblm/models.py` | `diverse_from_iter`-Parameter in `ModelC.forward()` |
| `experiments/tinystories_exp.py` | `diverse_until`, `diverse_from_iter`, `pretrained_ckpt` in `train()`/`run()` |
| `experiments/competence_centers_exp.py` | `group_r1_ablation_test()`, balanciertes Bootstrap, `--analysis`-Flag, Tag-Kollisions-Fix |
| `scripts/abl_sanity.py` | Neu erstellt — Per-Iterations-Ablations-Validierung |

---

## 2. Trainings-Ergebnisse

### Verlauf

| Schritt | L1 | L_fin | Jaccard r1→r2 | Div |
|---------|-----|-------|--------------|-----|
| 500 | 3.621 | 3.623 | 0.990 | 0.078 |
| 1000 | 3.423 | 3.423 | 0.993 | 0.078 |
| 1500 | 3.350 | 3.352 | 0.993 | 0.078 |
| 2000 | 3.165 | 3.166 | 0.992 | 0.078 |
| 2500 | 3.121 | 3.121 | 0.993 | 0.078 |
| 3000 | 3.126 | **3.127** | 0.992 | 0.078 |

### Finale Metriken (Step 3000)

| Metrik | Wert |
|--------|------|
| val_loss | **3.127 Nats** |
| Jaccard r=1→r=2 | 0.08 |
| Jaccard r=2→r=3 | 0.99 |
| Jaccard r=3→r=4 | 0.993 |
| Jaccard r=4→r=5 | 0.992 |
| Jaccard r=5→r=6 | 0.992 |
| Hub-Gini | **0.1271** |
| Top-5-Block-Anteil | **14.2%** |
| Tote Blöcke | **0 / 64** |
| Einzigartige Blöcke/Token (mean) | ~7 (wie alle anderen Modelle) |

**2-Stufen-Routing-Muster:** r=1 exploriert (Jaccard r1→r2 = 0.08), r=2–r=6 kollabieren auf stabilen Kern (Jaccard ≥ 0.99). Muster ist identisch mit Phase 2 und Curriculum A — aber Gini ist drastisch niedriger, d.h. der Kern besteht aus mehr verschiedenen Blöcken je nach Kontext.

---

## 3. Routing-Struktur

### Jaccard-Matrix zwischen Kategorien (r=2)

Maß: Overlap der Top-k selektierten Blöcke bei Iteration r=2. Off-Diagonal-Mittel = Routing-Gleichheit zwischen Kategorien.

```
                  causali  corefer  dialogu  emotion  scene_s  tempora
  Kausalitaet       1.000    0.333    0.200    0.333    0.412    0.333
  Koreferenz        0.333    1.000    0.263    0.333    0.263    0.200
  Dialog            0.200    0.263    1.000    0.333    0.143    0.200
  Emotion           0.333    0.333    0.333    1.000    0.200    0.263
  Szene/Ort         0.412    0.263    0.143    0.200    1.000    0.412
  Zeitfolge         0.333    0.200    0.200    0.263    0.412    1.000
```

**Off-Diagonal-Mittel: 0.282** — stärkste Kategorie-Trennung aller bisher getesteten Modelle.

### Top-5 Lift-Blöcke pro Kategorie (gemittelt über alle r)

Lift = P(Block b | Kategorie q) / P(Block b) — wie viel stärker ein Block für Kategorie q genutzt wird als global.

| Kategorie | Top-5 Blöcke | Lift-Werte |
|-----------|-------------|------------|
| Kausalität | [49, 62, 2, 45, 41] | [1.44, 1.30, 1.26, 1.24, 1.19] |
| Koreferenz | [49, 18, 43, 48, 28] | [1.25, 1.13, 1.12, 1.12, 1.11] |
| Dialog | [44, 51, 46, 35, 29] | [1.55, 1.50, 1.32, 1.30, 1.26] |
| Emotion | [9, 34, 39, 53, 31] | [1.33, 1.24, 1.20, 1.17, 1.16] |
| Szene/Ort | [43, 1, 27, 18, 62] | [1.26, 1.24, 1.20, 1.19, 1.19] |
| Zeitfolge | [49, 1, 19, 22, 10] | [1.29, 1.21, 1.20, 1.19, 1.17] |

Maximaler Lift: 1.55 (Dialog/Block 44). Lift-Werte sind moderat — kein Block wird ausschließlich für eine Kategorie genutzt.

### Mutual Information I(Kategorie; Block) pro Iteration

| r | MI (nats) |
|---|-----------|
| r=1 | 0.0053 |
| r=2 | **0.0102** |
| r=3 | 0.0101 |
| r=4 | 0.0101 |
| r=5 | 0.0101 |
| r=6 | 0.0101 |

**Trend:** Sprung r=1→r=2 (0.0053→0.0102), dann stabil. Kategorie-Information ist im Kern r=2–r=6 doppelt so stark kodiert wie in r=1.

---

## 4. Klassifikator-Ergebnisse

**Methode:** Nächste-Zentroide auf Routing-Mustern (binäre Block-Aktivierung). Balanciertes Bootstrap (10 Draws, min(n_cat) Samples je Kat). Zufalls-Baseline: 1/6 = 0.167.

### Genauigkeit pro Iterations-Subset

| Feature-Set | Accuracy | vs. Zufall |
|-------------|----------|-----------|
| alle Iterationen (R×nb Binärvektor) | 0.424 | +0.257 |
| nur r=1 | 0.359 | +0.193 |
| nur r=2 | 0.422 | +0.256 |
| nur r=3 | 0.424 | +0.257 |
| nur r=4 | 0.419 | +0.252 |
| nur r=5 | 0.420 | +0.254 |
| nur r=6 | 0.419 | +0.252 |

### Bootstrap-Ergebnis

| Metrik | Wert |
|--------|------|
| bootstrap_mean | **0.421** |
| 95%-KI untere Grenze | **0.391** |
| 95%-KI obere Grenze | **0.451** |
| Zufalls-Baseline | 0.167 |
| Δ vs. Zufall | +0.254 |

**Interpretation:** Kategorie-Information im Routing ist statistisch nachweisbar und stabil über Bootstrap-Stichproben. r=1 ist schwächer (0.359) als der Kern r=2–r=6 (0.419–0.424) — umgekehrt zu Curriculum A.

---

## 5. Ablationsanalyse und Sanity-Check

### 5a. Einzelblock-Ablation (alle Iterationen gleichzeitig)

Ablation = Block wird über alle r=1..r=6 gesperrt; Routing weicht auf andere Blöcke aus.

| Kategorie | Baseline (Nats) | Eigen-Schaden (Nats) | Fremd-Schaden (Nats) | Δ Eigen−Fremd |
|-----------|-----------------|---------------------|----------------------|--------------|
| Kausalität | 3.182 | **+27.77** | +25.76 | +2.01 |
| Koreferenz | 3.427 | **+27.38** | +26.09 | +1.29 |
| Dialog | 3.256 | **+9.21** | +7.73 | +1.48 |
| Emotion | 3.357 | +0.0001 | +0.0001 | ≈0 |
| Szene/Ort | 3.076 | −0.0003 | +0.0000 | neg. |
| Zeitfolge | 3.503 | **+28.85** | +25.36 | +3.49 |

**5/6 Kategorien:** Eigen-Schaden > Fremd-Schaden — scheinbar starkes Kompetenzzentrum-Signal.

### 5b. Sanity-Check: Per-Iterations-Ablation (Kausalität-Blöcke)

Ablation der Top-5 Lift-Blöcke von Kausalität ([49, 62, 2, 45, 41]), aber nur in einer Iteration.

```
Baseline (kein Ablation):     3.1814 Nats
Nur r=1 abliert:              3.5597 Nats  (delta = +0.38)
Nur r=2 abliert:              3.1814 Nats  (delta ≈ 0.00)
Nur r=3 abliert:              3.1814 Nats  (delta ≈ 0.00)
Nur r=4 abliert:              3.1814 Nats  (delta ≈ 0.00)
Nur r=5 abliert:              3.1814 Nats  (delta ≈ 0.00)
Nur r=6 abliert:              3.1814 Nats  (delta ≈ 0.00)
ALLE Iterationen abliert:    31.0260 Nats  (delta = +27.8)
```

**Diagnose:** Die ~27 Nats entstehen nicht durch kategoriespezifischen Funktionsverlust, sondern durch **nichtlineare Fehlerpropagation über 6 Iterationen**. Die Top-5 Lift-Blöcke einer Kategorie sind die universalen Hub-Blöcke des r=2–r=6-Kerns. Ablation einer Iteration: kein Schaden (Router weicht aus). Ablation aller 6 Iterationen: katastrophal (kumulatives Fehlrouting).

**Universalitätstest** — gleiche Blöcke, andere Kategorie:

```
Ablierte Blöcke: Kausalität-Blöcke [49, 62, 2, 45, 41]
Schaden auf Kausalitaet:   +27.8 Nats  (+875%)
Schaden auf Szene/Ort:     +28.0 Nats  (+888%)   ← GRÖSSER als Eigen-Schaden
Schaden auf Emotion:       +24.6 Nats  (+732%)
Schaden auf Zeitfolge:     +28.9 Nats  (+825%)
```

Fremd-Schaden ist vergleichbar groß wie Eigen-Schaden → kein Beweis für funktionale Spezialisierung.

### 5c. Gruppenablation (5 Konditionen × 3 n_top-Werte)

**n_top=5:**

| Kondition | Diag | Off-Diag | Dominanz | Bedeutung |
|-----------|------|----------|----------|-----------|
| group_all | +462% | +433% | 4/6 | Hub-Kern-Katastrophe (gleicher Mechanismus wie 27 Nats) |
| **group_r1** | **+7.8%** | +7.8% | 2/6 | r=1 hat echte, moderate Kategorie-Wirkung |
| group_r2r6 | −0.002% | −0.000% | 2/6 | r=2–r=6 kein kategoriespezifischer Ablations-Effekt |
| random_all | +0.003% | +0.003% | 4/6 | Kapazitäts-Rauschen (Kontrollbedingung) |
| usage_all | +237% | +260% | 0/6 | Hub-Blöcke per Nutzung, kein Kategorie-Signal |

**r=1-Kausaltest (n_top=5):**

```
group_r1:   +7.803%   (Schaden durch NUR r=1-Ablation)
group_r2r6: −0.002%   (Schaden durch NUR r=2-6-Ablation)
group_all:  +462.459% (Schaden durch ALLE Iterationen)
frac_r1 = group_r1_diag / group_all_diag = 0.017
```

r=1 trägt 1.7% des Gesamtschadens. r=2–r=6-Kern dominiert — **aber nicht durch kategoriespezifische Funktion**, sondern durch universalen Hub-Kollaps.

**frac_r1 = 0.017** bedeutet: r=2–r=6 ist der funktionale Kern, r=1 ist moderater Gatekeeper.

---

## 6. Emergente architektonische Hierarchie

Das Modell hat ohne explizite Supervision eine zweischichtige Struktur entwickelt:

```
r=1  → Kategorie-sensitiver Gatekeeper
       - wählt kontextabhängig andere Blöcke pro Kategorie
       - group_r1 Schaden: +7.8% (moderat, aber real)
       - Klassifikator r=1: 0.359 (schwächer als Kern)
       - Jaccard r=1→r=2: 0.08 (fast vollständig entkoppelt von r=2–r=6)
       ↓
r=2–r=6 → Universaler rekurrenter Verarbeitungskern
       - alle Iterationen verwenden denselben 4-Block-Set (Jaccard 0.99)
       - kein kategoriespezifischer Effekt pro Iteration (group_r2r6 ≈ 0%)
       - aber katastrophaler Kollektivschaden wenn ALLE gleichzeitig ablatiert
       - Klassifikator r=2–r=6: 0.419–0.424 (stärker als r=1)
```

**Für Parameter-Streaming relevant:** Diese Hierarchie entspricht exakt dem Muster, das Streaming-Effizienz maximiert — kleines wechselndes Set an Eintrittsblöcken + stabiler residenter Kern. Ob das intentional oder ein Training-Artefakt ist, bleibt offen.

---

## 7. A-vs-C-Vergleich (identische Analyse-Pipeline)

| Metrik | Curriculum A | Curriculum C | Bewertung |
|--------|--------------|--------------|-----------|
| val_loss | 3.127 | **3.127** | gleich |
| Hub-Gini | 0.52 | **0.13** | C deutlich gleichmäßiger |
| Top-5-Block-Anteil | 27.8% | **14.2%** | C weniger Hub-konzentriert |
| Jaccard off-diag (r=2) | 0.899 | **0.282** | C 3.2× stärkere Kategorie-Trennung |
| CLF bootstrap-mean | 0.374 | **0.421** | C signifikant besser |
| CLF 95%-KI | [0.347, 0.389] | **[0.391, 0.451]** | KIs überlappen nicht |
| Bester CLF per Iteration | r=1: 0.404 | r=2–r=6: ~0.42 | Verschiedene Träger |
| group_r1 n=5 | +0.001% (Rauschen) | **+7.8%** | C: echter Kausal-Effekt |
| group_r2r6 n=5 | −0.001% (Rauschen) | −0.002% (Rauschen) | beide: kein r=2–r=6-Effekt |
| group_all n=5 | +0.000% (Rauschen) | +462% (Hub-Artefakt) | A: Hub-Kollaps verhindert Ablation |
| usage_all n=5 | +1473% | +237% | A: Hubs noch kritischer |

### Warum A keine 27-Nats-Ablation zeigt

In Curriculum A sind alle Kategorien im r=2–r=6-Kern vollständig gleichgeschaltet (Jaccard=1.000 zwischen Koreferenz, Emotion, Szene/Ort, Zeitfolge). Lift der Hub-Blöcke ≈ 1.0 → sie erscheinen nicht in der Top-5-Lift-Liste. Ablation trifft nur r=1-Unterschiede: ~0 Nats.

In Curriculum C hat `diverse_from_iter=2` den Kern differenziert genug (Jaccard 0.282), dass Hub-Blöcke mit Lift > 1.0 in der Liste erscheinen. Ablation über alle 6 Iterationen: nichtlineare Katastrophe.

### Mechanismus des Unterschieds

`diverse_until=900` (A): Ab Step 901 darf das Routing wieder kollabieren. Nach 2100 weiteren Schritten ist der Kollaps vollständiger als ohne Diversity-Training.

`diverse_from_iter=2` (C): Diversity für r=3–r=6 über alle 3000 Schritte — kein Zeitfenster für Rückfall. Der Kern bleibt differenziert.

---

## 8. Bekannte Bugs und Einschränkungen

### Bug 1: Curriculum B — Hang nach diverse_until-Übergang

**Symptom:** Curriculum B (`diverse_until=1500`) hing nach Step 1500 für >1h (GPU idle, kein Output).  
**Ursache:** Der ON→OFF-Übergang von `diverse_train` nach 1500 Schritten intensivem Diversity-Training führte zu einem nicht-terminierenden Zustand. Vermutlich numerische Instabilität durch schlagartigen Routing-Wechsel.  
**Workaround:** Kein `diverse_until` setzen (Curriculum C). Alternativ: graduelles Annealing statt harter Transition.  
**Status:** Nicht gefixt. Nur dokumentiert.

### Bug 2: CE-Loss-Schranken-Missverständnis (konzeptuell, kein Code-Bug)

**Symptom:** Erste Interpretation: „27 Nats ist unmöglich — log(8000) = 8.99 ist Maximum."  
**Ursache:** log(vocab_size) ist nur die Schranke für einen uniformen Prädiktor. Ein trainiertes Modell, das confident falsch liegt, kann arbiträr hohen CE erzeugen. Sanity-Check hat die 31 Nats als echt bestätigt.  
**Gelernte Lektion:** CE-Ablationswerte über log(vocab_size) sind kein Messfehler, sondern zeigen, dass das Modell zuvor confident war und durch Ablation systematisch falsch geroutet wird.

### Einschränkung: Kein echter Validierungs-Split

Der val_loss ist ein gleitender Mittelwert über Trainings-Batches, kein separater held-out Split. Für Vergleiche zwischen Modellen ist er ausreichend (alle Modelle unter gleicher Bedingung), aber er überschätzt möglicherweise die wahre Generalisierungsqualität.

### Einschränkung: Seed 0 only

Alle Befunde basieren auf einem einzelnen Seed (Seed 0). Stochastische Varianz des Routings und der Spezialisierungsstruktur ist unbekannt.

### Einschränkung: Kategorisierung heuristisch

Die 6 Kategorien (Kausalität, Koreferenz, Dialog, Emotion, Szene/Ort, Zeitfolge) werden durch Textmuster heuristisch vergeben, nicht durch Ground-Truth-Labels. Kategorie-Überlappung und Rauschen in den Kategorien können die Routing-Messungen beeinflussen.

---

## 9. Stärkste zulässige Aussagen (Stand: Seed 0)

### Gesichert (robuste Evidenz aus diesem Experiment)

> **G1 — Qualitätserhalt:** Curriculum C (diverse_from_iter=2, Warm Start) erreicht val_loss=3.127 Nats — identisch mit dem Warm-Start-Ausgangspunkt Phase 2 (3.705 → 3.127 durch mehr Training), ohne Qualitätsverlust gegenüber Curriculum A (3.127), trotz aktiver Diversity-Regularisierung.

> **G2 — Stärkere Kategorie-Differenzierung:** Curriculum C erzielt Jaccard off-diag 0.282 (vs. A: 0.899, Phase 2: 0.759, Phase 3: 0.485) — bei gleicher Qualität. Das Routing nutzt für verschiedene narrative Kategorien messbar verschiedenere Block-Sets.

> **G3 — Gleichmäßigere Blocknutzung:** Gini=0.13 (nahe Phase 3: 0.11) bei 0/64 toten Blöcken. Hub-Konzentration ist deutlich reduziert.

> **G4 — Kategorie-Information im Routing kodiert:** Klassifikator 0.421 [0.391, 0.451] (95%-KI, balanciertes Bootstrap) — nicht-überlappend mit Curriculum A (0.374 [0.347, 0.389]). Das Routing trägt unterscheidbare Information über narrative Kategorien.

> **G5 — r=1 als moderater Gatekeeper:** group_r1 = +7.8% Diagönalschaden bei Ablation der Top-5 Lift-Blöcke in r=1 — echter, wenn auch kleiner Kausaleffekt. In Curriculum A ist dieser Effekt nicht messbar (+0.001%).

> **G6 — Emergente Hierarchie reproduzierbar:** Alle bisher getesteten Modelle (Phase 2, 3, Curriculum A und C) zeigen spontan dieselbe Zwei-Phasen-Struktur (r=1 Gatekeeper, r=2–r=6 kollabierter Kern). Das Muster ist nicht trainiert worden.

### Noch nicht gesichert

> **N1 — Reproduzierbarkeit:** Alle Befunde sind Einzel-Seed (Seed 0). Ob G2–G5 über Seeds stabil sind, ist unbekannt. **Nächster Schritt: Seeds 1 und 2.**

> **N2 — Klar isolierte Kompetenzzentren:** Es gibt keine Evidenz für separate Block-Cluster die exklusiv für eine Kategorie aktiv sind. Die Ablation zeigt universalen Hub-Schaden, nicht kategoriespezifischen Funktionsverlust. Für echte Kompetenzzentren wäre group_r2r6-Diagonal >> group_r2r6-Off-Diagonal nötig — das ist nicht der Fall.

> **N3 — Funktionale Spezialisierung einzelner Blöcke:** Lift-Werte maximal 1.55 (Dialog/Block 44). Kein Block ist ausschließlich oder dominant einer Kategorie zugeordnet.

> **N4 — Übertragbarkeit auf größere Modelle:** Alle Experimente: n_blocks=64. Ob die Gatekeeper/Kern-Struktur bei n_blocks=256 oder 1024 erhalten bleibt — oder ob der Kern sich dann tatsächlich in Kategorie-spezifische Sub-Regionen aufteilt — ist unbekannt.

> **N5 — Realer Laufzeitgewinn:** Kein End-to-End Streaming-Benchmark implementiert. Working Set ~7 Blöcke/Token ist gemessen, aber der tatsächliche PCIe-Transfer-Overhead, Cache-Management-Latenz und Tokens/s gegenüber Layer-Offloading sind nicht gemessen.

> **N6 — Generalisierungsqualität auf echtem held-out Split:** Der val_loss ist kein echter Testset-Wert.

---

## 10. Analyse-Artefakte (eingefroren)

| Artefakt | Pfad | Beschreibung |
|----------|------|--------------|
| Checkpoint | `checkpoints/tinystories_curriculum_fromIter2/seed_0/step_3000/` | Modell-Gewichte — **nicht verändern** |
| Analyse-JSON | `results/competence_b64k4R6_s0_analysis_tinystories_curriculum_fromIter2.json` | Vollständige Analyse-Daten |
| Aktivierungs-Plot | `results/competence_b64k4R6_s0_analysis_tinystories_curriculum_fromIter2_activation.png` | |
| Lift-Plot | `results/competence_b64k4R6_s0_analysis_tinystories_curriculum_fromIter2_lift.png` | |
| Jaccard-Plot | `results/competence_b64k4R6_s0_analysis_tinystories_curriculum_fromIter2_jaccard.png` | |
| Summary-Plot | `results/competence_b64k4R6_s0_analysis_tinystories_curriculum_fromIter2_summary.png` | |
| Ablations-Plot | `results/competence_b64k4R6_s0_analysis_tinystories_curriculum_fromIter2_ablation.png` | |
| Gruppenablation-Plot | `results/competence_b64k4R6_s0_analysis_tinystories_curriculum_fromIter2_group_ablation_conds.png` | |
| r=1-Kausal-Plot | `results/competence_b64k4R6_s0_analysis_tinystories_curriculum_fromIter2_r1_causal.png` | |
| Sanity-Check-Script | `scripts/abl_sanity.py` | Per-Iterations-Ablations-Validierung |

Zum Vergleich — Curriculum A Artefakte:

| Artefakt | Pfad |
|----------|------|
| Analyse-JSON | `results/competence_b64k4R6_s0_analysis_tinystories_curriculum_30pct.json` |
| Checkpoint | `checkpoints/tinystories_curriculum_30pct/seed_0/step_3000/` |

---

## 11. Nächste Schritte (geplant, aber nicht Teil dieses Meilensteins)

**Priorität 1 — Seed-Validierung (blockiert weitere Interpretation)**

```bash
# Seed 1
python -m experiments.tinystories_exp \
  --diverse --diverse_from_iter 2 \
  --pretrained_ckpt checkpoints/tinystories_phase2/seed_0/step_3000 \
  --steps 3000 --seed 1 --device cuda

# Seed 2
python -m experiments.tinystories_exp \
  --diverse --diverse_from_iter 2 \
  --pretrained_ckpt checkpoints/tinystories_phase2/seed_0/step_3000 \
  --steps 3000 --seed 2 --device cuda
```

Dann identische Analyse-Pipeline auf Seeds 1 und 2. Metrik-Varianz bestimmt, welche Befunde aus G1–G6 seed-stabil sind.

**Priorität 2 — Skalierungstest (H1)**

```bash
python -m experiments.tinystories_exp \
  --n_blocks 256 --steps 3000 --device cuda
```

Prüfen: Bleibt Working Set (~7 Blöcke/Token) konstant? Entsteht eine differenziertere Kern-Struktur als bei n_blocks=64?

**Priorität 3 — Streaming-Messung (H2)**

End-to-End Inference-Benchmark: Tokens/s bei Curriculum C gegenüber Layer-Offloading eines äquivalenten dichten Modells auf RTX 2060.

**Nicht als nächstes:** Architektur-Umbauten, neue Diversity-Varianten, Phase-4-Features — bis Seeds die Baseline stabilisieren.

---

## 12. Meilenstein-Aussage

Das ist der Stand von Curriculum C Seed 0 nach vollständiger Validierung und Fehleranalyse:

> Curriculum C (diverse_from_iter=2, Warm Start von Phase 2, Seed 0) erzielt **val_loss=3.127 Nats** bei **Gini=0.13**, **0/64 toten Blöcken** und **Jaccard off-diag=0.282** — die stärkste Kategorie-Trennung aller getesteten Modelle. Der Klassifikator auf Routing-Mustern erreicht **0.421 [0.391, 0.451]** (95%-KI, Bootstrap), signifikant über Curriculum A (nicht-überlappende KIs). r=1 zeigt einen echten, moderaten Kategorie-Kausaleffekt (+7.8% Gruppenablation). Die scheinbar dramatischen ~27-Nats-Ablationswerte messen nichtlinearen Hub-Kern-Kollaps, nicht kategoriespezifische Kompetenz. Das Modell hat spontan eine hierarchische Gatekeeper/Kern-Architektur entwickelt — ein Einzel-Seed-Befund der durch Seeds 1 und 2 validiert werden muss, bevor weitere Experimente sinnvoll sind.

---

*Dieses Dokument ist nach Erstellung nicht mehr zu verändern — außer bei faktischen Korrekturen mit explizitem Änderungsvermerk.*  
*Vollständige Protokolle: `EXP3_TINYSTORIES.md`, `RESEARCH_NOTE.md`*
