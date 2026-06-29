# Forschungsnotiz: Rekursives Block-Sparsames Sprachmodell
**Datum:** 2026-06-16  
**Status:** Interner Befundbericht — Phasen 1–3 abgeschlossen, Phase 4 (Skalierung) ausstehend  
**Modell:** 19.3M Parameter, TinyStories, RTX 2060 (6 GB VRAM)

---

## Kernhypothese

Ein rekurrent geroutetes, block-sparsames Sprachmodell lernt stabile, wiederverwendbare
Aktivierungsmuster (Tunnels) über natürliche Sprache. Diese Tunnel-Lokalität hält das
aktive Working Set pro Token klein und von der Bankgröße weitgehend unabhängig — was
Parameter-Streaming ermöglicht, bei dem nur ein Bruchteil der Gesamtgewichte pro Token
geladen werden muss.

---

## Architektur (Modell C)

```
Eingabe-Token
    ↓  Embedding + Positionscodierung + Kontext-Aufmerksamkeit
   h_0  (d=256)
    ↓
  [r=1] Router wählt top-k aus n_blocks: h_1 = h_0 + Σ_b gate_b · F_b(h_0)
  [r=2] Router wählt top-k aus n_blocks: h_2 = h_1 + Σ_b gate_b · F_b(h_1)
   ...
  [r=R] h_R → Ausgabe-Logits (Anytime: jedes r erzeugt Logits)
```

- **Blockbank:** n_blocks unabhängige MLP-Blöcke (ResidualMLP, 2 Schichten)
- **Routing:** gelerntes Query-Key-Produkt, top-k mit Gumbel-Noise, Softmax-Gates
- **Rekursion:** h_{r} = h_{r-1} + sparse_dispatch(h_{r-1}, top-k blocks)
- **Supervision:** End-gewichteter Loss über alle R Iterationen (Deep Supervision)
- **Konfiguration:** n_blocks=64, k=4, R=6, d=256, block_hidden=512

---

## Validierte Befunde

### F1 — Sprachlernen ✓

Verlust nach 3000 Schritten auf TinyStories: **3.71 Nats**  
Zufalls-Baseline (ln vocab=8000): **8.99 Nats**  
Das Modell lernt echte Sprache, kein Artefakt.

### F2 — Tunnel-Lokalität auf natürlicher Sprache ✓

| Iterationspaar | r=1→2 | r=2→3 | r=3→4 | r=4→5 | r=5→6 |
|---|---|---|---|---|---|
| Jaccard (3000 Schr.) | 0.202 | 0.983 | 0.984 | 0.985 | 0.984 |

**Zwei-Phasen-Muster:**
- Eintrittsphase (r=1→2, Jaccard=0.20): jedes Token wählt beim ersten Schritt
  einen individuellen Einstiegspfad in den Blockraum.
- Hauptkanal (r=2..6, Jaccard≈0.984): das Token verbleibt in einem stabilen
  lokalen Block-Subspace für alle weiteren Iterationen.

Dieses Muster tritt identisch auf synthetischen Daten auf (Exp2: 0.93–0.95)
und überträgt sich 1:1 auf natürliche Sprache.

Die **vollständige Jaccard-Matrix** (alle Iterationspaare, 200-Schr.-Modell) zeigt
zusätzlich eine langsame Drift: Jaccard(r2, r6) = 0.79, d.h. das Routing ist nicht
vollständig eingefroren — benachbarte Iterationen teilen 93–97% der Blöcke,
aber über 4 Schritte akkumuliert sich ein 21% Drift.

### F3 — Block-Spezialisierung ✓

| Metrik | 200 Schritte | 3000 Schritte |
|---|---|---|
| Gini-Koeffizient | 0.91 | 0.62 |
| Top-5-Block-Anteil | 83.4% | 37.6% |
| Tote Blöcke | 26 / 64 | 0 / 64 |

Der Load-Balancing-Loss sorgt für Aktivierung aller Blöcke bei ausreichendem Training.
Hub-Struktur (Konzentration) bleibt bestehen aber wird ausgeglichener.

### F4 — Kleines Working Set ✓

Einzigartige Blöcke pro Token: **mean=6.9, p50=7, p90=8** (Maximum=24=R×k)

Das Working Set bleibt bei ~7 unabhängig von der Hub-Verteilung (ob 26 oder 0 Blöcke tot
sind — die aktiven ~7 pro Token sind stabil). Das ist der Kernmechanismus für Streaming:

| Bankgröße | Theoretische aktive Blockmenge vs. Layer-Offloading |
|---|---|
| 64 Blöcke (gemessen) | 9.2× kleiner |
| 256 Blöcke (Projektion*) | ~37× kleiner |
| 1 000 Blöcke (Projektion*) | ~144× kleiner |

*Projektion unter der Annahme, dass Working Set, Routing-Qualität und Dispatch-Overhead
bei größeren Bänken stabil bleiben. Keines davon ist bei n_blocks > 64 empirisch gezeigt.

### F5 — Cache-Lokalität ✓

| Cache-Kapazität | Gelernt | Zufall | Faktor |
|---|---|---|---|
| 8 Blöcke (12.5%) | 0.191 | 0.896 | 4.7× |
| 16 Blöcke (25%) | 0.129 | 0.768 | 6.0× |
| 32 Blöcke (50%) | 0.054 | 0.515 | 9.5× |

Echte Cache-Lokalität gegenüber uniformem Zufalls-Routing — kein Artefakt der
Hub-Konzentration allein (Kontrolle: randomisiertes Routing mit gleicher Hub-Verteilung
hat deutlich höhere Miss-Rate).

### F6 — Rekursive Zustandsänderung ✓

Relative Zustandsänderung ||h_r − h_{r-1}|| / ||h_{r-1}||:

| r=1 | r=2 | r=3 | r=4 | r=5 | r=6 |
|---|---|---|---|---|---|
| **9.663** | 0.224 | 0.183 | 0.155 | 0.135 | 0.119 |

Die 9.663 bei r=1 ist der Sprung vom Kontext-Encoding zur ersten Block-Anwendung.
Für r=2..6: 12–22% Zustandsänderung pro Schritt — keine Fixed-Point-Konvergenz,
echte akkumulierte Berechnung.

### F7 — Steuerbarer Zielkonflikt: Reuse vs. Novelty ✓ (Kernbefund)

**Methode:** Erzwungene Diversität — bei Iteration r werden die k Top-Blöcke aus
Iteration r−1 gesperrt (kumulativ). Der Router muss einen anderen Block-Set wählen.

| Iter. r | Loss (Normal) | Loss (Forced) | Δ |
|---|---|---|---|
| 1 | 6.252 | 6.252 | ±0.000 |
| 2 | **6.215** | 6.238 | +0.023 |
| 3 | 6.224 | 6.202 | **−0.022** |
| 4 | 6.247 | 6.195 | **−0.053** |
| 5 | 6.276 | 6.206 | **−0.069** |
| 6 | 6.305 | **6.201** | **−0.104** |

*(200-Schritte-Modell; 3000-Schritte-Routing kollabiert stärker — Befund dort noch ausgeprägter)*

**Zwei Verlaufsformen:**
- **Normal-Routing:** Minimum bei r=2 (6.215), dann monoton schlechter bis 6.305.
  Iterationen r=3..6 schaden dem Modell mit dem gelernten Routing.
- **Forced-Diversity:** Monoton besser bis r=4 (6.195), dann stabil ~6.20.
  Die Architektur profitiert von Tiefe — wenn das Routing zur Diversität gezwungen wird.

**Schluss:** Das Routing-Kollaps-Muster (Jaccard≈0.984) ist ein **Trainingsdefekt**,
kein nutzbares Feature. Das Modell lernt, dieselben Blöcke zu wiederholen, weil das
kurzfristig die Loss-Oberfläche minimiert — auf Kosten der späteren Iterationen.

Daraus entsteht ein identifizierter, steuerbarer Zielkonflikt:

> **Reuse** (gleiche Blöcke → weniger Transfer, effizient)  
> **vs. Novelty** (neue Blöcke → bessere Ausgabe, tieferer Vorteil)

Dieser Konflikt ist lernbar: mit einem expliziten Diversitätsloss kann das Modell
einen optimalen Punkt auf dieser Kurve finden statt im Extrempunkt „immer Reuse" zu kollabieren.

---

## Abgrenzung

### Mixture-of-Experts (MoE)

MoE routet einmal pro Schicht, ohne Rekursion. Kein akkumulierter Zustand über Iterationen.
Kein Tunnel-Effekt. Parameter-Streaming ist kein Designziel. In Standard-MoE lädt
jede Schicht k von n Experten — aber über alle Schichten akkumuliert sich ein großes
Transfer-Budget. Unser Working Set ist über R Iterationen konstant ~7, nicht R × k.

### Recurrent / Looped Depth (Universal Transformer, DEQ)

Geteilte Gewichte über Iterationen (ein Block, oft voll dicht), kein Block-Routing.
Jede Iteration lädt alle Parameter → kein Streaming-Vorteil.
DEQ sucht explizit Fixed-Points — unser Modell konvergiert nicht zum Fixed-Point
(F6: 12–22% Zustandsänderung bleibt bestehen).

### Layer-Offloading (korrekte Baseline)

Das Vergleichssystem für dieses Projekt: alle Gewichte eines Tokens werden pro
Forward-Pass geladen (100% der Blöcke). Unser System lädt ~7 Blöcke (n-unabhängig).
Das ist der mechanische Vorteil — nicht Vergleich mit Quantisierung oder Pruning,
die orthogonale Techniken sind und kombiniert werden könnten.

---

## Was fehlt — Zwei harte Nachweise

### H1 — Skalierung

**These:** Working Set (~7 Blöcke/Token) bleibt konstant wenn n_blocks von 64 auf
256, 1000, 10000 wächst.

**Status:** Nicht gezeigt. Alle Experimente: n_blocks=64.  
**Risiko:** Bei sehr großen Bänken könnte der Router die Konzentration verlieren,
das Working Set wächst, der Streaming-Vorteil schrumpft.  
**Nächster Test:** Gleiche Konfiguration mit n_blocks=256, 1024. Working Set messen.
Jaccard und Gini vergleichen.

### H2 — Systemgewinn

**These:** Ein Modell größer als der VRAM läuft auf Consumer-Hardware (z.B. RTX 2060,
6 GB) mit brauchbaren Tokens/s und akzeptablem Qualitätsverlust vs. voll-residente Referenz.

**Status:** Nicht gemessen. Kein End-to-End Streaming-Benchmark implementiert.  
**Risiko:** Selbst bei kleinem Working Set könnte der Overhead (PCIe-Transfer, Cache-Management,
Dispatch-Latenz) die Tokens/s so stark drücken, dass das System praktisch nicht nutzbar ist.  
**Nächster Test:** Streaming-Simulator mit realen Timings. Dann GPU-Implementierung.

Ohne H1 und H2 ist das ein **Architekturbeleg** mit plausiblem Skalierungspfad —
kein systemischer Beweis.

---

## Exp4 — Kompetenzzentren-Analyse: Vollständige Ergebnisse (Phase 2, Phase 3, Curriculum C)

**Ziel:** Bildet das Modell ohne Aufgabenlabels spontan verschiedene funktionale
Block-Regionen für narrative Kategorien? Und welche Trainingstrategie erzeugt die
stärkste funktionale Spezialisierung?

**Methode:** 6 heuristische Kategorien (Dialog, Kausalität, Zeitfolge, Emotion,
Koreferenz, Szene/Ort) aus Textmerkmalen — nie im Training verwendet.
Metriken: P(b|q,r), Lift, Jaccard zwischen Kategorien, MI, Klassifikator,
Gruppenablation mit Per-Iterations-Kontrolle.

### Zentrale Vergleichstabelle (Phase 2 / Phase 3 / Curriculum C)

| Metrik | Phase 2 | Phase 3 | Curriculum C |
|--------|---------|---------|--------------|
| val_loss | 3.705 | 3.891 | **3.127** |
| Jaccard off-diag (Kategorien) | 0.759 | 0.485 | **0.282** |
| Bester Klassifikator | 0.380 | 0.430 | **0.421** (KI [0.391, 0.451]) |
| group_all n=5 (Diagonal) | ~0% | +0.963% | +462%* |
| group_r1 n=5 | ~0% | +0.037% | **+7.8%** |
| group_r2r6 n=5 | ~0% | +0.101% | ~0% |
| frac_r1 | Rauschen | 0.038 | 0.017 |
| Hub-Gini | 0.62 | **0.11** | **0.13** |

*462% misst Hub-Kollaps-Katastrophe, nicht Kompetenzzentren (siehe unten).

### Phase 2 — Schwache Spezialisierung

- Kausalität ↔ Koreferenz ↔ Szene/Ort: Jaccard=1.00 (identisches Routing!)
- Dialog als Satellit (Block 62 Lift=2.03, Jaccard=0.50 zum Rest)
- Einzelblock-Ablation: ±0.0001 Nats (Rauschen) — keine funktionalen Kompetenzzentren
- Gruppenablation: ~0% in allen Konditionen — kein Spezialisierungsnachweis

### Phase 3 — Moderate Spezialisierung

- Jaccard off-diag: 0.485 (Kategorien nutzen verschiedenere Blöcke)
- Einzelblock-Ablation Kausalität: +0.1259 Nats — erster echter Schaden
- group_r2r6 = +0.101%: Kern r=2–6 trägt Kategorie-Information
- frac_r1 = 0.038: r=2–6 ist der primäre Spezialisierungsträger

### Curriculum C — Stärkste Spezialisierung (mit Einschränkung)

**Robust:**
- Jaccard off-diag 0.282: stärkste Kategorie-Trennung aller Modelle
- CLF 0.421 (KI [0.391, 0.451]): signifikant, stabil über Bootstrap
- group_r1 = +7.8%: r=1 hat echte, moderate Kategorie-Spezialisierung
- Gini 0.13: breite Blocknutzung, kein Hub-Kollaps

**Sanity-Check — 27 Nats Einzelblock-Ablation erklärt:**

Die ursprüngliche Auswertung zeigte Ablationswerte von ~27 Nats — scheinbar
dramatische Kompetenzzentren. Ein Per-Iterations-Sanity-Check klärte:

```
Nur r=1 abliert:            +0.38 Nats   (kleiner aber echter Effekt)
Nur r=2–r=6 abliert:        ≈ 0.00 Nats  (kein Einzelschaden)
ALLE Iterationen abliert:  +27.8 Nats    (nichtlineare Katastrophe)
```

Die Top-5 Lift-Blöcke einer Kategorie sind die universalen Hub-Blöcke, die
r=2–r=6 immer selektieren (Jaccard 0.97–0.99). Ablation über alle Iterationen
aktiviert eine nichtlineare Fehlerpropagation, nicht kategoriespezifischen Schaden:

```
Ablate Kausalität-Blöcke → Schaden an Szene/Ort: +28.0 Nats (GRÖSSER als eigen!)
```

Das ist **kein** Beweis für Kompetenzzentren — es ist der kollabierte 2-Stufen-Kern.

### Architektonisch emergente Hierarchie (robuster Befund)

In allen getesteten Modellen zeigt sich spontan dieselbe Struktur:

```
r=1  → Kategorie-spezifischer Gatekeeper
       (token-individuell, wechselndes Block-Set)
↓
r=2–r=6 → Universaler rekurrenter Verarbeitungskern
           (iterativ, stabil, kategorie-unspezifisch)
```

Diese Struktur ist nicht trainiert worden — sie entsteht aus der Minimierung
des Language-Modeling-Loss. Sie entspricht genau dem Muster, das Parameter-
Streaming maximiert: stabiler residenter Kern + wechselnder Einstiegsblock.

---

## Curriculum-Diversity-Experimente

**Frage:** Kann Curriculum-Training (Warm Start von Phase 2 + graduierte Diversity)
die Qualitätslücke Phase 2 → Phase 3 schließen, ohne Spezialisierung zu opfern?

### Varianten

| Variante | Kommando | val_loss | Gini | Jaccard off-diag |
|----------|----------|----------|------|------------------|
| A (30/70) | `--diverse --diverse_until 900` | 3.127 | 0.52 | 0.899 |
| B (50/50) | `--diverse --diverse_until 1500` | — (Hang) | — | — |
| **C (r3–r=6)** | `--diverse --diverse_from_iter 2` | **3.127** | **0.13** | **0.282** |

Curriculum A und C erzielen identische Qualität (3.127 Nats), unterscheiden sich
aber erheblich in der Blockverteilung. Curriculum C (Diversity nur ab Iteration 3)
erzeugt Low-Gini (0.13) vergleichbar mit Phase 3, bei deutlich besserer Qualität.

### Curriculum B — Hang-Bug dokumentiert

Curriculum B hing nach Step 1500 für >1h (GPU idle, kein Output). Ursache:
der Übergang `diverse_until` ON→OFF bei Step 1501 nach 1500 Schritten intensivem
Diversity-Training führte zu einem nicht-terminierenden Zustand. Curriculum C
hat keinen `diverse_until`-Übergang → kein Hang.

### Vollständiger A-vs-C-Vergleich

| Metrik | Curriculum A | Curriculum C |
|--------|--------------|--------------|
| val_loss | 3.127 | **3.127** |
| Hub-Gini | 0.52 | **0.13** |
| Jaccard off-diag (Kategorien) | 0.899 | **0.282** |
| CLF bootstrap | 0.374 [0.347, 0.389] | **0.421 [0.391, 0.451]** |
| Bester CLF per Iteration | r=1: 0.404 (nur r=1) | r=2–6: 0.42 (verteilt) |
| group_r1 n=5 | +0.001% (Rauschen) | **+7.8%** |

Curriculum A kollabiert r=2–r=6 vollständiger (Jaccard zwischen Kategorien: 0.899,
4 Kategorien teilen identisches Routing). Kategorie-Signal liegt nur in r=1.
Curriculum C hält Diversity in r=3–r=6 durch: Kategorie-Signal auch im Kern.

**Mechanismus:** `diverse_until=900` (A) lässt nach Step 901 den Kollaps zurückkehren.
`diverse_from_iter=2` (C) erzwingt Diversity in r=3–r=6 ohne zeitliches Ablaufdatum.

### Haupterkenntnis

Curriculum C ist die beste getestete Variante: Qualität wie Phase 2 (3.127 Nats),
Blockverteilung wie Phase 3 (Gini 0.13), stärkste Kategorie-Trennung (Jaccard 0.282),
signifikant besser als Curriculum A bei identischer Qualität (nicht-überlappende 95%-KI).

Das Modell lernt spontan eine hierarchische Gatekeeper/Kern-Struktur (s. Exp4 oben).
Ob echte spätere Kompetenzzentren erreichbar sind, hängt vom Routing-Kollaps-Problem
bei größeren Bänken ab — offen für n_blocks=256.

---

## Phase 3 — Ergebnisse (diverse_train=ON, coord_w=0.05, 3000 Schritte)

**Implementierung:** Hard-Diversity-Training (bei r≥2: k global meistgenutzte Blöcke
aus r−1 werden gesperrt) + Koordinaten-Abstoßungsloss + trainierbare 3D-Koordinaten.

### Ergebnisse vs. Kriterien

| Metrik | Phase 2 (Baseline) | Ziel Phase 3 | Phase 3 (tatsächlich) | Status |
|---|---|---|---|---|
| Anytime-Delta L_1−L_min | 0.006 Nats | > 0.1 Nats | **0.030 Nats** | teilweise (200 Schr.: 0.183 ✓) |
| Jaccard(r→r+1) Hauptkanal | 0.984 | < 0.6 | **0.861** | teilweise (verbessert, Ziel nicht erreicht) |
| Forced-Diversity-Delta r=6 | −0.104 | > 0.0 | **+0.214** (200 Schr.) | erreicht |
| Verlust L_min | 3.705 | ≥ 3.705 | **3.891** | Regression: −5% |
| Gini | 0.62 | < 0.3 | **0.11** | deutlich übertroffen |
| Tote Blöcke | 0 / 64 | 0 / 64 | **0 / 64** | erfüllt |

### Unerwarteter Mechanismus

Phase 3 (200 Schritte) erzielte das größte Anytime-Delta (0.183 Nats) durch einen
anderen Mechanismus als geplant:
- **Erwartet:** diverse Block-Auswahl pro Iteration (verschiedene Blöcke)
- **Beobachtet:** echte iterative Verfeinerung mit denselben Blöcken an sich änderndem Zustand

Das Hard-Diversity-Training hat Blöcke gezwungen, an verschiedenen h-Zuständen nützlich
zu sein (h_r ≠ h_{r-1}). Diese Fähigkeit bleibt auch ohne Ablation bei Eval erhalten.
Forced-Diversity-Ablation bei r=6 schadet Phase-3-Modell (+0.214 Nats) statt zu helfen (−0.104).

### Identifizierter Zielkonflikt

Phase 3 liefert architektonisch Erwünschtes (Gini=0.11, Jaccard=0.861, Anytime-Kurve)
zu einem messbaren Qualitätspreis (3.891 vs. 3.705 Verlust, −5%).

**Offene Fragen für Phase 4:**
1. Ist der Qualitätspreis fundamental oder durch Curriculum reduzierbar?
2. Wie verhält sich der Mechanismus bei n_blocks=256 (H1-Skalierungstest)?
3. Fine-Tuning ohne Diversity nach Phase-3-Training als mögliche Brücke?

Der Weg zu H1 (Skalierung) und H2 (System-Benchmark) ist technisch offen,
aber die Qualitäts-Diversitäts-Abwägung muss zuerst gelöst werden.

---

## Einordnung

**Paperwürdig:** Ja.  
Tunnel-Lokalität auf natürlicher Sprache, Working-Set-Konstanz, Zwei-Phasen-Routing,
und der identifizierte Reuse-vs.-Novelty-Zielkonflikt (F7) sind originelle Beiträge
die so in der Literatur nicht dokumentiert sind.

**Industriell einsetzbar:** Noch nicht.  
H1 und H2 fehlen. Ohne Skalierungsbeweis und System-Benchmark bleibt der praktische
Nutzen spekulativ.

**Nächster Meilenstein der die Einordnung ändert:**

> Ein Modell mit mehr Parametern als verfügbarem VRAM läuft auf Consumer-Hardware mit
> brauchbaren Tokens/s und verliert gegenüber einer vollständig residenten Referenz
> weniger als ~10–15% Qualität (Perplexität oder Downstream-Aufgabe).
>
> Dann ist das keine Nice-to-Have-Architektur mehr, sondern ein potenziell disruptiver
> Inferenzmechanismus für ressourcenbeschränkte Umgebungen.

---

*Alle Experimente: `recursive-blocklm/` Repository, Ergebnisse in `results/`.*  
*Vollständige Protokolle: `EXP3_TINYSTORIES.md`, `experiments/tinystories_exp.py`.*
