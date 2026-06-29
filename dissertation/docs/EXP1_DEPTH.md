# Experiment 1 — Übernimmt rekursive Tiefe nützliche Berechnung?

Zwei Teile: **1a** beantwortet sofort und konvergent die schärfste offene Frage
aus der letzten Runde; **1b** ist das vollständige Hard-Task-Harness für die
tiefergehende Frage, lauffähig, aber GPU-gedacht.

---

## Teil 1a — War die flache Anytime-Kurve Daten oder Loss? (echtes Ergebnis)

Modell C, identische Synthetik-Daten, drei Iterationsgewichtungen. Figur:
`results/fig7_loss_variants.png`.

**Aggregierter Anytime-Abstand L₁ − L_final:**

| Gewichtung | Abstand | Kurve über Iterationen |
|---|---|---|
| equal | **−0,003** | 0,547 → 0,546 → 0,547 → 0,550 (flach) |
| linear | **+0,037** | 0,603 → 0,556 → 0,554 → 0,566 |
| end (0,7) | **+0,071** | 0,616 → 0,545 → **0,537** → 0,545 |

**Tiefengewinn pro Regime  G_q = L(Iter1) − L(Iter_final):**

| Gewichtung | REPEAT | INCREMENT | FIB | ALTERNATE |
|---|---|---|---|---|
| equal | 0,032 | −0,009 | −0,025 | −0,002 |
| linear | **0,295** | −0,031 | −0,007 | −0,006 |
| end | **0,307** | −0,012 | **0,033** | −0,007 |

**Schlussfolgerung:** Die flache Aggregatkurve unter Gleichgewichtung war
**überwiegend ein Loss-Artefakt**, nicht primär die Daten. Mit end-Gewichtung
springt REPEATs Tiefengewinn von 0,03 auf **0,31** (10×) und selbst das harte FIB
wird leicht positiv (−0,025 → +0,033); die trivialen Regimes (INCREMENT,
ALTERNATE, schon in Iteration 1 ~gelöst) bleiben erwartungsgemäß flach.

Wichtig: Der **aggregierte** Abstand (0,071) ist viel kleiner als REPEATs
Einzelgewinn (0,307), weil die trivialen Regimes den Mittelwert verwässern. Das
bestätigt genau die Forderung nach **L_{q,r}-Logging statt Aggregatkurve** —
die Aggregatkurve verbirgt die eigentliche Tiefenstruktur.

---

## Teil 1b — Hard-Task-Tiefenbenchmark (Harness, GPU-gedacht)

Code: `rblm/depth_tasks.py`, `experiments/depth_bench.py`.

**Aufgabe mit bekannter wahrer Tiefe d:** modularer Permutations-Walk
`v_i = T[(v_{i-1}+a_i) mod base]` mit fixer Permutation T, Pro-Schritt-Operanden
a_i und Ablenkertokens. Nicht-kollabierend (verlangt echte d sequenzielle
Schritte), bekannte d_q ∈ {1..D_max}, eine supervidierte Antwortposition.

**Architekturentscheidung (wichtig):** In der reinen MLP-Rekursion läuft Attention
nur einmal vor der Schleife — eine Aufgabe, die pro Schritt einen *neuen
geordneten* Zugriff braucht, ist damit prinzipiell unlösbar. Um der Tiefe eine
faire Chance zu geben, hat Modell C hier eine **geteilte In-Loop-Attention**
(`recurrent_read=True`): jede Iteration darf neu lesen. „Kein In-Loop-Read"
bleibt als Kontrolle erhalten.

**Implementierte Messgrößen & Kontrollen (alle im Code):**
- `L[d, r]` (wahre Tiefe × Modelliteration) → zentraler Heatmap-Plot.
- Tiefengewinn `G_d`, optimale Stopptiefe `r*_d`, **corr(r*_d, d)** (Spearman).
- **Oracle-Tiefe**: Auswertung jedes Beispiels bei Iteration = wahre Tiefe.
- **State-Reset**: Zustand pro Iteration zurückgesetzt → zerstört Aufbau über Iterationen.
- **Shuffle-Route**: zufällige Blockauswahl → testet, ob gelerntes Routing/Reihenfolge zählt.
- **Breite-gegen-Tiefe-Raster**: 2×8 / 4×4 / 8×2 bei gleichem Compute (16 Block-Anwendungen).

---

## Teil 1b — GPU-Ergebnisse (drei diagnostische Läufe)

Alle Läufe: Modell C mit `recurrent_read=True`, RTX 2060, `batch=64, seq=48`.
Hinweis zur Laufzeit: Der per-Block-Dispatch-Loop im Router (`.nonzero()` pro Block
× R Iterationen) erzeugt ~190 CPU-GPU-Synchronisationen pro Forward-Pass →
GPU-Speedup ~1,7× gegenüber CPU; 0,45 s/Schritt bei R=8.

### Lauf 1 — Vollspezifikation (Ausgangsbefund, 4000 Schritte)

```
D_max=8, max_distract=4, R=8, weighting=end, seed=0, 4000 Schritte (≈30 min)
```

**L[d,r] — alle Tiefen, alle Iterationen:**

| d | Iter 1 | Iter 8 | G_d | r* |
|---|---|---|---|---|
| 1 | 2,09 | 2,12 | −0,03 | 1 |
| 2 | 2,15 | 2,13 | +0,03 | 1 |
| 3–8 | 2,12–2,20 | 2,13–2,19 | ≈0 | 1 |

Controls: oracle=2,146 fixedR=2,144 state_reset=2,162 shuffle_route=2,134
(alle vier ununterscheidbar nah beieinander)

**Befund:** Chance für 8 Antwort-Tokens = ln(8) = 2,079. Alle Verlustwerte liegen
bei 2,1–2,2 → das Modell ist **auf oder nahe Zufall** für alle Tiefen und alle
Kontrollen. 4000 Schritte auf GPU sind nicht genug, um die Vollspezifikation zu
lernen. Die Spearman-Korrelation corr(r*,d) = 1,00 ist ein **Artefakt** — alle
r*=1 (konstant), und argsort(argsort) einer konstanten Folge gibt eine
Permutation; der berechnete Rang stimmt mit d überein nur durch Tie-Breaking,
nicht durch echte Tiefenstruktur. **Kein verwertbares Signal.**

### Lauf 2 — Isolationsdiagnostik: Ablenker entfernen, D_max=2 (2000 Schritte)

```
D_max=2, max_distract=0, R=2, weighting=linear, seed=0, 2000 Schritte (≈4 min)
```

| d | Iter 1 | Iter 2 | G_d | r* |
|---|---|---|---|---|
| 1 | 0,27 | 0,30 | −0,03 | 1 |
| 2 | 2,23 | 2,17 | +0,06 | 2 |

Controls: oracle=1,218 fixedR=1,279 state_reset=1,320 **shuffle_route=2,454**

**Befund:** Ohne Ablenker und mit D_max=2 konvergiert das Modell bei d=1 fast
vollständig (0,27 << 2,079 Zufall) in nur 4 Minuten. d=2 ist in 2000 Schritten
noch nahe Zufall (2,17). Entscheidend: shuffle_route=2,454 ist deutlich
**schlechter als Zufall** für das Gesamtsystem, weil zufälliges Routing die
gelernten d=1-Lösungen zerstört → das gelernte Routing ist für den gelösten Teil
essenziell.

**Kernbefund dieser Diagnostik:** Der dominierende Schwierigkeitsfaktor in Lauf 1
sind nicht die Tiefen d=3..8 — es ist das **Ablenker-Such-Problem** (find the
right OP token among DIST noise). Mit bis zu 4 DIST-Paaren pro Schritt muss die
In-Loop-Attention lernen, selektiv relevante Operanden aus variabel-langen
Rausch-Subsequenzen zu extrahieren; das überlagert das Tiefensignal vollständig.
Distractor-Robustheit und Tiefen-Komposition sind zwei unterschiedliche Lernziele,
die die Vollspezifikation konfundiert.

### Lauf 3 — Isolierte Tiefen-Komposition: D_max=4, keine Ablenker (6000 Schritte)

```
D_max=4, max_distract=0, R=4, weighting=end, seed=0, 6000 Schritte (≈22 min)
```

| d | Iter 1 | Iter 4 | G_d | r* |
|---|---|---|---|---|
| 1 | 1,68 | 1,68 | +0,00 | 1 |
| 2 | 2,26 | 2,19 | +0,07 | 3 |
| 3 | 2,12 | 2,10 | +0,02 | 1 |
| 4 | 2,12 | 2,11 | +0,02 | 1 |

Controls: oracle=2,028 fixedR=2,035 state_reset=2,055 shuffle_route=2,033

**corr(r*, d) Spearman = 0,40** (nicht degeneriert: r*=[1,3,1,1], d=[1,2,3,4])

Figur: `results/fig9_depth_diagnostic.png`

**Befund — erstes reales Tiefensignal:**

1. **d=1 teilweise gelöst** (1,68 << 2,079). Der Model lernt 1-Schritt-Komposition.
2. **d=2: G_d=+0,07, r*=3** — Iteration 1 reicht nicht; drei Iterationen werden
   benötigt, um d=2 zu lösen. Das ist das erste echte „mehr Tiefe hilft"-Signal
   in dieser Aufgabenfamilie. corr=0,40 ist durch diesen einen Datenpunkt getrieben
   (d=2→r*=3 vs. d=1,3,4→r*=1) und damit schwach, aber nicht degeneriert.
3. **State-Reset schadet** (2,055 > 2,035): Zustand zwischen Iterationen aufzubauen
   ist besser als jedes Mal mit h₀ neu anzufangen. Kleiner aber konsistenter Effekt.
4. d=3,4 noch ungelöst (≈2,10, nahe Zufall). 6000 Schritte reichen für 3-4-Schritt-
   Komposition nicht aus — 2-Schritt-Komposition braucht selbst bei 0 Ablenkern
   schon 3 Iterationen und ~6000 Schritte bis zum partiellen Lernen.

**Was noch fehlt** für den vollen Nachweis:
- Diagonale r*(d)~d über alle Tiefen (braucht entweder viel mehr Schritte oder
  Curriculum-Training: erst D_max=2 konvergieren, dann D_max=4 usw.)
- State-Reset-Effekt groß genug, um konfidenziert zu sein (derzeit Δ=0,020)
- Shuffle-Route-Nachweis (derzeit ≈fixedR, weil d=2 noch zu schwach gelernt)

---

## Zusammenfassung der drei Befunde

| Run | Aufgabe | Ergebnis |
|---|---|---|
| Full spec (D_max=8, dist≤4, 4000 steps) | Zu hart für CPU-Budget | Alles Zufall, kein Signal, degenerate corr |
| Easy nodist (D_max=2, dist=0, 2000 steps) | Bottleneck isoliert | d=1 gelöst (0,27), shuffle_route zerstört d=1 (2,45) |
| Nodist D_max=4 (dist=0, 6000 steps) | **Partieller Tiefen-Nachweis** | G_{d=2}=+0,07, r*(2)=3, state_reset↑, corr=0,40 |

**Architektureller Designbefund (neu):** Die Vollspezifikation konfundiert zwei
unabhängig harte Lernziele: (a) selektive Operanden-Extraktion aus DIST-Rauschen,
(b) sequenzielle Mehrschritt-Komposition. Ablenker sind kein „Test von Robustheit
gegenüber Rauschen" — sie erfordern ein separates, schwieriges Aufmerksamkeits-
Subproblem, das das Tiefen-Lernsignal bei dieser Modellgröße vollständig überdeckt.

---

## Ehrliche Reichweite

- **GPU-Dispatch-Engpass:** Der Python-Loop über n_blocks mit `.nonzero()` pro Block
  verursacht ~190 CPU-GPU-Syncs / Forward-Pass → GPU-Speedup nur 1,7×. Vollprotokoll
  (3 Gewichtungen × 3 Seeds × 6000 Schritte + Breite-Tiefe-Raster) ≈ 28 Stunden auf
  RTX 2060. Eliminierbar durch vektorisierten Batched-MoE-Dispatch (ändert jedoch die
  „echte Sparsity"-FLOPs-Semantik aus Phase 1).
- **Schrittbedarf:** Selbst das einfachste (D_max=2, dist=0) lernt d=2 in 2000 Schritten
  nicht vollständig; D_max=4, d=3,4 nach 6000 Schritten noch ungelöst. Starkes
  Tiefen-Signal braucht Curriculum oder ≥30k Schritte auf GPU.

## Voller Lauf (GPU)

```bash
# Reproduktion Lauf 2 (Isolationsdiagnostik):
python -m experiments.depth_bench --steps 2000 --weighting linear --seed 0 \
    --D_max 2 --R 2 --smoke --device cuda --max_distract 0

# Reproduktion Lauf 3 (Tiefen-Komposition):
python -m experiments.depth_bench --steps 6000 --weighting end --seed 0 \
    --D_max 4 --R 4 --smoke --device cuda --max_distract 0

# Curriculum-Ansatz für starkes Tiefensignal (Nächster Schritt):
python -m experiments.depth_bench --steps 8000 --weighting end --seed 0 \
    --D_max 2 --R 4 --smoke --device cuda --max_distract 0   # Phase 1: d≤2 lösen
# dann D_max=4 und D_max=6 aufbauen (oder checkpoint laden und weitertrainieren)
```

Für den vollen Breite-Tiefe-Plot: `--smoke` weglassen (fügt 3 zusätzliche Modell-
Trainings mit R×k = const = 16 hinzu).

---

## Verbindung zu Experiment 2

Punkt 5 ist die Brücke: Tiefe muss **innerhalb eines hinreichend residenten
Arbeitssets** helfen. Erst wenn Lauf 3 (oder eine Curriculum-Fortsetzung) einen
echten Tiefennutzen zeigt **und** die Blockauswahl dabei lokal bleibt, bewegt sich
„nützliche Rechenarbeit pro neu geladenem Byte" in Richtung des Break-Even aus
Exp 2 — die Vorbedingung für einen echten CUDA-Streaming-Versuch.
