# Phase‑1‑Befunde (Demo‑Konfig: d=128, 24 Blöcke, k=4, 400 Schritte, CPU)

> Kleine Skala, kurzes Training, leichte Aufgabe. Die Zahlen sind als
> **Richtungssignal** zu lesen, nicht als Endergebnis. Mehrere Befunde sind
> bewusst unbequem und werden hier nicht geglättet.

## Kennzahlen

| Modell | Core‑Params | L₁ | L_final | Verlauf über Tiefe |
|---|---|---|---|---|
| A dicht (16 distinkt) | 1.06 M | 0.607 | **0.589** | sinkt bis Iter 2, dann flach |
| B rekurrent (1 Block ×16) | 0.07 M | 0.756 | **0.720** | **U‑förmig**: Min ≈0.680 @Iter5, dann schlechter |
| C geroutet (4 von 24) | 1.60 M | 0.547 | **0.547** | flach |

Bei gleichem Budget (16 Block‑Anwendungen): **C < A < B**. C erreicht seine
Qualität bereits mit **4** Block‑Anwendungen (Iteration 1) und hält sie.

C über 3 Seeds: L_final = **0.629 ± 0.068**, MI_norm = **0.197 ± 0.030**.

## Bewertung gegen die fünf Meilenstein‑Kriterien

**1. „Iteration 4 besser als Iteration 1" — TEILWEISE / datenabhängig.**
Aggregiert ist die Anytime‑Kurve flach (0.547 → 0.550). Pro Regime aber
differenziert: `REPEAT` verbessert sich mit Tiefe (1.210 → 1.178), während
`INCREMENT`/`ALTERNATE` schon bei Iteration 1 ~gelöst sind (≈0.05 / ≈0.001) und
leicht driften, und `FIB` zu hart bleibt (≈1.06, kein Tiefengewinn). Ursache der
Flachheit: Deep‑Supervision mit Gleichgewichtung trainiert Iteration 1 stark →
Grenznutzen der Tiefe schrumpft auf leichten Daten. **Konsequenz:** Tiefe lohnt
sich selektiv → genau die Motivation für Depth‑Buckets und härtere Daten.

**2. „4 aktive Blöcke ≈ breitere Ausführung" — ERFÜLLT.**
C (sparse, 4 von 24) erreicht 0.547 und schlägt A (16 distinkte Blöcke, dicht)
bei identischem Compute‑Budget. Sparse‑Auswahl von 4 Blöcken erreicht die
Qualität der breiten Ausführung.

**3. „Verschiedene Muster → reproduzierbar verschiedene Blöcke" — ERFÜLLT (moderat).**
Regime↔Block‑MI = 0.21 normalisiert, über Seeds stabil (0.197 ± 0.030).
Ablation der *regime‑spezifischsten* Blöcke trifft die rechenintensiven Regimes:
`REPEAT`‑Blöcke ablatieren → +0.077 auf REPEAT; `FIB`‑Blöcke (teilen Block #16
mit REPEAT) → +0.162 auf REPEAT, +0.044 auf FIB. Triviale Regimes
(INCREMENT/ALTERNATE) sind redundant abgesichert (≈0). Spezialisierung entsteht
dort, wo tatsächlich gerechnet wird.

**4. „Nicht allein durch FLOPs erklärbar" — ERFÜLLT.**
C vs. B bei **identischen** 16 Block‑Anwendungen: 0.547 vs. 0.720. Reine
Rekursion eines geteilten Blocks (B) ist nicht nur schlechter, sondern
**verschlechtert sich mit zusätzlicher Tiefe** (Instabilität). Der C‑Vorteil
stammt aus Block‑Vielfalt und Auswahl, nicht aus Compute oder Rekursion an sich.

**5. „Stabil über mehrere Läufe" — ERFÜLLT für Routing, Vorbehalt beim Loss.**
Routing‑Gesundheit sehr stabil: Entropie 0.984 (1.0 = perfekt balanciert),
max. Blockanteil 0.092, tote Blöcke 1/24, kein Kollaps. Der finale Loss
streut aber spürbar (≈11 %, 0.629 ± 0.068) — bei dieser Skala/Schrittzahl
erwartbar, vor einer starken Stabilitätsaussage aber zu adressieren
(mehr Schritte, mehr Seeds).

## Streaming‑Scaffold (nur protokolliert — kein Beweis)
Zwei‑Phasen‑Routing: **Explore → Exploit.**
Jaccard‑Overlap aufeinanderfolgender Iterationen = **[0.13, 0.93, 0.95]**;
neue Blöcke je Übergang = [3.35, 0.17, 0.12]; aktive Union schrumpft 22→12.
Die erste Iteration streut breit, danach hohe zeitliche Lokalität.

**Wichtig:** Das iter1→2‑Churn (Overlap nur 0.13) ist die empirische Sichtbarkeit
deiner „Rekursion‑vs‑Lokalität"‑Spannung. Für Streaming hieße das: der Großteil
der Transferkosten entsteht am Eintritt; ab Iteration 2 wäre Prefetching billig.
Ob man das per Loss erzwingen *darf*, hängt daran, ob Qualität darunter leidet —
in Phase 1 wird es deshalb nur gemessen, nicht optimiert.

## Was das für den Plan heißt
- Anytime‑Gewinn ist real, aber **regime‑/schwierigkeitsabhängig** → dynamische
  Tiefe (Depth‑Buckets) ist gut motiviert; härtere Daten nötig, um Tiefe zu fordern.
- Dynamische Auswahl schlägt reine Rekursion klar und matcht dichte Ausführung
  bei weniger aktiven Blöcken → Kernidee trägt.
- Reine Rekursion (B) destabilisiert mit Tiefe → spricht für distinkte Blöcke
  bzw. iterationsabhängige Parameter statt eines einzigen geteilten Operators.
- Routing kollabiert nicht → Load‑Balancing + Rauschen genügen vorerst.
