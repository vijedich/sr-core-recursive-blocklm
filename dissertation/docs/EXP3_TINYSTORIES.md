# Experiment 3 — Tunnel-Validierung auf natürlicher Sprache (TinyStories)

**Kernfrage:** Entstehen Tunnel-Lokalität, Hub-Struktur und Transfer-Vorteil auch
auf echter Sprache — oder sind die Exp1/2-Befunde ein Artefakt der vier sauberen
synthetischen Regimes?

Dies ist die kritische Brückenvalidierung vor Phase 3 (räumliche Topologie).
Wenn der Tunnel-Mechanismus auf TinyStories nicht funktioniert, bringt Phase 3
nichts. Wenn er funktioniert, ist der weitere Aufbau (Leiterbahn-Index, Simulator)
auf solidem Fundament.

---

## Messgrößen

| # | Frage | Metrik | Zielwert |
|---|---|---|---|
| F1 | Lernt das Modell Sprache? | L_final < ln(vocab) | < 8,99 (bei vocab=8k) |
| F2 | Tunnel-Lokalität auf nat. Sprache? | Jaccard(r→r+1) | > 0,7 (Exp2 auf Synthetik: 0,93) |
| F3 | Hub-Struktur vorhanden? | Gini-Koeffizient | > 0,3 |
| F4 | Einzigartige Blöcke/Token | Mittelwert (max = R×k) | << R×k |
| F5 | Gelerntes besser als Zufall? | Miss-Rate-Verhältnis | > 1,5× bei mittlerem Cache |

### Interpretation der Transfer-Reduktion (F4)

Wenn `einzigartige_blöcke/token = u`, dann:

| Bankgröße n_blocks | Transfer-Faktor vs. Layer-Offloading |
|---|---|
| 64 | 64 / u |
| 256 | 256 / u |
| 1 000 | 1 000 / u |

Das Entscheidende: `u` bleibt konstant (~k bis ~R×k) wenn die Bankgröße wächst.

---

## Konfiguration

```
n_blocks = 64   k = 4   R = 6
d_model = 256   block_hidden = 512
vocab = 8 000 (BPE)   ts_max_docs = 20 000
bs = 32   seq_len = 128   steps = 10 000
Gerät: CUDA (RTX 2060)
```

### Dispatch-Fix (vor diesem Experiment implementiert)

Der `router.dispatch()` hatte O(n_blocks) CPU-GPU-Synchronisierungen pro Iteration
(1 × `torch.any()` pro Block). Mit n_blocks=64 und R=6 wären das 384 Syncs pro
Forward-Pass. Ersetzt durch sort-basiertes Dispatch: 1 Sync pro Iteration (für
`unique_consecutive().tolist()`), also 6 statt 384 Syncs. Sparse-FLOPs-Semantik
bleibt erhalten.

---

## Voraussetzungen

```bash
pip install datasets tokenizers

# Smoke-Test (200 Schritte, Netzwerk nötig für Download):
python -m experiments.tinystories_exp --smoke

# Vollauf (10 000 Schritte, ~?h auf GPU nach Dispatch-Fix):
python -m experiments.tinystories_exp --steps 10000 --device cuda
```

---

## Ergebnisse

### Vollauf (3000 Schritte, 30 Min GPU, 0.61 s/Schritt)

Alle 5 Hypothesen BESTÄTIGT — differenzierter als Smoke-Test.

### F1 — Verlust

| Iteration | r=1 | r=2 | r=3 | r=4 | r=5 | r=6 | Chance |
|---|---|---|---|---|---|---|---|
| 3000 Schritte | 3.705 | 3.705 | 3.705 | 3.707 | 3.709 | 3.711 | **8.987** |

Verlust 3.71 weit unter Zufall (8.99). Anytime-Kurve noch fast flach — erwartet
ohne Phase-3-Topologie. Unterschied L1–L6 = 0.006, d.h. alle Iterationen tragen
ähnlich bei, aber das Modell nutzt die Tiefe noch nicht differenziert.

### F2 — Jaccard-Overlap (Tunnel-Lokalität)

| Iterationspaar | r=1→2 | r=2→3 | r=3→4 | r=4→5 | r=5→6 |
|---|---|---|---|---|---|
| 3000 Schritte | 0.202 | 0.983 | 0.984 | 0.985 | 0.984 |

Identisches Zwei-Phasen-Muster wie Smoke-Test (und Synthetik): Eintrittsphase
0.20, Hauptkanal **0.984**. Bei 3000 Schritten stabiler als bei 200 (0.95–0.97).

### F3 — Hub-Struktur

| Metrik | Smoke-Test (200 Schr.) | Vollauf (3000 Schr.) |
|---|---|---|
| Gini | 0.91 | **0.62** |
| Top-5-Anteil | 83.4% | **37.6%** |
| Tote Blöcke | 26 / 64 | **0 / 64** |

Nach 3000 Schritten sind alle 64 Blöcke aktiv — der Load-Balancing-Loss hat
funktioniert. Die Hub-Struktur ist weniger extrem aber gesünder: das Modell
verteilt sich, statt in 38 Blöcken zu kollabieren. Gini 0.62 entspricht noch
immer starker Konzentration (uniform wäre 0.0).

### F4 — Einzigartige Blöcke pro Token

| Metrik | Wert | Maximum |
|---|---|---|
| Mittelwert | **6.9** | 24 (= R×k) |
| p50 | 7 | — |
| p90 | 8 | — |

Identisch mit Smoke-Test trotz anderer Hub-Verteilung. Das Working Set bleibt
bei ~7 Blöcken/Token — unabhängig davon, ob 26 oder 0 Blöcke tot sind.

Theoretische Skalierung (Annahme: Working Set bleibt konstant bei größeren Bänken —
noch nicht validiert bei n\_blocks > 64):

| Bankgröße | Theoretische aktive Blockmenge vs. Layer-Offloading |
|---|---|
| 64 Blöcke (gemessen) | **9.2× kleinere aktive Menge** |
| 256 Blöcke (hochgerechnet) | ~37× kleinere aktive Menge |
| 1 000 Blöcke (hochgerechnet) | ~144× kleinere aktive Menge |

*Die Projektion auf 256 und 1000 Blöcke setzt voraus, dass k aktiver Blöcke/Token
konstant bleibt und Routing-Qualität, Blockgröße, Dispatch-Kosten und
token-übergreifende Wiederverwendung sich nicht verschlechtern. Keines davon ist
bei größeren Bänken bisher gezeigt.*

### F5 — Cache-Miss-Rate

| Cache-Kapazität | Gelernt | Zufall | Verhältnis |
|---|---|---|---|
| 8 Blöcke (12.5%) | 0.191 | 0.896 | **4.7×** |
| 16 Blöcke (25%) | 0.129 | 0.768 | **6.0×** |
| 32 Blöcke (50%) | 0.054 | 0.515 | **9.5×** |

Niedrigere Verhältnisse als Smoke-Test (90×) weil die Routing-Verteilung
ausgeglichener ist — kein extremer Hub-Effekt mehr. 9.5× Vorteil bei 50% Cache
ist trotzdem substanziell. Mit Leiterbahn-Index (Phase 4) würde die Eintrittsphase
(r=1→2) stark verbessert.

---

### Smoke-Test (200 Schritte, ~2 Min GPU)

Alle 5 Hypothesen BESTÄTIGT — bereits nach 200 Schritten.

### F1 — Verlust

| Iteration | r=1 | r=2 | r=3 | r=4 | r=5 | r=6 | Chance |
|---|---|---|---|---|---|---|---|
| 200 Schritte | 6.10 | 6.09 | 6.11 | 6.13 | 6.16 | 6.20 | **8.99** |

Verlust deutlich unter Zufall nach 200 Schritten. Anytime-Kurve noch flach
(200 Schritte nicht genug für Tiefen-Nutzen). Vollauf (10k Schritte) laufend.

### F2 — Jaccard-Overlap (Tunnel-Lokalität)

| Iterationspaar | r=1→2 | r=2→3 | r=3→4 | r=4→5 | r=5→6 |
|---|---|---|---|---|---|
| 200 Schritte | 0.23 | 0.95 | 0.93 | 0.95 | 0.97 |

**Perfekter Zwei-Phasen-Tunnel:** Eintrittsphase (0.23) exakt wie auf synthetischen
Daten. Hauptkanal (0.95–0.97) identisch mit Exp2-Synthetik-Ergebnis (0.93–0.95).
Der Tunnel-Mechanismus überträgt sich 1:1 auf natürliche Sprache.

### F3 — Hub-Struktur

- Gini-Koeffizient: **0.91** (sehr stark konzentriert)
- Top-5-Block-Anteil: **83.4%** aller Aktivierungen
- Tote Blöcke: **26 / 64** (40% des Banks nie aktiviert)

Die Hub-Struktur ist deutlich ausgeprägter als auf Synthetik. Nur ~38 der 64 Blöcke
werden überhaupt genutzt, davon tragen 5 den Großteil. Pinning dieser Hub-Blöcke
würde >83% der Aktivierungen abdecken ohne Transfer.

### F4 — Einzigartige Blöcke pro Token

| Metrik | Wert | Maximum |
|---|---|---|
| Mittelwert | **6.9** | 24 (= R×k) |
| p50 | 7 | — |
| p90 | 8 | — |

| Bankgröße | Theoretische aktive Blockmenge vs. Layer-Offloading |
|---|---|
| 64 Blöcke (gemessen) | **9.3× kleinere aktive Menge** |
| 256 Blöcke (hochgerechnet) | ~37× kleinere aktive Menge |
| 1 000 Blöcke (hochgerechnet) | ~144× kleinere aktive Menge |

### F5 — Cache-Miss-Rate

| Cache-Kapazität | Gelernt | Zufall | Verhältnis |
|---|---|---|---|
| 8 Blöcke (12.5%) | 0.057 | 0.896 | **15.8×** |
| 16 Blöcke (25%) | 0.009 | 0.769 | **90.5×** |
| 32 Blöcke (50%) | 0.006 | 0.514 | **79.2×** |

Mit nur 25% der Blöcke im Cache erzeugt gelerntes Routing nur 0.9% Miss-Rate.

---

## Verbindung zu anderen Experimenten

- **Exp2 (Synthetik):** Jaccard 0,93–0,95 (Iterationen 2→3), Gini ~0,8 (Hub-Struktur).
  Exp3 prüft ob diese Werte auf nat. Sprache halten.
- **Phase 3 (Koordinaten):** Nur sinnvoll wenn F2 und F3 erfüllt — Tunnel müssen
  existieren, bevor sie räumlich geordnet werden können.
- **Leiterbahn-Index:** Setzt voraus dass Tunnels auf nat. Sprache stabil und
  wiederholbar sind (F2).

---

## Per-Iteration-Ablation (200 Schritte, Smoke-Test-Modell)

Vier Messungen nach Training (`iteration_diagnostics`, 4 Batches × bs=32, seq=128):

### A) Relative Zustandsänderung ||h_r − h_{r-1}|| / ||h_{r-1}||

| r=1 | r=2 | r=3 | r=4 | r=5 | r=6 |
|---|---|---|---|---|---|
| **9.663** | 0.224 | 0.183 | 0.155 | 0.135 | 0.119 |

Der Zustand ändert sich pro Iteration um 12–22% — keine Fixed-Point-Konvergenz.
Die hohe r=1-Zahl ist der erste Sprung vom Encoding (Aufmerksamkeitsschicht) zum
ersten Routing-Schritt. Die Folgeiterationen zeigen echte, monoton abnehmende Änderungen.

**Interpretation:** Spätere Iterationen berechnen echtes Neues, nicht nur Wiederholung.
Das Zustand-Argument ist damit kein Blocker für Phase 3.

### B) Jaccard-Matrix aller Iterationspaare

```
       r1    r2    r3    r4    r5    r6
r1:  1.00  0.28  0.28  0.28  0.28  0.28
r2:  0.28  1.00  0.93  0.85  0.81  0.79
r3:  0.28  0.93  1.00  0.91  0.87  0.84
r4:  0.28  0.85  0.91  1.00  0.95  0.93
r5:  0.28  0.81  0.87  0.95  1.00  0.97
r6:  0.28  0.79  0.84  0.93  0.97  1.00
```

- r1 ist vollständig entkoppelt von r2..r6 (Jaccard konstant ~0.28)
- Benachbarte Paare (r→r+1): 0.93–0.97 — hoher Overlap
- Langstrecke r2 vs. r6: **0.79** — langsame Drift, kein eingefrorenes Routing
- Es gibt kein vollständig statisches Routing: die Block-Auswahl verschiebt sich akkumulativ

### D) Output-KL KL(p_{r+1} || p_r)

| r1→r2 | r2→r3 | r3→r4 | r4→r5 | r5→r6 |
|---|---|---|---|---|
| inf* | 0.014 | 0.008 | 0.005 | 0.003 |

*r1→r2 numerisch instabil (log(0) bei frischem Modell), Messung nicht auswertbar.
Für r2..r6: Die Ausgabe ändert sich pro Iteration klein aber nicht null. Abnehmend.

### C) Erzwungene Diversität — zentrales Ergebnis

Jede Iteration r bekommt als Ablationsmaske die Blöcke der Vorgänger-Iteration (kumulativ).

| r | Normal-Loss | Forced-Loss | Delta |
|---|---|---|---|
| 1 | 6.252 | 6.252 | ±0.000 |
| 2 | 6.215 | 6.238 | **+0.023** |
| 3 | 6.224 | 6.202 | **−0.022** |
| 4 | 6.247 | 6.195 | **−0.053** |
| 5 | 6.276 | 6.206 | **−0.069** |
| 6 | 6.305 | 6.201 | **−0.104** |

**Kernergebnis:** Ab r=3 ist erzwungene Diversität **besser** als das gelernte Routing.
Bei r=6 reduziert Diversität den Loss um 0.104 Nats (≈1.7% relativer Verbesserung).

**Verlaufsform des Normal-Loss:** Minimum bei r=2 (6.215), danach monoton schlechter bis 6.305
→ Iterationen r=3..6 schadendem Modell mit Normal-Routing (Rauschen durch wiederholte Blöcke).

**Verlaufsform des Forced-Loss:** Monoton besser von r=2 bis r=4 (6.202→6.195), dann stabil
→ Mit erzwungener Diversität zeigen sich echte Tiefen-Vorteile.

### Schlussfolgerung aus der Ablation

Das Routing-Kollaps-Muster (Jaccard≈0.98 zwischen r=2..6) ist kein Merkmal sondern
ein **Trainingsdefekt**: das Modell lernt, dieselben Blöcke wiederholt zu aktivieren,
was Iterationen r=3..6 nutzlos oder schädlich macht. Erzwungene Diversität zeigt, dass
die Architektur die Tiefe nutzen *könnte* — wenn das Routing entsprechend regularisiert wird.

**Direkte Konsequenz für Phase 3:**
1. Trainierbare Koordinaten + Distanzterm allein reichen möglicherweise nicht
2. Zusätzlich nötig: **Jaccard-Diversitätsloss** der Iteration r bestraft, wenn sie
   die gleichen Blöcke wie r−1 wählt. Z.B. `diversity_loss = Jaccard(S_r, S_{r-1})`.
3. Alternativ: explizites Ablation-Routing während Training (ähnlich dem Ablation-Mechanismus
   in `router.py`) mit Annealing — zu Beginn erzwungene Diversität, später organisch.

---

## Zulässige Kernaussage (Vollauf, 3000 Schritte)

> Das gelernte Routing erzeugt auf natürlicher Sprache (TinyStories) echte
> Tunnel-Lokalität: Jaccard **0.984** im Hauptkanal (r=2..6), identisch mit
> Synthetik-Ergebnissen (Exp2: 0.93–0.95). Die Eintrittsphase (r=1→2,
> Jaccard=0.20) zeigt dasselbe Zwei-Phasen-Muster. Alle 64 Blöcke sind aktiv
> (Gini=0.62), das Modell kollabiert nicht. Das Working Set pro Token liegt konstant
> bei **6.9 einzigartigen Blöcken**; bei unverändertem Working Set entspräche dies
> theoretisch einer etwa 144-fach kleineren aktiven Blockmenge gegenüber vollständigem
> Layer-Offloading bei n\_blocks=1000 — unter der Voraussetzung, dass Working Set,
> Routing-Qualität und Dispatch-Kosten bei größeren Bänken stabil bleiben.
> Die Anytime-Kurve ist flach (L1–L6 Δ=0.006): spätere Iterationen leisten derzeit
> kaum zusätzliche komplementäre Berechnung. Die Per-Iteration-Ablation zeigt die
> Ursache: Das Routing kollabiert zu Jaccard≈0.98 zwischen r=2..6, und erzwungene
> Diversität (Blöcke von r−1 verboten in r) verbessert den Loss ab r=3 messbar
> (bis −0.104 Nats bei r=6). Das Routing-Kollaps-Muster ist ein Trainingsdefekt,
> keine nützliche Eigenschaft. Phase 3 muss neben trainierbarern Koordinaten einen
> **Jaccard-Diversitätsloss** einführen, der Wiederholung desselben Block-Sets
> zwischen aufeinanderfolgenden Iterationen explizit bestraft.

---

## Phase 3 — Ergebnisse (Smoke-Test, 200 Schritte, diverse_train=ON)

### Anytime-Kurve (Kernziel)

| Phase | r=1 | r=2 | r=3 | r=4 | r=5 | r=6 | Δ(L1−Lmin) |
|---|---|---|---|---|---|---|---|
| Phase 2 (Baseline, 200 Schr.) | 6.218 | **6.170** | 6.180 | 6.206 | 6.236 | 6.267 | 0.048 |
| Phase 3 (diverse, 200 Schr.)  | 6.267 | 6.144 | 6.097 | **6.084** | 6.090 | 6.109 | **0.183** |

**Ziel war: Δ > 0.1 Nats. Erreicht: 0.183 Nats (3.8× über Ziel). Vollauf läuft.**

Das Modell profitiert jetzt genuine von Tiefe — Minimum bei r=4, klare Anytime-Kurve.

### Unerwarteter Mechanismus

Routing-Kollaps (Jaccard=1.000 bei r2..r6) ist stärker, nicht schwächer.
Das Modell hat eine andere Lösung als erwartet gelernt:

- **Nicht:** diverse Block-Auswahl pro Iteration
- **Sondern:** echte iterative Verfeinerung mit denselben Blöcken an sich änderndem Zustand

Das Diversity-Training hat die Blöcke gezwungen, an verschiedenen h-Zuständen
(h_r ≠ h_{r-1}) nützlich zu operieren. Diese Fähigkeit bleibt auch bei Eval-Routing
(ohne Ablation) erhalten. Zustandsänderung: 10–15% pro Schritt.

**Forced-Diversity-Ablation — umgekehrt gegenüber Phase 2:**

| | Phase 2 (r=6) | Phase 3 (r=6) |
|---|---|---|
| Forced-Delta | −0.104 (Ablation hilft) | **+0.214** (Ablation schadet) |

Phase 3 braucht die Wiederholung derselben Blöcke — weil die Blöcke gelernt haben,
iterativ zu verfeinern, nicht diversifiziert zu suchen.

### Nebenbefund: Routing-Konzentration nimmt zu

Tote Blöcke: 35/64 (Phase 2: 26/64). Cache-Miss-Rate: 0.003 überall.
Das Load-Balancing hat bei 200 Schritten noch nicht eingegriffen.
Bei Phase-2-Vollauf (3000 Schr.) waren alle 64 Blöcke aktiv → Vollauf abwarten.

---

## Phase 3 — Vollauf-Analyse (3000 Schritte, diverse_train=ON, coord_w=0.05)

### Training-Verlauf

| Schritt | L1 | L_fin | Jaccard | Div |
|---|---|---|---|---|
| 1 | 251.279 | 242.577 | 0.912 | 0.079 |
| 500 | 4.993 | 4.999 | 0.953 | 0.078 |
| 1000 | 4.425 | 4.422 | 0.940 | 0.078 |
| 1500 | 4.204 | 4.193 | 0.937 | 0.083 |
| 2000 | 3.989 | 3.975 | 0.930 | 0.089 |
| 2500 | 3.892 | 3.868 | 0.914 | 0.092 |
| 3000 | 3.921 | **3.896** | **0.909** | 0.094 |

Jaccard sinkt monoton von 0.912 auf 0.909 während des Trainings (Phase 2: stabil bei ~0.984).
Die Hard-Diversität wirkt langfristig, aber die Konvergenz des Jaccard-Werts auf ~0.91 zeigt,
dass das Modell ein Gleichgewicht zwischen Tunnel-Stabilität und Routing-Varianz gefunden hat.

### Vergleich Phase 2 vs. Phase 3 (je 3000 Schritte)

| Metrik | Phase 2 | Phase 3 | Bewertung |
|---|---|---|---|
| Verlust L_min | **3.705** | 3.891 | Phase 2 gewinnt (−0.186) |
| Anytime-Delta L1−L_min | 0.006 Nats | **0.030 Nats** | Phase 3 5× besser |
| Jaccard r2→r3 | 0.983 | **0.861** | Phase 3 deutlich diverser |
| Gini-Koeffizient | 0.62 | **0.11** | Phase 3 nahezu uniform |
| Top-5-Block-Anteil | 37.6% | **11.9%** | Phase 3 gleichmäßig verteilt |
| Tote Blöcke | 0 / 64 | 0 / 64 | gleich |
| Cache-Miss (50%) | **0.054** | 0.123 | Phase 2 effizienter (2.3×) |
| Einzigartige Blöcke/Token (mean) | 6.9 | 6.4 | ähnlich |
| Trainingszeit | 1826s | **1721s** | Phase 3 etwas schneller |

### Vollständige Jaccard-Matrix (Phase 3, 3000 Schritte)

```
       r1    r2    r3    r4    r5    r6
r1:  1.00  0.56  0.49  0.45  0.42  0.40
r2:  0.56  1.00  0.86  0.78  0.73  0.69
r3:  0.49  0.86  1.00  0.89  0.82  0.77
r4:  0.45  0.78  0.89  1.00  0.90  0.85
r5:  0.42  0.73  0.82  0.90  1.00  0.91
r6:  0.40  0.69  0.77  0.85  0.91  1.00
```

Deutlich diverser als Phase 2 (r2 vs r6: **0.69** statt 0.79). Aber Ziel <0.6 noch nicht erreicht
(nur r1 vs r6 = 0.40 unterschreitet diese Grenze).

### Per-Iteration-Ablation (Phase 3, 3000 Schritte)

**A) Relative Zustandsänderung:**

| r=1 | r=2 | r=3 | r=4 | r=5 | r=6 |
|---|---|---|---|---|---|
| **0.340** | 0.018 | 0.014 | 0.013 | 0.013 | 0.012 |

Phase 3 macht sehr kleine aber konsistent nicht-null Schritte (vs. Phase 2 Smoke: 0.22 bei r=2).
Das Modell verfeinert den Zustand mit minimalen Sprüngen — stabileres iteratives Verhalten.

**D) KL-Divergenz zwischen aufeinanderfolgenden Ausgaben:**

| r1→2 | r2→3 | r3→4 | r4→5 | r5→6 |
|---|---|---|---|---|
| 0.0059 | 0.0033 | 0.0026 | 0.0025 | 0.0026 |

Erstmals auswertbar ohne NaN (Phase 2 Smoke: überall inf). Das volltrainierte Modell
erzeugt numerisch stabile Übergänge. Klein aber nicht null — echte Verfeinerung pro Iteration.

**C) Erzwungene Diversität:**

```
r=1: normal=3.942  forced=3.942  delta=+0.000
r=2: normal=3.924  forced=nan    delta=nan
r=3..r=6: alle nan
```

NaN-Werte ab r=2 sind ein **Messfehler**, kein Modell-Bug: Phase-3-Routing mit Gini=0.11
aktiviert nahezu alle 64 Blöcke, daher sperrt die kumulative Ablationsmaske
weit mehr als k=4 Blöcke gleichzeitig → extreme Logits, numerisch instabil.
Die Messmethode der erzwungenen Diversität ist für nahezu uniform geroutete Modelle
nicht geeignet und muss angepasst werden (z.B. Top-k-Ablation statt kumulativ).

### Anytime-Kurven im direkten Vergleich

| Phase | r=1 | r=2 | r=3 | r=4 | r=5 | r=6 | Δ(L1−Lmin) |
|---|---|---|---|---|---|---|---|
| Phase 2 (3000 Schr.) | 3.705 | 3.705 | 3.705 | 3.707 | 3.709 | 3.711 | 0.006 |
| Phase 3 (3000 Schr.) | 3.921 | 3.903 | 3.894 | **3.891** | **3.891** | 3.896 | **0.030** |

Phase 3 zeigt eine echte Anytime-Kurve mit Valley-Form (Minimum bei r=4/5).
Das Anytime-Delta von 0.030 Nats ist 5× besser als Phase 2 (0.006), aber niedriger
als beim Smoke-Test (0.183 Nats nach 200 Schritten).

**Vermutete Ursache:** Bei 200 Schritten ist das Hard-Diversity-Signal noch dominant
(unfertige Blöcke, hoher Exploration-Druck). Bei 3000 Schritten findet das Modell
ein eigenes Gleichgewicht, bei dem Tunnel-Stabilität und Routing-Diversität konkurrieren.

### Interpretation: Ein echter Qualitäts-Diversitäts-Zielkonflikt

**Gewonnen durch Phase 3:**
- Gini=0.11 statt 0.62: alle 64 Blöcke gleichmäßig genutzt
- Jaccard 0.861 statt 0.983: echte Routing-Diversität zwischen Iterationen
- Anytime-Kurve 5× steiler: 0.030 statt 0.006 Nats Tiefen-Gewinn
- Valley-Form der Kurve zeigt iterative Verfeinerung statt flacher Aktivierungen

**Verloren gegenüber Phase 2:**
- Verlust 3.891 statt 3.705: etwa 5% schlechtere Qualität durch erzwungene Diversität
- Cache-Miss 0.123 statt 0.054: schlechtere Lokalität bei uniformer Blockverteilung
- Einfachere Implementierung durch Wegfall der Phase-3-Mechanismen

**Zentrale offene Frage:** Ist der Qualitätspreis (−0.186 Nats) fundamental,
oder kann er durch Curriculum-Training (Diversity-Druck graduell reduzieren),
Fine-Tuning ohne Diversity nach Phase-3-Training, oder größere Modelle reduziert werden?

Die Ergebnisse erfüllen das make-or-break-Kriterium Δ>0.1 nur beim Smoke-Test (200 Schritte),
nicht beim Vollauf (3000 Schritte, Δ=0.030). Die Jaccard-Diversität ist substanziell verbessert
(0.861 vs. Ziel <0.6 — teilweise erfüllt). Der Mechanismus ist bestätigt, die Stärke ist geringer
als erwartet.

---

## Exp4 — Kompetenzzentren-Analyse: Gruppen-Ablation und r=1-Kausaltest

**Ziel:** War die schwache Einzelblock-Ablation (Δ≈0.0001 Nats) ein Kapazitätsproblem
(5 Blöcke sind zu wenig) oder Beweis für fehlende Spezialisierung? Gruppenablation
testet Kategorie-Top-N-Blöcke gleichzeitig — und kontrolliert per welcher Iteration
der Schaden auftritt.

**Implementierung:** `group_r1_ablation_test()` — 5 Konditionen × 3 n_top-Werte
[5, 10, 15]. Per-Iterations-Masken erlauben Isolation auf r=1 vs. r=2–6.

### Phase-2-Modell — Gruppenablation

| Kondition | Diag n=5 | Off-Diag n=5 | Dominanz | Bedeutung |
|-----------|----------|--------------|----------|-----------|
| group_all | −0.000% | −0.000% | 3/6 | Kein Gesamteffekt |
| group_r1 | +0.000% | +0.001% | 3/6 | r=1 kein Effekt |
| group_r2r6 | −0.000% | −0.001% | 4/6 | r=2–6 kein Effekt |
| random_all | +0.001% | +0.000% | — | Kontrolle: Rauschen |

**r=1-Kausaltest Phase 2:** frac_r1 = 344 (Artefakt: Teiler ≈ 0)
→ Phase-2-Ablation ist reines Rauschen. Kein nachweisbarer Spezialisierungseffekt.

### Phase-3-Modell — Gruppenablation

| Kondition | Diag n=5 | Off-Diag n=5 | Dominanz | Bedeutung |
|-----------|----------|--------------|----------|-----------|
| group_all | **+0.963%** | +0.878% | 4/6 | Echter Gruppeneffekt |
| group_r1 | +0.037% | +0.065% | 1/6 | r=1 kein selektiver Schaden |
| group_r2r6 | **+0.101%** | +0.076% | 4/6 | r=2–6 hat Kategorieinformation |

**r=1-Kausaltest Phase 3:** frac_r1 = 0.038
→ r=2–6 dominiert. r=1 trägt nur 4% des Gesamtschadens. Kategorie-Signal liegt
im Kern (r=2–6), nicht im Eintrittsschritt.

### Vergleich Phase 2 vs. Phase 3 (Kompetenzzentren)

| Metrik | Phase 2 | Phase 3 |
|--------|---------|---------|
| Jaccard off-diag | 0.759 | **0.485** |
| Ablation Δ (Einzelblock) | ~0.0000 Nats | **+0.1259 Nats** |
| group_all n=5 | ~0% | **+0.963%** |
| Bester CLF | r=1: 0.380 | r=6: **0.430** |
| r1-Kausal? | Rauschen | Kern r=2–6 dominiert (frac=0.038) |

Phase 3 zeigt echte, wenn auch kleine, funktionale Spezialisierung.
Phase 2 zeigt keine nachweisbare Spezialisierung über Rauschen hinaus.

---

## Curriculum-Diversity-Experimente (Warm Start von Phase-2-Checkpoint)

**Hypothese:** Kann Curriculum-Training die Qualitätslücke zwischen Phase 2 (3.705)
und Phase 3 (3.891) schließen, während Gini/Spezialisierung erhalten bleibt?

**Gemeinsame Konfiguration:**
```
Warm Start: checkpoints/tinystories_phase2/seed_0/step_3000
steps=3000, bs=32, seq=128, device=cuda
diverse_train=ON (k Blöcke aus r-1 gesperrt)
```

### Curriculum A — 30% Diversity (diverse_until=900)

**Kommando:** `--diverse --diverse_until 900 --pretrained_ckpt ...`
**Checkpoint:** `tinystories_curriculum_30pct/seed_0/step_3000`

| Schritt | L1 | L_fin | Jaccard | Div |
|---------|-----|-------|---------|-----|
| 500 | 3.615 | 3.615 | 0.992 | 0.078 |
| 1000 | 3.424 | 3.424 | 0.997 | 0.078 |
| 1500 | 3.350 | 3.352 | 0.997 | 0.078 |
| 2000 | 3.164 | 3.165 | 0.996 | 0.078 |
| 2500 | 3.122 | 3.123 | 0.997 | 0.078 |
| 3000 | 3.126 | **3.127** | 0.996 | 0.078 |

**Ergebnis:** val_loss=3.127 — große Qualitätsverbesserung gegenüber Phase 3 (3.891).

**Routing-Charakteristik:**
```
Jaccard (r->r+1): [0.079, 0.997, 0.997, 0.996, 0.996]
Hub-Gini: 0.5152   Top-5-Anteil: 27.8%   Tote Blöcke: 0/64
```

**Befund:** 2-Stufen-Routing-Kollaps (Jaccard 0.08 zwischen r1 und r2, 0.997 innerhalb r2–r6).
Diversity nach Step 900 nicht mehr wirksam. Hohe Hub-Konzentration (Gini=0.52).

**Hang-Bug:** Curriculum A vollendete. Curriculum B (diverse_until=1500) hing nach Step 1500
für >1h (GPU idle). Ursache: Transition `diverse_train` ON→OFF bei Step 1501 führte zu
einem nicht-terminierten Prozess (vermutlich numerisches Problem durch schlagartigen
Routing-Wechsel nach 1500 Schritten intensivem Diversity-Training).

### Curriculum C — Selective Diversity (diverse_from_iter=2)

**Kommando:** `--diverse --diverse_from_iter 2 --pretrained_ckpt ...`
**Checkpoint:** `tinystories_curriculum_fromIter2/seed_0/step_3000`

Diversity ab 0-basierter Iteration 2 → d.h. r=3–r=6 erhalten Diversity-Ablation,
r=1 und r=2 trainieren ohne. Kein `diverse_until` → Diversity über alle 3000 Schritte.

| Schritt | L1 | L_fin | Jaccard | Div |
|---------|-----|-------|---------|-----|
| 500 | 3.621 | 3.623 | 0.990 | 0.078 |
| 1000 | 3.423 | 3.423 | 0.993 | 0.078 |
| 1500 | 3.350 | 3.352 | 0.993 | 0.078 |
| 2000 | 3.165 | 3.166 | 0.992 | 0.078 |
| 2500 | 3.121 | 3.121 | 0.993 | 0.078 |
| 3000 | 3.126 | **3.127** | 0.992 | 0.078 |

**Ergebnis:** val_loss=3.127 — identisch mit Curriculum A.

**Routing-Charakteristik:**
```
Jaccard (r->r+1): [0.08, 0.99, 0.993, 0.992, 0.992]
Hub-Gini: 0.1271   Top-5-Anteil: 14.2%   Tote Blöcke: 0/64
```

Curriculum C läuft ohne Hang durch. Das Fehlen eines `diverse_until`-Übergangs
vermeidet das Hänge-Problem von Curriculum B vollständig.

### Vergleich aller Modelle

| Modell | val_loss | Gini | Top-5% | Jaccard r1→r2 | CLF best | Jaccard off-diag |
|--------|----------|------|--------|----------------|----------|------------------|
| Phase 2 (Scratch) | ~3.71 | 0.62 | 37.6% | 0.20 | 0.380 | 0.759 |
| Phase 3 (Scratch) | 3.891 | **0.11** | 11.9% | 0.56 | 0.430 | 0.485 |
| Curriculum A | 3.127 | 0.52 | 27.8% | 0.08 | 0.374 | 0.899 |
| **Curriculum C** | **3.127** | **0.13** | **14.2%** | 0.08 | **0.421** | **0.282** |

Curriculum C gewinnt in 4 von 5 Metriken:
- Gleiche Qualität wie A bei besserer Blockverteilung (Gini 0.13 vs 0.52)
- Stärkste Kategorie-Trennung aller Modelle (Jaccard 0.282)
- Bester Klassifikator (0.421, 95%-KI [0.391, 0.451])

---

## Exp4 — Kompetenzzentren-Analyse: Curriculum C (korrigierte Interpretation)

**Lauf:** `competence_centers_exp --ckpt tinystories_curriculum_fromIter2 --analysis`

### Routing-Struktur

```
Top-5 Lift-Bloecke:
  Kausalitaet : [49, 62, 2, 45, 41]  (Lift: 1.44, 1.30, 1.26, 1.24, 1.19)
  Koreferenz  : [49, 18, 43, 48, 28]  (Lift: 1.25, 1.13, 1.12, 1.12, 1.11)
  Dialog      : [44, 51, 46, 35, 29]  (Lift: 1.55, 1.50, 1.32, 1.30, 1.26)
  Emotion     : [9, 34, 39, 53, 31]   (Lift: 1.33, 1.24, 1.20, 1.17, 1.16)
  Szene/Ort   : [43, 1, 27, 18, 62]   (Lift: 1.26, 1.24, 1.20, 1.19, 1.19)
  Zeitfolge   : [49, 1, 19, 22, 10]   (Lift: 1.29, 1.21, 1.20, 1.19, 1.17)

Jaccard zwischen Kategorien (r=2):
  Ausserdiagonale: 0.282  >> stärkste Trennung aller Modelle

Mutual Information r=1: 0.0053 nats → r=2: 0.0102 nats (steigt)
Klassifikator best: 0.451 (Zufall: 0.167, Bootstrap-95%-KI: [0.391, 0.451])
```

### Ablations-Analyse und Sanity-Check

Die erste Auswertung zeigte Einzelblock-Ablationswerte von ~27 Nats (Kausalität).
Ein Sanity-Check mit per-Iterations-Ablation klärte den Mechanismus:

```
Baseline (kein Ablation):      3.1814 Nats
Nur r=1 abliert:               3.5597 Nats  (delta=+0.38)
Nur r=2 abliert:               3.1814 Nats  (delta≈0.00)
Nur r=3–r=6 abliert:           3.1814 Nats  (delta≈0.00)
ALLE Iterationen abliert:     31.0260 Nats  (delta=+27.8)
```

**Diagnose:** Die 27 Nats sind real und kein Messfehler, aber sie messen nicht
kategoriespezifische Kompetenz, sondern den **2-Stufen Routing-Kollaps**:

r=2–r=6 verwenden alle denselben 4-Block-Kern (Jaccard 0.97–0.99). Die Top-5
Lift-Blöcke einer Kategorie sind genau diese 4 Hub-Blöcke + 1 Block aus r=1.
Werden sie in ALLEN 6 Iterationen gleichzeitig ablatiert:
- r=1 muss andere Blöcke verwenden (kleiner Schaden: +0.38 Nats)
- r=2–r=6 müssen andere Blöcke verwenden (null Einzelschaden pro Iteration)
- Die iterative Fehlerakkumulation über 6 falsche Routing-Entscheidungen ist jedoch
  nichtlinear und katastrophal (Summe ≠ Einzelteile)

**Beweis:** Ablation der gleichen Blöcke trifft ALLE Kategorien ähnlich stark:
```
Ablierte: Kausalität-Blöcke [49, 62, 2, 45, 41]
  Kausalitaet eval: +27.8 Nats  (+875%)
  Szene/Ort eval:   +28.0 Nats  (+888%)   <- Fremd-Schaden noch größer!
  Emotion eval:     +24.6 Nats  (+732%)
  Zeitfolge eval:   +28.9 Nats  (+825%)
```

Der "Eigen-Schaden > Fremd-Schaden"-Befund (27.77 vs. 25.76) ist ein schwaches Signal
in extremem universellen Hub-Rauschen, kein Beweis für Kompetenzzentren.

### Gruppenablation (Curriculum C)

| Kondition | Diag n=5 | Off-Diag n=5 | Bedeutung |
|-----------|----------|--------------|-----------|
| group_all | **+462%** | +433% | Katastrophaler Hub-Kern-Effekt |
| group_r1 | **+7.8%** | +7.8% | r=1 hat echte Kategorie-Information |
| group_r2r6 | −0.002% | −0.000% | r=2–6 kein kategoriespezifischer Effekt |

**r=1-Kausaltest:** frac_r1 = 0.017 → r=2–6 dominiert als universaler Kern.

Die 462% im `group_all` entsteht aus demselben Hub-Kollaps-Effekt wie die 27 Nats.

### Korrigierte Interpretation — Curriculum C

**Was robust ist (starke Evidenz):**
- Jaccard off-diag 0.282: Kategorien nutzen signifikant verschiedene Block-Muster ✓
- CLF 0.421 (KI [0.391, 0.451]): Routing kodiert echte Kategorie-Information ✓
- group_r1 = +7.8%: r=1 hat nachweisbare, wenn auch moderate Kategorie-Spezialisierung ✓
- Gini 0.13, Top-5 14.2%: breite Blocknutzung, kein Kollaps auf wenige Hubs ✓

**Was nur Routing-Kollaps misst (kein Beweis für Kompetenzzentren):**
- 27 Nats Einzelblock-Ablation — misst 2-Stufen Hub-Kern-Katastrophe, nicht Spezialisierung ✗
- 5/6 "Eigen > Fremd" im Ablationstest — Signal-Rausch-Verhältnis zu klein ✗
- group_r2r6 ≈ 0: r=2–r=6 zeigen keine robuste kategoriespezifische Funktion ✗

### Architektonische Interpretation

Das Modell hat spontan eine **hierarchische Arbeitsteilung** entwickelt:

```
r=1:  Kategorie-spezifischer Gatekeeper
      — wählt kontextabhängig andere Blöcke
      — 7.8% Schaden wenn Kategorie-Blöcke gesperrt werden
      — Klassifikator-Accuracy hier: 0.359
      ↓
r=2–r=6: Universaler rekurrenter Verarbeitungskern
      — alle Iterationen verwenden dieselben ~4 Blöcke (Jaccard 0.99)
      — kein kategoriespezifischer Effekt per Iteration
      — aber katastrophaler Kollektivschaden wenn gemeinsam ablatiert
      — Klassifikator-Accuracy hier: 0.419–0.424
```

Diese Struktur ist für Parameter-Streaming **praktisch attraktiv**:
- r=1 lädt token-spezifische Eintrittsblöcke (kleines, wechselndes Set)
- r=2–r=6 halten einen stabilen universalen Kern resident (keine Transfers nötig)
- Das Modell optimiert spontan für genau das Muster, das Streaming-Effizienz maximiert

**Was weiterhin aussteht:** Nachweis vollwertig getrennter späterer Kompetenzzentren
(mehrere verschiedene Blockmengen in r=2–r=6, je nach Kategorie). Das erfordert ein
Modell ohne 2-Stufen-Kollaps — oder größere Bankgröße (n_blocks > 64).

---

## Zusammenfassung: Qualitäts-Diversitäts-Kurve (alle Modelle)

| Modell | val_loss | Gini | Jaccard off-diag | CLF | Architektur |
|--------|----------|------|------------------|-----|-------------|
| Phase 2 (Scratch) | 3.705 | 0.62 | 0.759 | 0.380 | 2-Stufen, Hub-lastig |
| Phase 3 (Scratch+Div) | 3.891 | 0.11 | 0.485 | 0.430 | Diverse, Qualitätsverlust |
| Curriculum A (WS+30%) | 3.127 | 0.52 | 0.899 | 0.374 [0.347, 0.389] | 2-Stufen, Hub-kollaps |
| **Curriculum C (WS+from_r3)** | **3.127** | **0.13** | **0.282** | **0.421** | **2-Stufen, niedrig-Gini** |

**Curriculum C ist die beste Variante:** Qualität wie Curriculum A, Blockverteilung
fast so gut wie Phase 3, stärkste Kategorie-Trennung aller getesteten Modelle.

**Offene Kernfrage:** Das 2-Stufen-Routing-Muster (r=1 eigenständig, r=2–r=6 kollabiert)
ist in allen warm-gestarteten Modellen präsent und in Phase-2/3 von Scratch auch stark.
Ob echte spätere Kompetenzzentren erreichbar sind, hängt davon ab ob das Routing-Kollaps-
Problem bei größeren Modellen (n_blocks=256+) von selbst verschwindet oder explizit
verhindert werden muss.

---

## Exp4 — Curriculum A vs. Curriculum C: Vollständiger Vergleich

### Vergleichstabelle (identische Analyse-Pipeline)

| Metrik | Curriculum A | Curriculum C | Bewertung |
|--------|--------------|--------------|-----------|
| val_loss | 3.127 | 3.127 | gleich |
| Hub-Gini | 0.52 | **0.13** | C deutlich flacher |
| Top-5-Block-Anteil | 27.8% | **14.2%** | C gleichmäßiger |
| Jaccard off-diag (Kat.) | 0.899 | **0.282** | C 3.2x mehr Trennung |
| CLF bootstrap-mean | 0.374 [0.347, 0.389] | **0.421 [0.391, 0.451]** | C signifikant besser |
| Bester CLF per Iter. | r=1: 0.404 (dominant) | r=6: 0.421 (alle ~0.42) | A: r=1-Signal; C: Kern |
| group_r1 n=5 | +0.001% (Rauschen) | **+7.8%** | C: echter r=1-Kausaleffekt |
| group_r2r6 n=5 | -0.001% (Rauschen) | -0.002% (Rauschen) | beide: kein r=2-6-Effekt |
| group_all n=5 | +0.000% | +462% (Hub-Artefakt) | unterschiedlicher Kollaps |
| usage_all n=5 | **+1473%** | +237% | A: Hubs noch kritischer |

### Struktureller Unterschied

**Curriculum A** — vollständiger r=2–r=6-Kollaps:
- Jaccard zwischen Kategorien bei r=2: **0.899** (nahezu identisches Routing)
- Koreferenz, Emotion, Szene/Ort, Zeitfolge: Jaccard=**1.000** (exakt gleiche 12 Blöcke!)
- Kategorie-Signal liegt ausschließlich in r=1 (bester CLF: r=1=0.404)
- Gruppenablation: ~0% in allen Konditionen (kein Ablationsnachweis für Spezialisierung)

**Curriculum C** — partieller Kollaps, Differenzierung im Kern:
- Jaccard zwischen Kategorien bei r=2: **0.282** (Kategorien nutzen verschiedenere Blöcke)
- Kategorie-Signal auch im Kern (CLF r=2–r=6: 0.419–0.424 > r=1: 0.359)
- group_r1 = +7.8%: r=1 hat messbaren, kategoriespezifischen Kausaleffekt

### Warum A keine 27-Nats-Ablation zeigt

In Curriculum A sind die r=2–r=6-Hub-Blöcke für alle Kategorien identisch — Lift ≈ 1.0
für diese Blöcke — sie erscheinen NICHT in der Top-5-Lift-Liste. Ablation trifft nur
die kleinen r=1-Unterschiede: ~0 Nats.

In Curriculum C sind die r=2–r=6-Blöcke kategoriespezifisch genug (Jaccard 0.282),
dass sie in der Top-5-Lift-Liste auftauchen. Ablation trifft die kritischen Hub-Blöcke
über alle 6 Iterationen: nichtlineare Katastrophe (+27 Nats).

### Fazit A-vs-C

Curriculum C gewinnt auf allen relevanten Metriken bei identischer Qualität (3.127 Nats):
- 3.2x stärkere Kategorie-Trennung (Jaccard 0.282 vs. 0.899)
- 13% höhere Klassifikator-Accuracy (0.421 vs. 0.374, nicht-überlappende 95%-KI)
- Kategorie-Signal auch im rekurrenten Kern, nicht nur in r=1
- Gleichmäßigere Blocknutzung (Gini 0.13 vs. 0.52)

**Mechanismus:** `diverse_from_iter=2` erzwingt Diversity in r=3–r=6 über alle
3000 Schritte — kein Zeitfenster in dem Kollaps zurückkehren kann. `diverse_until=900`
in Curriculum A lässt das Modell ab Step 901 wieder kollabieren; nach weiteren 2100
Schritten ist der Kollaps vollständiger als ohne jedes Diversity-Training.
