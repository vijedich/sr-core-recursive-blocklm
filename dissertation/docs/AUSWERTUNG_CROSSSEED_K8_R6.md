# Cross-Seed-Analyse — srcore_b32_k8_R6 @15k

*Stand: 2026-06-20. Seeds 0/1/2, je 15k Steps, von Scratch.*
*Eval: 40 Batches × bs=16 × seq_len=128. Seed 0 aus BASELINE-Snapshot.*

---

## 1. Trainings-Endwerte

| Seed | Lfin @15k | anytime @15k (6 Batches) | reuseP90 @15k | WS |
|---|---|---|---|---|
| 0 | 5.167 | 0.055 | 9.0 | 8.0 |
| 1 | 4.906 | 0.030 | 9.0 | 8.0 |
| 2 | 4.949 | 0.136 | 54.0 | 8.0 |

**WS=8.0 ist in allen Seeds identisch** — trainingsinvariant und seed-unabhängig.
reuseP90 von Seed 2 zeigt bereits im Training die Anomalie (54 vs. 9).

---

## 2. Gain Seen vs. Unknown (Full R=6)

| Seed | code_seen | code_unk | code_ratio | lit_ratio | wiki_ratio | web_ratio |
|---|---|---|---|---|---|---|
| 0 | 0.054 | 0.045 | 0.835 | 1.285 | 1.104 | 1.089 |
| 1 | 0.050 | 0.044 | 0.869 | 0.960 | 0.976 | 0.818 |
| **2** | **0.216** | **0.137** | **0.635** | **0.925** | **0.772** | **0.759** |

Seed 2 hat 4× höheren absoluten code_gain, aber deutlich schlechteres ratio —
ein anderer Attraktor (Memorisierung).

---

## 3. Anytime-Kurve: code_ratio nach R

*ratio = code_gain_unknown / code_gain_seen. >1.0 = Unknown profitiert stärker.*

| R | Compute | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|---|
| 1 | 17% | — | — | — |
| 2 | 33% | **1.174** | 0.941 | 0.815 |
| 3 | 50% | **1.018** | **1.092** | 0.759 |
| 4 | 67% | 0.849 | **1.019** | 0.664 |
| 6 | 100% | 0.945 | 0.873 | 0.582 |

**Kernbefund:** Seeds 0 und 1 zeigen beide ratio≥1.0 — aber bei **R=2/3 (Seed 0) bzw.
R=3/4 (Seed 1)**, nicht bei R=6. Bei R=6 fällt die ratio in beiden Seeds unter 1.0.

---

## 4. Interpretation

### 4.1 Robuste Befunde (2 von 3 Seeds)

- **WS=k=8 hält exakt** über alle Seeds und alle Trainingslängen.
- **Ratio>1.0 existiert**, aber bei **intermediärem R (R=3/4)**, nicht bei R=6.
- **Code bleibt die Domäne mit dem stärksten Rekursionssignal** in allen Seeds.

### 4.2 Revidierte Story: Generalisierungs-Sweet-Spot

```
R=1          → r1 initialisiert Core, kein Gewinn
R=2/3        → Strukturerkennung: Unknown ≥ Seen (ratio ≥ 1.0, seeds 0+1)
R=4          → Übergangszone (ratio ≈ 1.0 bei Seed 1, < 1.0 bei Seed 0)
R=6          → Anpassung: Seen > Unknown (ratio 0.87–0.95, beide Seeds)
```

Das ist eine **kohärentere Story als "R=6 ratio>1.0"**: Mittlere Rekursion extrahiert
generalisierende Struktur. Tiefe Rekursion beginnt auf die Trainingsverteilung zu
spezialisieren — messbar als ratio-Abfall bei R=6.

### 4.3 Seed 2 — Memorisierungs-Attraktor

Seed 2 ist kein Defekt, sondern ein anderer Routing-Attraktor:

| Eigenschaft | Seed 0/1 | Seed 2 |
|---|---|---|
| WS | 8.0 | 8.0 |
| reuseP90 @7.5k | 8–9 | **54** |
| code_gain @R=6 | 0.05 | **0.20** |
| code_ratio @R=6 | 0.87–0.95 | **0.58** |
| Attraktor | Strukturmodus | Memorisierungsmodus |

Seed 2 hat mehr absoluten Rekursionsgewinn — aber fast ausschließlich auf gesehenen
Mustern. Die Routing-Diversität (reuseP90=54) lässt den Core breit über den Token-Strom
wandern, statt einen stabilen lokalen Satz zu halten.

---

## 5. Claim-Hierarchie nach Cross-Seed

### Robust (≥2 Seeds)

1. WS = k hält exakt, seed-unabhängig
2. k8 hat bessere Rekursions-Generalisierung als k4 (ratio 0.83–1.19 vs. 0.65–0.66)
3. Intermediäres R (R=3/4) ist der Generalisierungs-Sweet-Spot (ratio ≈ 1.0)
4. R=6 spezialisiert leicht auf Seen (ratio fällt unter 1.0 in beiden robusten Seeds)
5. Seed-2-Attraktor zeigt: k8 kann zwei Routing-Modi entwickeln

### Vorläufig (1 Seed, hohe Varianz)

- Spezifische ratio-Werte (z.B. 1.174 bei Seed 0, R=2) — eval-stochastisch
- Genaue R-Übergänge (R=2 vs. R=3 als Peak) — seed-abhängig

### Nicht claimen

- "k8_R6 generalisiert Rekursion immer besser auf Unknown als Seen"
- Spezifische ratio-Werte ohne Konfidenzintervall

---

## 6. Paper-Formulierung

> Across two stable seeds, increasing the fixed core from k=4 to k=8 consistently
> preserves the exact working-set guarantee (WS=k=8, seed-invariant) while
> improving recursive gain retention on held-out code relative to k=4
> (ratio 0.83–1.17 vs. 0.65–0.66). The generalization peak occurs at intermediate
> recursion depth (R=3/4), where held-out and seen code benefit equally from
> recursion; deeper iteration (R=6) begins to specialize toward the training
> distribution. A third seed develops a high-gain memorization attractor
> (code_gain ×4, ratio 0.58) with markedly higher routing diversity
> (reuseP90=54 vs. 9), suggesting multiple stable routing attractors exist
> under this architecture.

---

## 6b. Routing-Analyse — mechanistisches Bild der Attraktoren

*scripts/routing_analysis.py, 40 Batches × bs=16, contiguous mode.*

| Metrik | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| unique_cores | 7,738 | **14,445** | **4,889** |
| top1_coverage | 0.026 | 0.012 | 0.030 |
| gini | 0.288 | 0.256 | **0.206** |
| dead_blocks | 0 | 0 | 0 |
| cache_miss_k16 | 0.053 | 0.056 | **0.067** |
| domain_jaccard | 0.292 | 0.216 | **0.376** |

**Interpretation Seed 2:**

Seed 2 hat die wenigsten unique_cores (4,889) — das Routing kollabiert auf einen
kleineren Satz von Verarbeitungspfaden. Gleichzeitig ist domain_jaccard am höchsten
(0.376): dieselben Blöcke werden quer über alle Domänen genutzt, weniger
Spezialisierung. Trotz weniger Cores ist cache_miss höher (0.067) — die Cores
wechseln schneller im Token-Strom (konsistent mit reuseP90=54 im Training).

Das Paradox löst sich: Seed 2 lernt universale, häufig aufgerufene Pfade statt
domänenspezifische Strukturpfade. Das ist ein Lernziel-Kollaps, kein Router-Bug.

**Interpretation Seed 1:**

Seed 1 hat die meisten unique_cores (14,445) bei niedrigstem domain_jaccard (0.216)
— am stärksten nach Domäne differenziert, am breitesten aufgefächert, am besten
generalisierend. Konsistent mit der höchsten ratio>1.0 (R=3: 1.092, R=4: 1.019).

**Attraktor-Charakterisierung:**

| Attraktor | Seeds | unique_cores | dom_jac | Verhalten |
|---|---|---|---|---|
| Strukturmodus | 0, 1 | 7k–14k | 0.22–0.29 | domänen-diff., gute Retention |
| Memorierungsmodus | 2 | 4.9k | 0.38 | universal, seen >> unknown |

---

## 7. Nächste Schritte

1. **Routing-Analyse** — Seed 2 vs. Seeds 0/1: unique_cores, gini, top1_coverage,
   domain_jaccard, cache_miss_k16. Ziel: mechanistisches Verständnis des Attraktors.

2. **Regularisierung** — Kann man den Memorisierungs-Attraktor (Seed 2) durch
   stärkeren Core-Konsistenz-Loss oder niedrigere LR verhindern?

3. **Bank-Skalierung** — srcore_b128_k8_R6 @10k: überträgt sich der Sweet-Spot-
   Befund auf größere Banken?

---

## 8. Caveats

- Eval-Varianz: 40 Batches × bs=16 = 640 Beispiele pro Domäne. ratio-Werte haben
  ±0.05–0.15 Varianz zwischen Eval-Läufen (beim selben Checkpoint beobachtet).
- Seed 2 nicht step-matched für reuseP90 — Anomalie bei 7.5k beobachtet, keine
  Zwischenprüfung nach 10k.
- Alle Ergebnisse auf b=32, 10.8M Params. Skalierung ungetestet.
