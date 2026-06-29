# Abschlussbericht — SR-Core k8-Experimente auf HeteroMini

*Eingefroren: 2026-06-20. Kanonisches Modell: `srcore_b32_k8_R6 @15k`.*
*Snapshot: `BASELINE_srcore_b32_k8_R6_15k.pt` (unveränderlich).*

**Gesamtbefund:**

> k8 öffnet einen adaptiven Rekursionsbereich.
> R3/4 maximiert Generalisierung bei halbem Compute.
> R6 maximiert absoluten Gain — aber auf Kosten der Unknown-Retention.
> KL-Stopping findet diesen Bereich automatisch, ohne Retraining,
> mit domain-adaptiver Compute-Allocation.
> Routing-Traces sind keine reinen Sparse-Selections:
> Sie exponieren laufzeit-nutzbare Cache- und Prefetch-Strukturen.
> Größere Bänke (b64) erzeugen funktionale Code-Spezialisierung ohne Domain-Block-Separation:
> Mehr Routing-Diversität verbessert Code-Muster sichtbar, erschwert aber Cache-Pinning.

---

## 1. Forschungsfrage

**Haupthypothese:** Ist k=4 zu eng als Fixed Core? Kann ein größerer Working Set
(k=8) die Rekursionstiefe besser ausnutzen und dabei auf unbekannten Dokumenten
generalisieren?

**Erweiterung (nach Adaptive-Stopping-Experiment):** Ist die Rekursionstiefe R ein
fixer Hyperparameter — oder ein adaptiver Compute-Mechanismus der token-lokal den
sweet spot zwischen Generalisierung und absolutem Gain findet?

**Kontrollvariablen:** Selbes Backbone (10.8M Params, d=256), selbe Bank (b=32),
selbe Trainingsinfrastruktur, kein domain-label im Modell.

---

## 2. Modelle und Trainingsverlauf

| Modell | k | R | Steps | Lfin (final eval) |
|---|---|---|---|---|
| dense_d24 | — | — | 10k | 5.133 |
| naked_b32_R6 | 4 | 6 | 10k | 5.200 |
| srcore_b32_R6 (k4) | 4 | 6 | 10k | 5.147 |
| **srcore_b32_k8_R6** | **8** | **6** | **15k** | **5.167** |
| srcore_b32_k8_R8 (Ablation) | 8 | 8 | 10k | 5.347 |

**Trainings-Verlauf k8_R6 (Milestone-Eval, 6 Batches):**

| Step | Lfin | anytime | WS |
|---|---|---|---|
| 5k | 5.654 | 0.016 | 8.0 |
| 10k | 5.526 | 0.029 | 8.0 |
| 12.5k | 5.413 | 0.034 | 8.0 |
| **15k** | **5.167** | **0.055** | **8.0** |

WS=8.0 und reuse_p90=9.0 sind **trainingsinvariant** — die Routing-Statistik
stabilisiert sich nicht durch mehr Training, sondern ist strukturell durch
`core_mode="per_token"` garantiert.

---

## 3. Hauptbefund: Rekursions-Generalisierung

### 3.1 Anytime-Gewinn und Transferqualität

*gain = loss_r1 − loss_rR pro Token. anytime_ratio = gain_unknown / gain_seen.*
*Wert ≥ 1.0 = Rekursion überträgt sich vollständig auf unbekannte Dokumente.*

| Modell | code_gain_seen | code_gain_unk | anytime_ratio |
|---|---|---|---|
| dense_d24 @10k | 0.034 | 0.018 | 0.43 |
| naked_b32_R6 @10k | 0.066 | 0.043 | 0.78 |
| srcore_b32_R6 (k4) @10k | 0.038 | 0.025 | 0.66 |
| **srcore_b32_k8_R6 @15k** | **0.063** | **0.057** | **0.90–1.05** |

**Befund (robust über beide Eval-Läufe @10k und @15k):**

k8_R6 erreicht mit code_gain=0.063 fast das Niveau von Naked (0.066), während
es den anytime_ratio von 0.66 (k4) auf **0.90** steigert. Bei Domain-spezifischer
Auswertung übersteigt code_gain_unk gelegentlich code_gain_seen (Ratio>1.0) —
die zusätzlichen Iterationen helfen auf unbekanntem Code stärker als auf
gesehenen Mustern.

### 3.2 Domain-Partition — kein Confound

Alle Modelle zeigen schwache Domänen-Partition (excl ≈ 0.40, Zufall 0.25,
domain-Jaccard ≈ 0.30). k8 lernt keine Domänen-Silos — die Cores sind
funktional gemischt. Befund über zwei unabhängige Eval-Läufe stabil.

---

## 4. Anytime-Inferenz-Kurve (Adaptive Compute)

*Für k8_R6@15k: Modell läuft immer mit vollem R=6, logits[r-1] wird als
Ausgabe der Iteration r verwendet. Compute linear proportional zu r.*

| R | Compute | code_gain_seen | code_gain_unk | code_ratio |
|---|---|---|---|---|
| 1 | 17% | 0.000 | 0.000 | — |
| 2 | 33% | 0.029 | 0.026 | 0.87 |
| **3** | **50%** | **0.047** | **0.044** | **0.94** |
| **4** | **67%** | **0.051** | **0.051** | **1.00** |
| 6 | 100% | 0.056 | 0.061 | 1.09 |

**Mechanistisches Bild:**

- **R=1:** Keine Verbesserung. `r1` wählt und initialisiert den Core — die
  eigentliche rekursive Verarbeitung beginnt ab `r2`.
- **R=3:** 50% Compute, 84% des finalen code_gains, ratio=0.94.
- **R=4:** 67% Compute, 91% des finalen code_gains, ratio=1.00 — Rekursion
  überträgt sich vollständig auf unbekannte Dokumente.
- **R=6:** 100% Compute, ratio=1.09 — unknown übertrifft seen.

SR-Core hat damit zwei natürliche Betriebsmodi:

```
Standardmodus  R=3/R=4   50–67% Compute   ratio ≈ 1.0
Quality-Modus  R=6       100% Compute     ratio = 1.09
```

Der Core (`k=8` Blöcke, WS=8.0) bleibt in beiden Modi identisch.
Adaptive Compute zahlt ausschließlich Iterations-Kosten, nicht Speicher.

---

## 5. R8-Ablation — Sättigungsnachweis

*k8_R8 @10k, von Scratch. Anytime-Kurve:*

| R | code_gain_seen | code_gain_unk | ratio |
|---|---|---|---|
| 2 | 0.009 | 0.007 | 0.84 |
| 3 | 0.014 | 0.014 | 0.97 |
| 4 | 0.019 | 0.018 | 0.95 |
| 6 | 0.022 | 0.024 | 1.09 |
| **8** | **0.026** | **0.028** | **1.07** |

**Interpretation:**

R8@10k ist strukturell korrekt (ratio>1.0 ab R=6) aber quantitativ unreif —
analog zu R6@10k (anytime=0.029) vs. R6@15k (anytime=0.055). Der Modell hat
mehr Training gebraucht um alle Tiefen zu nutzen.

**Marginalgewinn R=6→R=8:** code_gain +0.004, ratio 1.085→1.065 (leicht fallend).
Die zwei Extra-Iterationen fügen realen aber kleinen Zuwachs hinzu. Die Ratio
fällt leicht — R=8 nutzt denselben Core und beginnt, auf gesehene Patterns zu
fitten, wo R=6 noch robuster generalisiert.

**Entscheidung:** R8@15k nicht durchgeführt. Der Grenznutzen nach R6 ist klein.
k8_R6 ist der **praktische Sweet Spot**.

---

## 6. Working Set und Offload-Projektion

| Modell | WS | Transfervorteil @b32 | Transfervorteil @b8192 |
|---|---|---|---|
| Dense d24 | 24.0 | 1× | 1× |
| Naked R6 | 5.75 | 4× | ~1420× |
| k4_R6 | 4.0 | 6× | 2048× |
| **k8_R6** | **8.0** | **3×** | **1024×** |

*Transfervorteil = n_blocks / WS. Bei b=8192 dominiert die Blockgröße.*

k8 hat den doppelten WS von k4, bleibt aber im Zielregime (b=8192) bei
1024× Transfervorteil gegenüber Dense. Mehr `k` kostet Transfervorteil linear —
aber gewinnt dafür Code-Generalisierung.

---

## 7. Adaptive Stopping — Finding the Generalization Sweet Spot

*`scripts/adaptive_stopping.py`. Offline-Simulation: volles R=6 Forward-Pass,
Stopping-Entscheidung nachträglich per Token. Kein Re-Training.*

### 7.1 Motivation

Die Anytime-Kurve (Abschnitt 4) und Cross-Seed-Analyse (AUSWERTUNG_CROSSSEED_K8_R6.md)
zeigen: R=3/4 maximiert code_ratio, R=6 maximiert absoluten Gain — aber auf Kosten der
Generalisierung (ratio fällt in beiden stabilen Seeds unter 1.0 bei R=6). Die Frage:
Kann ein einfaches Token-lokales Kriterium diesen Sweet Spot automatisch finden?

### 7.2 Methode

Drei Kriterien, getestet mit je zwei min_R-Werten (2, 3) und mehreren Thresholds:

| Kriterium | Beschreibung |
|---|---|
| `top1_stable` | Stoppe wenn argmax(logits[r]) = argmax(logits[r-1]) für N Iterationen |
| `entropy_drop` | Stoppe wenn \|H(r) − H(r−1)\| < θ |
| `kl_div` | Stoppe wenn KL(p_r ‖ p_{r−1}) < θ |

Plus fixe Baselines (R=2, 3, 4, 6). Alle 20 Configs in einem Forward-Pass —
kein Messrauschen durch separate Läufe. Ergebnisse: Seeds 0+1 (Hauptbefund),
Seed 2 (Stress-Test). Gesamt: 40 Batches × bs=16 × seq_len=128.

### 7.3 Hauptbefund — Seed-Mittelwert (Seeds 0+1)

| Config | ratio | cg_unk | mean_R | saved | Lfin |
|---|---|---|---|---|---|
| fixed_R2 | 1.016 | 0.027 | 2.00 | 67% | 5.172 |
| fixed_R3 | 1.007 | 0.042 | 3.00 | 50% | 5.161 |
| fixed_R4 | 1.001 | 0.050 | 4.00 | 33% | 5.155 |
| **kl_div_minR3_t0.005** | **1.022** | **0.048** | **3.25** | **46%** | **5.158** |
| kl_div_minR3_t0.01 | 1.001 | 0.045 | 3.08 | 49% | 5.159 |
| top1_stable_minR3_c1 | 1.005 | 0.043 | 3.04 | 49% | 5.161 |
| kl_div_minR2_t0.005 | 1.038 | 0.043 | 2.39 | 60% | 5.162 |
| **fixed_R6** | **0.990** | **0.055** | **6.00** | **0%** | **5.153** |

**Kernergebnis:** Fixed R=6 hat den schlechtesten code_ratio aller Configs mit
mean_R ≥ 3. Jede adaptive Konfiguration die im Bereich mean_R ≈ 3–4 stoppt
übertrifft R=6 bei ratio.

**Bester Kandidat: `kl_div_minR3_t0.005`**

```
vs. fixed_R6:

code_ratio:     1.022  vs.  0.990  (+3.2 Punkte — Unknown-Retention besser)
code_gain_unk:  0.048  vs.  0.055  (87% des R6-Unknown-Gains erhalten)
mean_R:         3.25   vs.  6.00   (46% weniger Rekursionscompute)
Lfin:           5.158  vs.  5.153  (nahezu identische absolute Qualität)
```

**Paper-Formulierung:**

> A simple KL-based stopping rule with a minimum depth of three recursions
> recovers 87% of the fixed-R6 held-out code gain while using only 54% of
> the recursive compute. It also improves the seen-to-held-out gain ratio
> from 0.990 to 1.022 and allocates more iterations to code tokens than to
> other domains, indicating that recursive depth can be used as an adaptive
> compute mechanism rather than a fixed hyperparameter.

### 7.4 Per-Domain Compute Allocation

Code-Token erhalten in beiden stabilen Seeds konsistent mehr Iterationen:

| Domäne | S0 mean_R | S1 mean_R |
|---|---|---|
| web | 3.257 | 3.086 |
| wiki | 3.286 | 3.114 |
| lit | 3.333 | 3.228 |
| **code** | **3.410** | **3.280** |

Das Modell entscheidet token-lokal: Code braucht mehr Verarbeitung. Dieser Befund
ist seed-übergreifend stabil und tritt ebenso bei top1_stable auf.

### 7.5 Seed-2 Stress-Test — Attraktor-Rescue

Auch im Memorisierungs-Attraktor (Seed 2) verbessert frühes Stoppen die Retention:

| Config | ratio | mean_R |
|---|---|---|
| fixed_R6 | 0.586 | 6.0 |
| kl_div_minR3_t0.05 | 0.723 | 3.01 |
| fixed_R3 | 0.730 | 3.0 |

Ratio steigt von 0.586 auf 0.723 (+23 Punkte) — adaptive Stopping mildert
den Seen-Spezialisierungseffekt selbst im ungünstigsten Seed.

### 7.6 Interpretation: Zwei Phasen der Rekursion

```
R=1          Adressierung: Core-Selektion, kein Gain
R=2–4        Strukturphase: generalisierende Verarbeitung,
             Unknown profitiert ≥ Seen (ratio ≥ 1.0 in Seeds 0+1)
R=5–6        Spezialisierungsphase: mehr absoluter Gain,
             aber stärkere Anpassung an Trainingsverteilung
             (ratio fällt unter 1.0)
```

KL-Divergenz mit min_R=3 erkennt den Übergang von Struktur- zu Spezialisierungsphase
token-lokal und stoppt vor dem Generalisierungsknick.

---

## 8. Leiterbahn-Simulation — Trace-getriebene Offloading-Analyse

*Methode: trace-getriebener Cache-Simulator. Policies gelernt auf Seen-Daten (Calib),
getestet auf Held-out-Daten. Stopping-Config: kl_minR3_t0.005. Seeds 0+1.*

### 8.1 SR-Core Transfer Law

SR-Core lädt in `per_token`-Modus k=8 Blöcke **genau einmal pro Token**, unabhängig
von R. Alle weiteren Rekursionen reuse-n denselben Core ohne Nachladen.

```
bytes/token = k × block_bytes_fp16 ≈ 8 × 514.5 KB ≈ 4116 KB (raw)
Miss-Anteil bei K=16: ~0.1 → Demand ≈ 1270–1346 KB/token
```

**Konsequenz:**
- R ist ein **Compute-Dial**, kein Transfer-Dial
- Adaptive Stopping spart Compute (mean_R 6→3.25), nicht Transfer
- Transfer-Optimierung ist Aufgabe des Cache/Prefetch-Systems

### 8.2 Trace-getriebener Prefetch: Transition-Prefetch

Die Routing-Sequenz zeigt zeitliche Struktur: Token t selektiert Core C, Token t+1
selektiert mit hoher Wahrscheinlichkeit einen Nachfolger-Core D aus der
Transitions-Tabelle.

**Befund:** `transition_pf` bei K=16, held-out:

| Seed | LRU | transition_pf | Reduktion |
|---|---|---|---|
| 0 | 1266 KB | 1117 KB | −11.8% |
| 1 | 1345 KB | 1282 KB | −4.8% |

Prefetch-Accuracy bei K=16: 0.36–0.54 (40–46% der prefetchten Blöcke werden vor
Eviction genutzt). Bei K=32 (Sättigungspunkt): 1.000.

### 8.3 Routing-inferred Domain Cache (ohne Oracle-Label)

**Idee:** Statt echter Domain-Labels (oracle) kann die Domäne aus dem ersten Core
der Sequenz via Block-Overlap-Vergleich mit kalibrierten Domain-Profilen inferiert
werden.

**Befund:** `trans+inferred_dom` bei K=16, held-out:

| Seed | oracle | inferred | oracle-Gap |
|---|---|---|---|
| 0 | 1025 KB | 1088 KB | +63 KB (+6.1%) |
| 1 | 1170 KB | 1270 KB | +100 KB (+8.6%) |

`trans+inferred_dom` schlägt `transition_pf` in beiden Seeds ohne Oracle-Label:
- S0: 1088 vs 1117 KB; S1: 1270 vs 1282 KB

Das Routing-Signal ist ein brauchbarer Domain-Detektor.

### 8.4 Demand vs. Total Bytes

Prefetch reduziert blockierende Demand-Transfers, verschiebt aber Bandbreite in
nicht-blockierende Prefetch-Loads.

Bei K=32 (Sättigungsgrenze), trans+inferred_dom:
- Demand: 56/61 KB (vs. LRU 117/126 KB)
- Total (inkl. Prefetch-Waste): 95/104 KB
- transition_pf total zum Vergleich: 119/128 KB

Domain-aware Policies reduzieren **sowohl** Demand als auch Total — kein reines
Verschieben auf Prefetch-Waste.

### 8.5 Hot-Core Floor am Sättigungspunkt

Bei K=32 passt die gesamte b32-Blockbank in den Cache. Demand-Misses entstehen
ausschließlich als Cold-Start-Kosten der ersten ~4 Tokens jeder Sequenz.

**Befund:** `trans+hot_core` bei K=32 — besser als oracle domain:

| Policy | S0 | S1 |
|---|---|---|
| LRU | 117 KB | 126 KB |
| trans+domain_hot **(oracle)** | 57 KB | 63 KB |
| trans+inferred_dom | 56 KB | 61 KB |
| **trans+hot_core** | **55 KB** | **60 KB** |

Globale Hot-Core-Pinning (Top-1-Core = 8 Blöcke, domain-agnostisch) schlägt
domain-spezifische Profile, weil diese 8 Blöcke cross-domain die universellsten
Routing-Pfade halten.

### 8.6 Limitation: K=32 = b32-Saturation, kein allgemeiner Offloading-Befund

K=32 bei b32 bedeutet: gesamte Bank resident. Das ist kein Offloading-Regime,
sondern die Cache-Obergrenze des b32-Modells.

```
b32 Cache-Regime:
  K=8  →  25% der Bank   (echter Offloading-Druck)
  K=16 →  50% der Bank   (Haupt-Offloading-Regime)
  K=24 →  75% der Bank
  K=32 → 100% der Bank   (Saturation: Upper Bound)
```

Die 53%-Demand-Reduktion bei K=32 ist ein **Sanity/Upper-Bound-Befund**, nicht
direkt extrapolierbar auf größere Banken (b64/b128).

Der robuste, extrapolierbare Befund liegt bei K=16–24:

> In the non-saturated cache regime (K=16–24), policies that combine transition
> prefetching with routing-inferred domain profiles reduce held-out demand transfer
> by 5–20% relative to LRU without oracle domain labels. At the b32 saturation
> boundary (K=32), global hot-core pinning achieves the lowest cold-start demand
> because it preserves cross-domain routing paths more uniformly than domain-specific
> profiles.

### 8.7 Strukturaussage

> SR-Core routing traces are not merely sparse selections; they expose
> runtime-usable structure for cache and prefetch policies.

---

## 9. Zusammenfassung der Befunde (aktualisiert)

| Befund | Status | Robustheit |
|---|---|---|
| k8_R6 erreicht Qualität von k4/Naked | ✓ | 2 Eval-Läufe |
| WS=k=8 exakt und seed-invariant | ✓ | 3 Seeds |
| code_gain_ratio k8 > k4 (0.83–1.02 vs. 0.65–0.66) | ✓ | Cross-Seed stabil |
| R=3/4 ist Generalisierungs-Sweet-Spot (ratio ≥ 1.0) | ✓ | Seeds 0+1 |
| R=6 spezialisiert auf Seen (ratio < 1.0) | ✓ | Seeds 0+1 |
| kein Domänen-Partition-Confound | ✓ | 2 Eval-Läufe |
| R8 bringt Marginalgewinn nach R6 | ✓ | R8-Ablation @10k |
| r1=Adressierung, r2+=Verarbeitung | ✓ | R=1 gibt 0 Gain |
| KL-Stopping findet R3/4-Sweet-Spot automatisch | ✓ | Seeds 0+1 + Seed 2 |
| Code bekommt automatisch mehr Iterationen | ✓ | Seeds 0+1 konsistent |
| Zwei Routing-Attraktoren (Struktur vs. Memorisierung) | ✓ | Cross-Seed |
| bytes/token R-invariant (SR-Core Transfer Law) | ✓ | Seeds 0+1 |
| transition_pf −5–12% Demand bei K=16 (held-out) | ✓ | Seeds 0+1 |
| trans+inferred_dom schlägt transition_pf ohne oracle | ✓ | Seeds 0+1 |
| Oracle-Gap K=16 nur +6–9% | ✓ | Seeds 0+1 |
| trans+hot_core schlägt oracle bei K=32 (Saturation) | ✓ | Seeds 0+1 |
| K=32 = b32-Saturation, nicht allg. Offloading-Befund | ✓ | (Caveat) |

**Konzeptionelle Gesamtaussage:**

> k8 öffnet einen adaptiven Rekursionsbereich. R3/4 maximiert Generalisierung
> bei halbem Compute. R6 maximiert absoluten Gain bei Kosten der Unknown-Retention.
> KL-Stopping findet diesen Bereich automatisch, ohne Retraining, mit
> domain-adaptiver Compute-Allocation: Code bekommt mehr Iterationen als Web/Wiki.
> SR-Core ist damit kein reiner Offloading-Trick, sondern ein Adaptive-Compute-
> und Leiterbahn-Kandidat: Routing-Traces exponieren laufzeit-nutzbare Cache-
> und Prefetch-Strukturen ohne Oracle-Labels.

---

## 10. Offene Punkte (nächste Schritte)

**Nach dieser Runde abgeschlossen:**
- k8_R6 Cross-Seed (Seeds 0/1/2) ✓
- Adaptive Stopping Experiment ✓
- R8-Ablation ✓
- Leiterbahn-Simulator v1 (seen-oracle) ✓
- Leiterbahn-Simulator v3 (calib/held-out split, oracle vs. inferred) ✓

**Offene Skalierungsfrage:**

Die Leiterbahn-Befunde bei K=16–24 sind claim-sauber, aber auf b32 beschränkt.
Die zentrale offene Frage:

> Bleiben Transition-/Inferred-Domain-/Hot-Core-Leiterbahnen nützlich,
> wenn die Bank größer ist als der Cache und **nicht** vollständig resident
> werden kann?

Das erfordert Experimente mit größerer Bank bei gleichem k=8:

```
Ziel: cache fraction = K/n_blocks dauerhaft < 1
b32: K=16 → 50%  (bereits simuliert)
b64: K=16 → 25%  (echter Offloading-Druck)
b128: K=16 → 12.5% (starkes Offloading-Regime)
```

**Abgeschlossen:**

1. ✓ **b64/k8/R6 Training** (@10k + @15k) — Leiterbahn, gain_su, routing, qual_gen vollständig
2. ✓ **Leiterbahn v3 auf b64-Traces** — policies bei K/n=25% schwächer als bei b32 K/n=50%

**Sinnvolle nächste Schritte:**

1. **Plot:** Compute saved vs. demand_bytes/token (Pareto-Kurve: b32 vs b64, adaptive stopping)
2. **Routing-Druck-Experiment** — Auxiliary Loss für Lokalität/Stabilität: löst das den Jaccard=0.984-Befund?
3. **CPU/RAM Offload Prototype v0** — direkte Zeitmessung statt Simulation
4. **b128 Cache-Regime** — nur wenn Routing-Druck-Exp Jaccard < 0.5 zeigt

**Nicht mehr notwendig:**
- k8_R8 @15k — Grenznutzen zu gering
- R10/R12 — kein Befund motiviert weitere Tiefe
- Weitere Policy-Varianten auf b32 — Sättigungseffekt dominiert ab K=32
- b64 weiteres Training — b64@15k abgeschlossen, Kernbefunde stabil

---

## 11. Caveats

- Alle Metriken auf b=32 (32 Blöcke à 263k Params). Skalierung zu b≥64 ungetestet.
- k8@15k vs. andere @10k: nicht step-gematcht — k8 hat 50% mehr Tokens gesehen.
- Adaptive Stopping: offline simuliert — echter Speedup erfordert Batch-Masking
  (gestoppte Tokens nicht weiterrechnen). Die ratio/gain-Aussagen sind real,
  der compute_saved-Wert ist ein Proxy.
- ratio-Metrik hat Eval-Varianz (±0.05 zwischen Runs). Werte gelten als Trend,
  nicht als Punktaussage.
- Leiterbahn-Simulation: Cache-Sim ist per-Sequenz, kein cross-Sequenz-Cache.
  Reale Serving-Systeme können über Sequenzen hinweg cachen — das würde
  Demand-Bytes weiter senken.
- K=32 = b32-Saturation: der 53%-Demand-Reduktionsbefund ist ein Upper-Bound
  für b32, nicht extrapolierbar auf b64/b128 ohne weiteres Training und Simulation.
- trans+inferred_dom inferiert Domain aus dem **ersten** Core der Sequenz.
  Bei sehr kurzen Prompts oder Domaenwechsel innerhalb der Sequenz ist die
  Inferenz suboptimal.
- Seed-2-Attraktor: Ursache des Routing-Kollaps (reuseP90=54) noch ungeklärt.

---

## 12. Qualitative Generation: Functional Code Specialization without Domain Block Separation

To complement the loss-, gain-, and routing-based evaluations, we compared qualitative generations from `b32/k8/R6@15k`, `b64/k8/R6@10k`, and `b64/k8/R6@15k` across code, wiki, literature, and web-style prompts.

The qualitative results confirm that increasing the bank from 32 to 64 blocks does not simply improve all domains uniformly. Instead, `b64@15k` develops visibly more specific code-like behavior while remaining weak or even confused in several non-code prompts.

### 12.1 Code-specific improvements

The clearest qualitative improvement appears in code prompts. Compared to `b32@15k`, `b64@15k` produces more Python-specific continuations:

* For `def fibonacci(`, `b32@15k` collapses into repetitive pseudo-symbols, while `b64@15k` produces class/test-style patterns such as `self.assertEqual(...)`.
* For `try: import`, `b32@15k` falls into docstring-like repetition, while `b64@15k` produces relative-import and module-path fragments.
* For `class User:`, `b64` more reliably enters docstring/class-body patterns.
* Code-domain token hits and `distinct_2` are higher in `b64@15k`, indicating less repetitive and more domain-specific output.

The per-iteration comparison also shows one of the few visible greedy-generation changes: for the `def fibonacci(` prompt, `b64@15k` shifts from a generic `self.get(...)` pattern at R1 to a more test-framework-like `self.assertEqual(...)` pattern at R6. This suggests that recursive refinement can sharpen a chosen code-like mode, even when greedy decoding often masks smaller distributional improvements.

### 12.2 Non-code degradation and domain confusion

The improvement is not uniform. Wiki and web prompts remain weak, and in several cases `b64@15k` is worse than `b32@15k`.

Examples include:

* `The capital of France is` degenerating into repeated generic tokens.
* `In mathematics, a function is` failing to preserve a mathematical/wiki-style continuation.
* `How to make a cup of tea:` collapsing into blank-line repetition.
* `== History of` producing code-comment-like tokens such as repeated `#`.

This is important because it matches the routing analysis: `b64@15k` does not form clear domain-separated hot-block experts. The code/other hot-block Jaccard remains extremely high, indicating that domains continue to share nearly the same hot blocks. The model therefore becomes more code-specific in behavior without developing simple code-specific block sets.

### 12.3 Interpretation

The qualitative generation results support the broader b64 finding:

Increasing the bank size gives the model more functional capacity, but this capacity is not organized into clean domain experts. Instead, `b64@15k` appears to develop functional specialization over shared blocks. Code benefits most visibly from this: generations become more Python-like, less repetitive, and more structurally specific. However, the same shared routing structure can interfere with wiki and web-style prompts, producing domain confusion and repetitive generic continuations.

This explains the apparent tension between the metrics:

* Routing remains domain-uniform.
* Unique core combinations increase substantially.
* Code generations become more specific.
* Code held-out retention remains weak, suggesting seen-code specialization.
* Wiki/web/lit behavior does not improve uniformly.

Thus, b64 should not be described as learning domain experts. A more accurate interpretation is:

> b64 increases functional routing diversity and enables stronger code-pattern specialization, but without domain-level block separation. The additional capacity improves some structured behaviors while also exposing domain-confusion failure modes.

### 12.4 Consequence for the architecture story

This result sharpens the design picture. Larger banks can reduce capacity pressure and allow more specialized behavior, but specialization does not automatically become cache-friendly or domain-aligned. In fact, as routing diversity increases, simple hot-block and hot-core pinning become less effective.

This suggests a future research direction: if larger sparse recursive banks are to support both specialization and offloading, routing may need additional pressure toward locality, stable core reuse, or cache-aware organization. Otherwise, larger banks may improve model behavior while making runtime caching harder.
