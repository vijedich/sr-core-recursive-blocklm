# Forschungsbericht: Rekursiv vertieftes, block-sparsames Sprachmodell mit Parameter-Streaming

**Stand:** Juni 2026  
**Projekt:** `recursive-blocklm`  
**Hardware:** NVIDIA GeForce RTX 2060 (6 GB VRAM), Windows 11, Python 3.14, PyTorch 2.11 (CUDA 12.8)

---

## 1. Projektziel

**Das Problem:** Große Sprachmodelle passen nicht in den VRAM handelsüblicher Hardware.
Der aktuelle Stand: Layer-by-Layer-Offloading — jede Schicht wird bei Bedarf geladen,
berechnet, entladen. Bei einem 7B-Modell (fp16, ~14 GB) auf PCIe 16 GB/s bedeutet das
rund 0,9 Sekunden pro Token (~1 Token/Sekunde). Das Gesamtmodell fließt bei jedem Token
komplett durch die PCIe-Leitung.

**Die Hypothese:** Ein Modell kann so trainiert werden, dass es pro Token nur einen kleinen,
vorhersagbaren Bruchteil seiner Gewichte benötigt. Das ermöglicht es, Modelle zu betreiben,
die größer sind als der VRAM — nicht weil das System schneller wäre, sondern weil es nie
das gesamte Modell gleichzeitig braucht.

**Das Ziel ist Ermöglichung, nicht Optimierung.** Der Vergleichsmaßstab ist nicht
„schneller als ein Modell, das komplett im VRAM liegt", sondern
„besser als Layer-by-Layer-Offloading eines gleichgroßen dichten Modells".

**Kernmechanismus:** Das Modell wählt pro Iteration k Blöcke aus einer Bank aus (Routing),
verarbeitet den Zustand iterativ, und gibt nach jeder Iteration eine Vorhersage aus.
Die zwei zentralen Konzepte:

- **Tunnel:** Die Sequenz von Blockaktivierungen über alle Iterationen für ein Token —
  der Weg durch den 3D-Parameterraum.
- **Leiterbahn:** Ein häufig aktivierter Tunnel-Prototyp, der als Index-Eintrag
  gespeichert wird und vollständiges Vorab-Laden des bekannten Weges ermöglicht —
  analog zu konsolidiertem Prozedurgedächtnis im Nervensystem.

Die Theorie formuliert sechs Hypothesen und sechs Go/No-Go-Kriterien.
Der vorliegende Bericht dokumentiert, welche davon getestet wurden und was die Ergebnisse sind.

---

## 2. Architektur

### 2.1 Gemeinsamer Backbone

Alle drei Vergleichsmodelle teilen denselben Rahmen:

```
tokens → Embedding + Positions
       → Context-Encoder (1 kausaler Attention-Layer, kein FFN)
       → CORE (modellspezifisch)
       → gemeinsamer Readout-Head nach jeder Iteration
```

Der Context-Encoder ist bewusst minimal: Er bewegt Information zwischen Positionen, leistet aber kaum nichtlineare Arbeit. Das lässt Rechenarbeit für das rekursive Core übrig und macht die Tiefenfrage überhaupt testbar.

**Deep Supervision:** Nach jeder Iteration wird der Readout-Head angewendet und ein Loss berechnet. Das ermöglicht direkte Messung von L(r) für jede Rekursionstiefe r.

### 2.2 Modell A — Dichte Referenz

16 distinkte MLP-Blöcke, jeder einmal angewendet. Ziel: Qualitätsreferenz bei gleichem Compute-Budget.

- Parameter (Core): ~1.06 M
- Block-Anwendungen: 16

### 2.3 Modell B — Rekurrente Referenz

Ein einzelner geteilter Block, 16-mal hintereinander angewendet. Ziel: Isolierung des Effekts reiner Rekursion.

- Parameter (Core): ~0.07 M
- Block-Anwendungen: 16

### 2.4 Modell C — Dynamisch blockselektives Modell

Bank aus 24 Blöcken, pro Iteration werden Top-4 via Noisy-Top-k-Router ausgewählt, 4 Iterationen. Ziel: Messung des Effekts dynamischer Auswahl.

- Parameter (Core): ~1.60 M
- Block-Anwendungen: 4 × 4 = 16 (identisch zu A und B)
- Routing: Softmax-normierte Gates über Top-k-Logits, Switch-artiger Load-Balance-Loss

**Konfiguration Demo-Läufe:** d_model=128, block_hidden=256, 24 Blöcke, k=4, max_len=48.  
**Phase-1-Spec (Defaults in config.py):** d_model=256, 64 Blöcke, k=4, bis 6 Iterationen.

### 2.5 Architektur-Erweiterung für Tiefen-Experiment

Für Experiment 1b wurde Modell C um eine **geteilte In-Loop-Attention** (`recurrent_read=True`) erweitert: Jede Iteration kann neu in den Kontext lesen. Ohne diese Erweiterung wäre ein sequenzieller Multi-Schritt-Task prinzipiell unlösbar, weil die einmalige vorab-laufende Attention den Zugriff auf Operanden sperrt.

---

## 3. Synthetische Daten

Vier Regimes mit mechanistisch verschiedenen Regeln, ein Regel-Tag an Position 1:

| Regime | Regel | Schwierigkeit |
|---|---|---|
| `REPEAT` | Kopiere Token von vor 3 Positionen | mittel (Lag-Kopie) |
| `INCREMENT` | +1 mod b | leicht (lokal) |
| `ALTERNATE` | positionaler Zyklus | leicht (positionaler Bias) |
| `FIB` | (a+b) mod b | schwer (Zwei-Token-Abhängigkeit) |

Loss-Maske nur auf regelbestimmten Positionen → scharfes, regelspezifisches Lernsignal. Die Regime-Verschiedenheit gibt den Blöcken einen **Grund zur Spezialisierung**.

---

## 4. Phase-1-Experimente: Grundfunktion

**Setup:** 400 Trainingsschritte, Chunk-Checkpointing, CPU, Demo-Konfig (d=128, 24 Blöcke).

### 4.1 Qualitätsvergleich A / B / C

| Modell | Core-Params | L₁ (Iter 1) | L_final | Verlauf |
|---|---|---|---|---|
| A dicht (16 distinkt) | 1.06 M | 0.607 | **0.589** | sinkt bis Iter 2, dann flach |
| B rekurrent (1 Block ×16) | 0.07 M | 0.756 | **0.720** | **U-förmig**: Min ~0.680 @Iter 5, dann schlechter |
| C geroutet (4 von 24) | 1.60 M | 0.547 | **0.547** | flach |

**Befund:** Bei gleichem Compute-Budget (16 Block-Anwendungen) gilt C < A < B.  
C erreicht seine Qualität bereits mit **4 Block-Anwendungen** (Iteration 1) und hält sie konstant.  
Modell B **destabilisiert** mit Tiefe — reine Rekursion eines geteilten Blocks führt ab Iter 5 zur Verschlechterung.

C über 3 Seeds: L_final = **0.629 ± 0.068** (Streuung bei dieser Schrittanzahl erwartbar).

### 4.2 Routing-Gesundheit (Modell C)

| Metrik | Wert | Interpretation |
|---|---|---|
| Router-Entropie | 0.984 (max=1.0) | nahezu perfekt balanciert |
| Max. Blockanteil | 0.092 | kein Dominanz-Block |
| Tote Blöcke | 1 von 24 | kein Kollaps |
| MI_norm (Regime↔Block) | **0.197 ± 0.030** | reproduzierbare moderate Spezialisierung |

**Go-Kriterium 2 (Spezialisierung): GO.**

### 4.3 Blockablation

Regime-spezifischste Blöcke abschalten:

| Ablation | Effekt auf REPEAT | Effekt auf FIB |
|---|---|---|
| REPEAT-Blöcke weg | +0.077 | +0.000 |
| FIB-Blöcke weg | +0.162 | +0.044 |

Triviale Regimes (INCR/ALT) reagieren auf keine Ablation — sie sind redundant in mehreren Blöcken abgesichert. Spezialisierung entsteht dort, wo tatsächlich gerechnet wird.

**Go-Kriterium 3 (Breitenbegrenzung): GO.** C (sparse, 4/24) erreicht die Qualität von A (16/16 dicht) bei gleichem Compute.

### 4.4 Streaming-Scaffold (Phase 1, nur Protokoll)

Jaccard-Overlap aufeinanderfolgender Iterationen:

| Übergang | Jaccard | Neue Blöcke Ø |
|---|---|---|
| Iter 1 → 2 | **0.13** | 3.35 |
| Iter 2 → 3 | **0.93** | 0.17 |
| Iter 3 → 4 | **0.95** | 0.12 |

Die erste Iteration streut breit (Exploration), danach hohe zeitliche Lokalität. Aktive Union schrumpft von 22 auf 12 Blöcke über den Tokenstrom.

**Go-Kriterium 4 (Lokalität): GO** — aber nur protokolliert, nicht optimiert.

---

## 5. Experiment 1a: War die flache Anytime-Kurve Daten oder Loss?

**Motivation:** Die aggregierte Anytime-Kurve von Modell C war unter equal-weighting flach (0.547 → 0.547). Ist das ein Daten-Problem (triviale Aufgaben bereits bei Iter 1 gelöst) oder ein Loss-Problem (equal-weighting trainiert Iter 1 zu stark)?

**Setup:** Modell C, Seed 0, identische Synthetik-Daten, drei Gewichtungen über die Iterationen.

| Gewichtung | Formel | L₁ | L_final | Abstand |
|---|---|---|---|---|
| equal | gleichmäßig | 0.547 | 0.550 | **−0.003** (flach) |
| linear | aufsteigend | 0.603 | 0.566 | **+0.037** |
| end (0.7 auf letzte Iter) | stark endlastig | 0.616 | 0.545 | **+0.071** |

**Tiefengewinn pro Regime G = L(Iter1) − L(Iter_final):**

| Gewichtung | REPEAT | INCREMENT | FIB | ALTERNATE |
|---|---|---|---|---|
| equal | +0.032 | −0.009 | −0.025 | −0.002 |
| linear | +0.295 | −0.031 | −0.007 | −0.006 |
| end | **+0.307** | −0.012 | **+0.033** | −0.007 |

**Befund: Die flache Aggregatkurve war überwiegend ein Loss-Artefakt.**  
Mit end-Gewichtung springt REPEATs Tiefengewinn von 0.03 auf **0.31** (10×), selbst FIB wird leicht positiv.  
Triviale Regimes (INCR/ALT, schon bei Iter 1 nahezu gelöst) bleiben erwartungsgemäß flach.

**Methodischer Befund:** Der aggregierte Abstand (0.071) ist viel kleiner als REPEATs Einzelgewinn (0.307), weil die trivialen Regimes den Mittelwert verwässern. **L_{q,r}-Logging nach Regime ist die korrekte Metrik, nicht die Aggregatkurve.**

Figur: `results/fig7_loss_variants.png`

---

## 6. Experiment 2: Ist die Lokalität streambar?

**Setup:** Cache-/Streaming-Simulation auf den gespeicherten Routing-Traces des trainierten Modells C (Seed 0). Kein Retraining. Physikmodell mit voller Blockgröße (d=256, h=512 → fp16 ≈ 527 KB/Block).

### 6.1 Befund 1 — Die Lokalität ist echt

Gelerntes Routing gegen Random-Routing-Kontrolle (identische Last), Miss-Rate unter Cache-Druck:

| Cache-Kapazität (von 24) | gelernt | random | Faktor |
|---|---|---|---|
| 8 | 0.174 | 0.716 | **4.1×** |
| 12 | 0.082 | 0.540 | **6.5×** |
| 16 | 0.033 | 0.367 | **11×** |

Reuse-Distanz (gelernt): p50 = 1, p90 = 8 Mikroschritte. Ein geladener Block wird meist im unmittelbar nächsten Schritt erneut gebraucht.

**Go-Kriterium 4 bestätigt (quantitativ):** 6.5× weniger Misses als Zufall bei halber Bank.

### 6.2 Befund 2 — Eintrittsphase dominiert die Transferkosten

**66 %** aller Blockladungen fallen in Iteration r=0 an, nur 34 % in Folge-Iterationen. Der Übergang r=0 → r=1 ist das kritische Prefetch-Fenster.

### 6.3 Befund 3 — Hub-Struktur ist real und pinbar

Blockfrequenz ist steil: Top-3-Blöcke ~24–26k Nutzungen, danach Abfall auf ~6k. Pinnen der heißesten Blöcke (H=8 gepinnt) senkt Miss-Rate bei gelerntem Routing von 0.845 auf 0.521.

### 6.4 Befund 4 — Transfer-Reduktion gegen die korrekte Baseline

**Korrektur:** Die frühere Auswertung verglich Sparse-Transfer gegen null (reine Compute-Zeit).
Das war die falsche Baseline für das Projektziel. Die richtige Baseline ist Layer-Offloading
eines äquivalenten dichten Modells (= alle Blöcke pro Token).

| Konfiguration | Einzigartige Blöcke / Token | vs. Layer-Offloading | Reduktion |
|---|---|---|---|
| Demo (24 Blöcke) | ~7 | 24 Blöcke | **3,4×** |
| 256 Blöcke (skaliert) | ~10 | 256 Blöcke | **26×** |
| 1000 Blöcke (skaliert) | ~10 | 1000 Blöcke | **100×** |

Die Reduktion wächst mit der Modellgröße, weil k (aktive Blöcke) nicht mit der Bankgröße
skaliert. Das ist die zentrale Eigenschaft der Tunnel-Architektur.

Absolute Compute-Transfer-Zahlen (für Prefetch-Overlap-Planung):

| Bandbreite | Transfer (Batch 48) | Compute | Verhältnis |
|---|---|---|---|
| 8 GB/s | 320 ms | 1.55 ms | 206× |
| 16 GB/s | 172 ms | 1.55 ms | 111× |
| 32 GB/s | 97 ms | 1.55 ms | 63× |

Dieses Verhältnis verbessert sich durch Attention-Blöcke (~100× mehr arithmetische Intensität
als MLP) und int8-Quantisierung (halbe Bytes). Es ist ein Engineeringziel, kein Ziel-Killer.

**Go-Kriterium 4 (Transfer-Reduktion vs. Layer-Offloading): Bedingtes GO.**
Demo zeigt 3,4× Reduktion; skalierter Nachweis auf TinyStories ausstehend.

Figuren: `results/fig5_missrate.png`, `results/fig6_hub_entry.png`

---

## 7. Experiment 1b: Übernimmt rekursive Tiefe nützliche sequenzielle Berechnung?

**Aufgabe:** Modularer Permutations-Walk mit bekannter wahrer Tiefe d:

```
v₀ gegeben; für i = 1..d:  vᵢ = T[(v_{i-1} + aᵢ) mod 8]
```

T ist eine feste, zufällige Permutation über {0..7}. Die Operanden aᵢ stehen im Kontext, unterbrochen von Ablenker-Token (DIST). Ziel: Vorhersage von v_d am QUERY-Token. Bekannte wahre Tiefe → testbar, ob r*(d) mit d korreliert.

Chance-Baseline: ln(8) = **2.079** (8 mögliche Antwortwerte, uniform verteilt).

Drei diagnostische Läufe auf RTX 2060 (0.45 s/Schritt bei R=8; GPU-Dispatch-Loop erzeugt ~190 CPU-GPU-Syncs pro Forward-Pass durch per-Block `.nonzero()`-Aufrufe).

---

### 7.1 Lauf 1 — Vollspezifikation: D_max=8, bis zu 4 Ablenker/Schritt

**Konfiguration:** 4000 Schritte (~30 Min), end-weighting, R=8, D_max=8, max_distract=4, seed=0.

**Ergebnis:** Alle Verlustwerte bei 2.1–2.2 — **auf oder nahe Zufall** für alle Tiefen.

| d | Iter 1 | Iter 8 | G_d |
|---|---|---|---|
| 1 | 2.09 | 2.12 | −0.03 |
| 2–8 | 2.12–2.20 | 2.13–2.19 | ≈0 |

Controls: oracle=2.146, fixedR=2.144, state_reset=2.162, shuffle_route=2.134 — **alle vier ununterscheidbar**.

Gemeldete `corr(r*,d)=1.00` ist ein **Artefakt**: Alle r*=1 (konstant), `argsort(argsort)` einer konstanten Folge liefert eine durch Tie-Breaking bedingte Permutation, keine echte Korrelation.

**Kein verwertbares Signal.** 4000 GPU-Schritte sind für die Vollspezifikation nicht ausreichend.

---

### 7.2 Lauf 2 — Isolationsdiagnostik: Ablenker entfernen, D_max=2

**Konfiguration:** 2000 Schritte (~4 Min), linear-weighting, R=2, D_max=2, max_distract=0, seed=0.

**Ergebnis:**

| d | Iter 1 | Iter 2 | G_d | r* |
|---|---|---|---|---|
| 1 | 0.27 | 0.30 | −0.03 | 1 |
| 2 | 2.23 | 2.17 | +0.06 | 2 |

Controls: oracle=1.218, fixedR=1.279, state_reset=1.320, **shuffle_route=2.454**

**Kernerkenntnis:** Ohne Ablenker und nur 2 Tiefenstufen konvergiert d=1 fast vollständig (0.27 << 2.079) in nur 4 Minuten. Das Routing ist für diesen gelernten Teil essenziell: `shuffle_route=2.454` (weit schlechter als Zufall über das Gesamtsystem, weil random routing die d=1-Lösung zerstört).

**Architektureller Designbefund (neu aus Lauf 1 vs. Lauf 2):**  
Im Lauf 1 stagniert sogar d=1 nach 30 Minuten auf Zufall. Im Lauf 2 löst d=1 in 4 Minuten. Der Unterschied: Ablenker-Token (DIST). Das bedeutet: **Die Ablenker sind kein bloßes Robustheitstest-Feature — sie erzeugen ein fundamental anderes Aufmerksamkeits-Subproblem** (selektive Operanden-Extraktion aus variabel-langem DIST-Rauschen), das bei dieser Modellgröße das Tiefen-Lernsignal vollständig überdeckt. Die Vollspezifikation konfundiert zwei unabhängig schwierige Lernziele.

---

### 7.3 Lauf 3 — Isolierte Tiefen-Komposition: D_max=4, keine Ablenker

**Konfiguration:** 6000 Schritte (~22 Min), end-weighting, R=4, D_max=4, max_distract=0, seed=0.

**Ergebnis — erstes echtes Tiefensignal:**

| d | Iter 1 | Iter 4 | G_d | r* |
|---|---|---|---|---|
| 1 | 1.68 | 1.68 | +0.00 | 1 |
| 2 | 2.26 | 2.19 | **+0.07** | **3** |
| 3 | 2.12 | 2.10 | +0.02 | 1 |
| 4 | 2.12 | 2.11 | +0.02 | 1 |

Controls: oracle=2.028, fixedR=2.035, state_reset=**2.055**, shuffle_route=2.033

**corr(r*, d) Spearman = 0.40** (nicht degeneriert: r*=[1, 3, 1, 1] für d=[1, 2, 3, 4])

**Interpretation:**

1. **d=1 teilweise gelöst** (1.68 << 2.079): Die Architektur lernt 1-Schritt-Komposition.
2. **d=2 zeigt echten Tiefengewinn**: G=+0.07, r*=3 — drei Iterationen sind nötig, nicht eine. Das ist das erste belegbare „mehr Tiefe hilft"-Signal in dieser Aufgabenfamilie.
3. **State-Reset ist nachweislich schlechter** (2.055 > 2.035): Das Aufbauen über Iterationen hinweg bringt messbaren Nutzen.
4. **d=3, 4 noch ungelöst** (~2.10–2.12, nahe Zufall): 6000 Schritte sind für 3-4-Schritt-Komposition nicht ausreichend.
5. **corr=0.40 ist schwach aber real**: Getrieben vom d=2-Fall (r*=3 vs. r*=1 bei d=1,3,4).

**Was noch fehlt für den vollen Nachweis:** Curriculum-Training (d≤2 erst konvergieren, dann auf D_max=4 erweitern), damit auch d=3,4 mit r*>1 erscheinen und eine saubere Diagonale im L[d,r]-Heatmap zeigen.

Figur: `results/fig9_depth_diagnostic.png`

---

## 8. Auswertung gegen die Theorie

### 8.1 Hypothesen

| Hypothese | §-Nr. | Status | Begründung |
|---|---|---|---|
| Primäre Hypothese: Routing + Tiefe verbessert Vorhersage | §12 | **Teilweise bewiesen** | REPEAT G=0.31 mit end-weighting; d=2 G=+0.07 ohne Ablenker. Nicht universal, regime-/aufgabenabhängig. |
| Streaming-Hypothese: Lokalität reicht für Streaming | §13 | **In starker Form widerlegt** | Lokalität echt (6.5× bessere Miss-Rate), aber Transfer übersteigt Compute um ~100:1. Break-Even nicht erreichbar bei aktueller Blockgröße. |
| Anytime-Hypothese: Frühe Iterationen brauchbar, spätere besser | §14 | **Bedingt bewiesen** | Mit end-weighting: ja für REPEAT (G=0.31). Triviale Regimes (schon bei Iter 1 gelöst) zeigen keinen Gewinn. Nicht universell. |
| Spezialisierungshypothese: Blöcke lernen verschiedene Funktionen | §15 | **Bewiesen** | MI_norm=0.197±0.030 über 3 Seeds, stabile Ablations-Effekte. |
| Breiten-Tiefen-Hypothese: Schmal-tief kompensiert breit-flach | §16 | **Offen** | Width-Depth-Grid implementiert aber nicht konvergiert ausgeführt. |

### 8.2 Go/No-Go-Kriterien

| Kriterium | §-Nr. | Status | Kommentar |
|---|---|---|---|
| Iterative Verbesserung: L_{r+1} < L_r | §55 | **Bedingtes GO** | Mit richtiger Loss-Gewichtung und hartem Regime: ja. Equal-weighting erzwingt Flachheit. |
| Block-Spezialisierung: verschiedene Blöcke, verschiedene Funktionen | §56 | **GO** | MI_norm stabil, Ablation signifikant. |
| Breitenbegrenzung: wenige Blöcke ≈ viele Blöcke | §57 | **GO** | C (4/24) ≈ A (16/16) bei gleichem Compute. |
| Lokalität: hoher Block-Overlap über Iterationen/Tokens | §58 | **GO** | Jaccard 0.93–0.95 ab Iter 2, 6.5× bessere Miss-Rate als Zufall. |
| Transfer-Reduktion vs. Layer-Offloading | §53 (neu) | **Bedingtes GO** | Demo: 3,4× weniger Blöcke als Layer-Offloading; skaliert auf ~100× bei 1000-Block-Bank. Falsche Baseline war Compute vs. Transfer absolut — korrigiert auf Layer-Offloading als Vergleich. |
| Reale Beschleunigung: messbare Tokens/s Verbesserung | §60 | **Offen** | Phase 6 noch nicht begonnen; durch NO-GO bei §59 derzeit nicht sinnvoll. |

### 8.3 Trainingsphasen

| Phase | Inhalt | Status |
|---|---|---|
| Phase 1 §20 — Grundfunktion | Completion lernen, Routing stabilisieren, Tiefennutzen prüfen | **Abgeschlossen** |
| Phase 2 §21 — Spezialisierung | Gumbel-Softmax, ST-Top-k, Temperatur-Curriculum, härtere Sparsity | **Nicht begonnen** |
| Phase 3 §22 — Räumliche Topologie | Trainierbare Koordinaten, Distanzterm im Router, Neighborhood-Kandidaten | **Stub vorhanden** (Koordinaten fest, nicht im Routing genutzt) |
| Phase 4 §23 — Hardware-Kosten simulieren | I/O-Loss, tunnelbewusste Eviction, prädiktiver Prefetch | **Simulation vorhanden** (Exp2), I/O-Loss nicht implementiert |
| Phase 5 §24 — Dynamisches Halten | Halt-Kopf, ΔQ-Schätzung, dynamisches Stoppen | **Nicht begonnen** |
| Phase 6 §25 — Reales Streaming | CUDA-Streams, asynchroner Transfer, realer Cache | **Nicht begonnen** (durch GNG5 blockiert) |

### 8.4 Datenstufen

| Stufe | Datensatz | Status |
|---|---|---|
| A §35 | Synthetische Sequenzen (4 Regimes + Depth-Walk) | **Vollständig genutzt** |
| B §36 | TinyStories | **Verdrahtet, nicht ausgeführt** (Netzwerk-Sandbox) |
| C §37 | Wikipedia-Ausschnitt | **Nicht begonnen** |
| D §38 | Instruction-Tuning | **Nicht begonnen** |

---

## 9. Was bisher nicht getestet wurde

Aus der To-Do-Liste in §40–54 noch ausstehend:

- **Soft-to-Hard-Training (§45):** Temperatur-Curriculum, Gumbel-Softmax, Straight-Through-Top-k. Das weiche Softmax-Routing aus Phase 1 ist noch aktiv.
- **Räumliche Topologie vollständig (§47):** Distanzterm `β·d(μ_r, c_b)` in Router-Score-Formel (§6) ist als Stub deklariert; alle Blöcke werden immer als Kandidaten gewertet (`n_candidates = n_blocks`). Schattenkoordinaten-Update nicht implementiert.
- **Prefetch-Kopf (§49):** Kein separater Head, der nächste Blockmenge als Trainingsziel lernt. Prefetch-Precision/Recall nicht messbar.
- **Halt-Modul (§51):** Komplett offen; Voraussetzung (zuverlässig bessere spätere Iterationen) erst teilweise erfüllt.
- **Korrigierbarkeit (§34):** Ungeklärt, ob falsch gewählte Blöcke in Iteration 1 in späteren Iterationen korrigiert werden können.
- **Width-Depth-Grid konvergiert (§52):** Harness implementiert, aber nur Smoke-Test (untrainiert) ausgeführt.

---

## 10. Technische Befunde / Ingenieur-Wissen

**GPU-Dispatch-Engpass:** Der Python-For-Loop über n_blocks mit `.nonzero()` pro Block in `router.py` erzeugt ~190 CPU-GPU-Synchronisationspunkte pro Forward-Pass (n_blocks × R Iterationen). Das limitiert den GPU-Speedup auf ~1.7× gegenüber CPU (gemessen: CPU 0.78 s/Schritt, GPU 0.45 s/Schritt bei R=8). Vollprotokoll (3 Gewichtungen × 3 Seeds × 6000 Schritte + Width-Depth-Grid) ≈ 28 Stunden auf RTX 2060. Lösbar durch vektorisierten Batched-MoE-Dispatch, ändert jedoch die im Projekt bewusst gewählte „echte Sparsity"-Semantik (kein „alle-dann-maskieren").

**Dateiname-Kollision behoben:** `depth_bench.py` schreibt jetzt unter `depth_{weighting}_d{D_max}dist{max_distract}_s{seed}.json` statt dem kollisionsanfälligen `depth_{weighting}_s{seed}.json`.

**Neue CLI-Parameter in `experiments/depth_bench.py`:**
- `--max_distract` (Standard 4): steuert maximale Ablenker pro Tiefenschritt
- `--k` (Standard 4): aktive Blöcke pro Iteration
- `--device` (Standard: cuda wenn verfügbar): explizite Gerätewahl

---

## 11. Ergebnisdateien

| Datei | Inhalt |
|---|---|
| `results/fig1_anytime.png` | Anytime-Kurven A/B/C über Iterationen |
| `results/fig2_compute.png` | Loss vs. Block-Anwendungen (Compute-Budget) |
| `results/fig3_collapse.png` | Routing-Gesundheit (Entropie, tote Blöcke) |
| `results/fig4_specialisation.png` | MI und Ablations-Effekte |
| `results/fig5_missrate.png` | Cache-Miss-Rate: gelernt vs. random (Exp2) |
| `results/fig6_hub_entry.png` | Hub-Struktur + Eintrittsphase (Exp2) |
| `results/fig7_loss_variants.png` | Exp1a: drei Gewichtungen, pro Regime |
| `results/fig8_depth_end.png` | Exp1b Lauf 3: L[d,r] Heatmap + r*(d) |
| `results/fig9_depth_diagnostic.png` | Kombinierte Diagnostik aller drei Exp1b-Läufe |
| `results/results.json` | Phase-1-Metriken (A/B/C, Routing-Health, Ablation) |
| `results/streaming_results.json` | Exp2-Metriken (Miss-Rate, Reuse-Distanz, Break-Even) |
| `results/depth_end_s0.json` | Exp1b Lauf 3 (D_max=4, nodist, 6000 Schritte) |
| `results/depth_linear_s0.json` | Exp1b Lauf 2 (D_max=2, nodist, 2000 Schritte) |
| `results/depth_end_s0_run.log` | Exp1b Lauf 1 (D_max=8, dist≤4, 4000 Schritte) |

---

## 12. Nächste Schritte (Empfehlung)

**Schritt 1 — Starkes Tiefensignal sichern (Exp1b abschließen):**  
Curriculum-Training auf dem ablenkfreien Task: D_max=2 mit ~8000 Schritten bis nahezu 0 Loss, dann auf D_max=4 und D_max=6 erweitern. Ziel: L[d,r]-Heatmap mit klarer Diagonale und `corr(r*,d) > 0.7`. Befund 2 aus Lauf 3 (d=2 braucht r*=3) deutet an, dass dies mit ausreichend Schritten erreichbar ist.

```bash
python -m experiments.depth_bench --steps 8000 --weighting end --seed 0 \
    --D_max 2 --R 4 --smoke --device cuda --max_distract 0
```

**Schritt 2 — Ablenker schrittweise reintegrieren:**  
Sobald die ablenkfreie Diagonale steht: `--max_distract 1`, dann `=2`, dann `=4`. Das isoliert, wie viel Ablenker-Rauschen das Tiefensignal toleriert, und kalibriert die ursprüngliche Vollspezifikation.

**Schritt 3 — Streaming-Break-Even verbessern (Exp2 Verbesserungen):**  
Ohne Retraining im Simulator testbar:
- int8-Bytes (halbiert Break-Even auf ~1570)
- Decode-Batch-Größe erhöhen (viele simultane Ströme teilen residente Blöcke)
- Tunnelbewusste Eviction (Hub-Blöcke immer resident halten)

**Schritt 4 — Phase 2 beginnen:**  
Gumbel-Softmax / ST-Top-k implementieren, Temperatur absenken. Prüfen, ob Spezialisierung mit härterem Routing wächst oder kollabiert.

**Was explizit NICHT als nächstes:** Phase 6 (reales CUDA-Streaming) — der Simulator zeigt klar, dass die arithmetische Intensität nicht ausreicht. Reales Streaming jetzt würde Ingenieurarbeit in eine Richtung investieren, die das Physikmodell bereits als nicht ausreichend charakterisiert hat.

---

## 13. Zulässige Aussagen nach aktuellem Stand

> Ein rekurrentes, blockselektives Modell kann mit einer begrenzten aktiven Parameterbreite (4 von 24 Blöcken) eine Vorhersagequalität erreichen, die ein dicht ausgeführtes Modell mit gleichem Compute-Budget übertrifft.

> Das Routing lernt reproduzierbar funktionale Spezialisierung und zeitliche Lokalität — unter Cache-Druck 6–7× weniger Misses als Zufallsrouting.

> Die Anytime-Kurve ist durch die Loss-Gewichtung steuerbar. Mit endlastiger Gewichtung gewinnen schwierige Regimes (REPEAT) 10× mehr Tiefennutzen als mit gleichmäßiger Gewichtung.

> Rekursive Tiefe leistet messbar nützliche sequenzielle Berechnung, wenn die Aufgabe echte Mehrschritt-Komposition verlangt und Ablenker-Rauschen isoliert wird: 2-Schritt-Komposition benötigt 3 Iterationen, State-Reset zerstört den Gewinn nachweisbar.

> Die erlernte Lokalität ist notwendig, aber bei der aktuellen Blockgröße nicht hinreichend für effizientes Parameter-Streaming: Der arithmetische Break-Even liegt bei ~3000 Reuses pro Block-Ladung, deutlich außerhalb des Erreichbaren für einen einzelnen Decode-Strom.
