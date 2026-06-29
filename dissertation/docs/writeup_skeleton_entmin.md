# Write-up Skeleton: Cache-aware Router Consolidation in Sparse Recursive Fixed-Core Models

**Stand: 2026-06-23 | Basis: results_note_entmin_sweep.md (Sections 1–15)**

Dieses Dokument ist kein Fließtext, sondern ein Claim-Skeleton.
Pro Section: Zweck, Hauptclaim, Figur/Tabelle, Resultate, verbotene Formulierungen.

---

## Section 1 — Introduction / Motivation

**Zweck:** Kontext setzen; erklären warum Working-Set-Kontrolle bei Sparse-Modellen wichtig ist
und warum b64 das Problem schärfer stellt.

**Hauptclaim:**
> Sparse Recursive Fixed-Core (SR-Core) decouples bank size, working set, recursive depth, and
> per-token transfer: a 64-block bank with k=8 active blocks and R=6 recurrent passes maintains
> a fixed working set of 8 blocks regardless of bank size. This creates an attractive
> memory-bounded inference regime — if routing is sufficiently stable and local.

**Figur/Tabelle:** keine Pflicht-Figur; ggf. kurze Architektur-Schemabox (WS=k, R=Inferenzzeit).

**Resultate zu nennen:**
- SR-Core b64 k8 R6: n_blocks=64, k=8, R=6 → WS=8 (12.5% der Bank pro Token)
- Ohne Routing-Konsolidierung: 25,853 unique core combinations pro Eval-Lauf → schlechte LRU-Trefferrate

**Nicht erlaubt:**
- "Sparse beats Dense" oder "SR-Core ist qualitativ besser als Dense"
- Offloading-Zeiten versprechen (kein realer Timing-Benchmark vorhanden)
- b64 als final framen — es ist der experimentelle Träger dieses Sweeps

---

## Section 2 — Problem: Scaling the Block Bank Breaks Naive Cacheability

**Zweck:** Zeigen, dass eine große Bank allein nicht reicht — das Routing muss stabil genug sein,
damit ein LRU-Cache nutzbringende Trefferraten erreicht.

**Hauptclaim:**
> Increasing the block bank from b32 to b64 improves model capacity, but produces diffuse routing:
> ~26,000 unique core combinations per evaluation run, low hard overlap (~0.25), and K=24 LRU
> cache requirements of ~1,103 KB/token. Under memory-bounded inference, this diffuseness is
> the bottleneck — not bank size.

**Figur/Tabelle:**
- Fig 2 (Router Consolidation), λ=0 Datenpunkt als Ausgangslage
- Tabelle mit ctrl-Metriken: router_entropy=3.846, unique_cores=25,853, hard_overlap=0.251, K24=1,103 KB/token

**Resultate zu nennen:**
- ctrl b64 k8 R6 @17k: router_entropy=3.846, unique_cores=25,853
- hard_overlap_eval=0.251 → jeder Token wählt eine weitgehend neue Block-Kombination
- K16=1,852 KB/token, K24=1,103 KB/token → LRU-Cache muss groß sein, um nennenswerten Nutzen zu bringen

**Nicht erlaubt:**
- Behaupten, b32 sei "besser" als b64 ohne Konsolidierung (anderes Trade-off, keine direkte Wertung)
- Das Problem als b64-spezifisches Versagen framen — es ist ein allgemeines Problem bei großen Bänken

---

## Section 3 — Method: Entropy-based Router Consolidation

**Zweck:** Den Mechnanismus erklären: Entropie-Druck auf pre-topk Routerverteilungen schärft
die Auswahl; Jaccard-Locality-Ansätze scheitern, weil sie nicht auf die harte Top-k-Selektion
einwirken können.

**Hauptclaim:**
> Adding a scalar entropy loss `H(p) = −∑ p log p` on pre-softmax router probabilities directly
> pressures the selection distribution, reducing routing diffusion without modifying the sparse
> forward pass. Jaccard-based locality losses, by contrast, optimize soft distributional similarity
> but cannot steer the non-differentiable top-k selection — and empirically worsen hard overlap.

**Figur/Tabelle:**
- Code-Snippet: `router_entropy_loss()` aus `rblm/reg_losses.py`
- Tabellarische Gegenüberstellung: Jaccard-Loss Wirkung (soft_overlap ↑, hard_overlap ↓, unique_cores ↑)
  vs. Entropy-Loss Wirkung (router_entropy ↓, hard_overlap ↑, unique_cores ↓)

**Resultate zu nennen:**
- Jaccard softfull@17k: hard_overlap ↓ 0.253→0.245, unique_cores ↑ 25,941→27,204
- Entropy lam003@17k: hard_overlap ↑ 0.251→0.262, unique_cores ↓ 25,853→23,077
- Mechanismus: Entropie-Druck macht die Verteilung spitzer → Top-k greift stabiler auf dieselben Blöcke

**Nicht erlaubt:**
- Entropy-Loss als neue oder unbekannte Technik framen — es ist eine Standard-Entropie-Regularisierung,
  neu ist die Anwendung auf pre-topk Router in Sparse-Recursive-Modellen
- Behaupten, Jaccard-Losses seien generell nutzlos (sie optimieren ein anderes Ziel)

---

## Section 4 — Negative Controls

**Zweck:** Systematisch zeigen, warum alternative Ansätze (Jaccard-Locality, Noise-Reduktion) nicht
das gewünschte Verhalten erzeugen. Dient als Absicherung der Method-Section.

**Hauptclaim:**
> Gate-only losses, softfull, and softsharp_a2 all improve soft distributional overlap metrics
> while degrading hard top-k locality. Reducing routing noise (noise_std=0.3→0.1) similarly
> worsens overlap. These controls confirm that improving hard-topk cache behavior requires
> sharpening the pre-topk selection distribution, not constraining the selection pattern directly.

**Figur/Tabelle:**
- Negative-Control-Tabelle: ctrl | softfull | softsharp_a2, Metriken: soft_overlap, hard_overlap, unique_cores

| Metrik | ctrl | softfull | softsharp_a2 |
|---|---|---|---|
| soft_overlap_raw | 0.724 | 0.731 | 0.729 |
| hard_overlap_eval | 0.253 | 0.245 | 0.247 |
| unique_cores | 25,941 | 27,204 | 26,819 |

**Resultate zu nennen:**
- Noise-Reduktion (noise_std=0.3→0.1): hard_overlap ↓ −0.029, unique_cores ↑ +9,000
- noise_std=0.3 ist impliziter Load-Balancing-Regularizer, kein Hyperparameter-Rauschen

**Nicht erlaubt:**
- Jaccard-Losses als "falsch" labeln — sie sind korrekte Optimierung eines falschen Ziels
- noise_std als tunable Hyperparameter für Cache-Optimierung vorschlagen

---

## Section 5 — Pareto Sweep: Entropy Pressure Creates Controllable Operating Points

**Zweck:** Die fünf Punkte der λ-Kurve beschreiben, drei Regime herausarbeiten, Pareto-Frontier
zwischen Cache-Effizienz und Code-Generalisierung zeigen.

**Hauptclaim:**
> Across λ ∈ {0, 0.001, 0.003, 0.005, 0.007}, entropy pressure monotonically reduces router
> entropy, unique core combinations, and K=24 LRU bytes/token, while increasing hard overlap.
> Three operating regimes emerge: a generalization-balanced point (λ=0.003, dominates ctrl on
> both cache and code-ratio axes in seed 0), a cache sweet spot (λ=0.005, −77 KB/token vs ctrl),
> and a boundary (λ=0.007, further cache gain with degraded seen-code quality).

**Figur/Tabelle:**
- **Fig 1 (Pareto):** K24 vs. code-ratio-heldout, alle Punkte inkl. Target-Entropy-Referenz
- **Fig 2 (Router Consolidation):** λ → router_entropy / unique_cores / K24 / hard_overlap

**Resultate zu nennen (Seed 0):**

| λ | router_entropy | unique_cores | K24 (KB/tok) | hard_overlap | code_ratio_HO |
|---|---|---|---|---|---|
| 0.000 (ctrl) | 3.846 | 25,853 | 1,103 | 0.251 | 0.886 |
| 0.003 | 3.799 | 23,077 | 1,085 | 0.262 | 0.866 |
| 0.005 | 3.732 | 21,273 | 1,020 | 0.266 | 0.877 |
| 0.007 | 3.647 | 18,817 | 1,009 | 0.278 | 0.884 |

- lam003 dominiert ctrl: K24 ↓ (−18 KB) UND code-ratio ↓ (−0.020) — kein Tradeoff in Seed 0
- Echter Pareto-Tradeoff beginnt erst bei lam003→lam005: K24 ↓ −65 KB, code-ratio ↑ +0.011

**Nicht erlaubt:**
- "lam003 ist generell besser als ctrl auf Code" — dieser Befund ist Seed-0-spezifisch
  (Seed 1 zeigt stärkere Verbesserung −0.073, aber das ist nicht Punkt hier)
- code-ratio als absolute Qualitätsmetrik framen — es ist ein relatives Verhältnis, 40-batch-Schätzung
- Target-Entropy-Punkte (H375/H370) als Pareto-Fortschritt darstellen — sie sind negative Referenz

---

## Section 6 — Cross-Seed Replication

**Zweck:** Robustheit des Mechanismus und der Hauptbefunde über zwei Trainingsläufe (Seeds 0 und 1)
prüfen. Separiert robust reproduzierbare Befunde von seed-spezifischen Signalen.

**Hauptclaim:**
> Entropy-based router consolidation induces a reproducible cache/locality axis across seeds:
> both seeds show consistent K16/K24 reductions for λ=0.005. Mild consolidation (λ=0.003)
> consistently improves held-out code ratio in both seeds. Stronger consolidation (λ=0.005)
> consistently improves cache metrics, but its code-ratio effect is seed-dependent (sign flip
> between seeds), indicating route-attractor sensitivity at this pressure level.

**Figur/Tabelle:**

| Metrik | lam003 s0 | lam003 s1 | lam005 s0 | lam005 s1 |
|---|---|---|---|---|
| K16 delta vs ctrl (KB/tok) | −18 | ~−18 | −96 | ~−96 |
| code-ratio-HO delta | −0.021 | −0.073 | +0.019 | −0.022 |

**Resultate zu nennen:**
- Cache-Achse (K16/K24): konsistent in beiden Seeds für lam003 und lam005
- lam003 code-ratio-HO: ↓ in s0 (−0.021) und s1 (−0.073) → robust
- lam005 code-ratio-HO: ↑ in s0 (+0.019) aber ↓ in s1 (−0.022) → Vorzeichenwechsel, NICHT robust

**Nicht erlaubt:**
- "lam005 hat Code-Generalisierungskosten" — das ist ein s0-spezifisches Signal
- Seed 1 lam003-Verbesserung (−0.073) als Hauptzahl nennen ohne s0 zu erwähnen
  (s1-Effekt ist größer, aber das Seed-gemittelte Bild ist das sichere Statement)
- n=2 Seeds als "robuste Replikation" im starken statistischen Sinn framen

---

## Section 7 — Dense Baseline: Quality Upper Bound

**Zweck:** Den Claim-Bereich von Sparse+entmin klar abgrenzen. Dense ist Qualitätsmaßstab;
Sparse+entmin konkurriert nicht auf dieser Ebene, sondern auf der Systems-/Memory-Ebene.

**Hauptclaim:**
> Dense d24 trained from scratch for 17k steps remains the quality upper bound across all
> held-out domains. The gap to the best sparse model (ctrl) is 0.24 nats on held-out code.
> Entropy-minimized sparse variants (lam003, lam005) are comparable to ctrl within evaluation
> noise. The contribution of entmin is therefore systems-oriented: improved cache behavior within
> the sparse family, not a step toward dense-quality recovery.

**Figur/Tabelle:**
- **Fig 3 (Dense vs Sparse):** Held-out Loss nach Domäne

| Modell | Lfin_HO web | Lfin_HO wiki | Lfin_HO code | Lfin_HO lit |
|---|---|---|---|---|
| Dense d24 | 5.448 | 5.456 | 4.621 | 4.735 |
| SR-Core ctrl | 5.807 | 5.841 | 4.856 | 5.286 |
| + lam003 | ~5.80 | ~5.84 | ~4.85 | ~5.29 |
| + lam005 | ~5.80 | ~5.84 | ~4.86 | ~5.30 |

*(lam003/lam005-Werte aus separaten Eval-Runs — Absolutwerte nur richtungsweisend, Deltas vs Dense aus Dense-paired Runs)*

**Resultate zu nennen:**
- Dense code heldout 4.621 vs ctrl 4.856: Δ=0.235 nats (aus gleicher Eval-Batch-Stichprobe → valides Delta)
- Dense vs lam003: Δ≈0.23–0.25 nats (code heldout)
- Dense vs lam005: Δ≈0.24–0.26 nats (code heldout)
- Adaptive Stopping: Dense d24 sättigt bei Layer ~3/24 (87% saved) → Dense ist over-deep für HeteroMini-v1;
  das relativiert den absoluten Qualitätsabstand nicht, aber erklärt das Effizienzverhalten

**Nicht erlaubt:**
- "Sparse+entmin verbessert Quality gegenüber ctrl" als starken Claim — Eval-Noise (~0.05 auf code-ratio)
  deckt die Deltas zwischen Sparse-Varianten weitgehend ab
- Dense adaptive-stopping Effizienz als "Dense-Schwäche" framen — es ist eine andere Design-Achse
- Cross-Run code-ratio-Vergleiche (sparse vs dense) ohne Noise-Vorbehalt

---

## Section 8 — Discussion

**Zweck:** Befunde einordnen, Mechanismus-Story schließen, implizite Annahmen benennen.

**Hauptclaim:**
> Cache-aware routing is possible but not free. Entropy pressure induces a Pareto frontier between
> routing diversity (beneficial for generalization) and routing locality (beneficial for cache
> efficiency). The sweet spot (λ=0.005) achieves consistent cache improvements at the cost of a
> seed-dependent quality effect; mild pressure (λ=0.003) is the more reproducible operating point.
> The mechanism acts cleanly on pre-topk distributions; bounded variants (target entropy) converge
> too slowly at the tested scale to improve upon the direct penalty.

**Zu adressieren:**
- Warum Jaccard scheitert, Entropie aber wirkt: Nicht-Differenzierbarkeit von Top-k
- Warum Target-Entropy (bounded loss) schlechter abschneidet: zu langsame Konvergenz in 2k Steps
- Working-Set-Geometrie: Routing-Konsolidierung schafft Bedingungen für effiziente LRU-Caches,
  aber kein realer Timing-Benchmark vorhanden (→ Next Work)
- Domäne Code profitiert am stärksten von Routing-Stabilität (höchster Rekursionsgewinn in früheren Runs)

**Nicht erlaubt:**
- Offloading-Speedup-Zahlen aus dem Offload-Sim als valides Timing darstellen (das ist eine Simulation,
  kein Prototyp)
- Behaupten, der Mechanismus sei auf Dense übertragbar

---

## Section 9 — Limitations

**Zweck:** Ehrlichkeit über Reichweite der Befunde. Diese Section schützt alle anderen Claims.

| Limitation | Konsequenz |
|---|---|
| Kleines Modell (~HeteroMini-Skala, b64 k8) | Routing-Geometrie kann sich bei 1B+ anders verhalten |
| HeteroMini-v1 Datensatz | Vier Domänen, ~6.6M Tokens — kein echtes Pretraining |
| 40-batch eval noise | ~0.05 Varianz auf code-ratio-heldout; kleine vertikale Abstände in Fig 1 nicht überinterpretieren |
| n=2 Seeds | Cross-Seed-Replikation ist Plausibilitätscheck, kein statistischer Nachweis |
| Kein realer Offload-Prototyp | Alle Cache-Metriken sind LRU-Simulation; kein Throughput/Latency gemessen |
| Target-Entropy negativ | Bounded loss bei λ=0.005, 2k Steps zu langsam; längeres Training oder höheres λ nicht getestet |
| Adaptive Stopping (KL-Threshold) | Dense d24 Effizienz (87% saved) ist HeteroMini-spezifisch, nicht verallgemeinerbar |

---

## Section 10 — Next Work

**Zweck:** Offene Fragen benennen, die aus den Befunden direkt folgen.

| Priorität | Thema | Motivation |
|---|---|---|
| 1 | Real offload timing prototype | Mechanismus ist bekannt, Speedup ist noch nicht gemessen |
| 2 | Timing simulator (B/s, Latenz) | Brücke zwischen LRU-Simulation und Prototyp |
| 3 | Global balance loss + entmin | Routing-Konsolidierung + Auslastungsverteilung gleichzeitig |
| 4 | Hierarchical routing | Coarse routing → fine selection, verringert unique-core explosion bei größeren Bänken |
| 5 | 100M-Skalierung | Routing-Geometrie bei realistischer Modellgröße validieren |
| 6 | Längeres Target-Entropy-Training | Bounded loss braucht möglicherweise 5k+ Steps statt 2k |

---

## Zentraler Claim-Schutz (gilt für alle Sections)

```
NICHT SAGEN:
  "Sparse beats Dense."
  "Entropy minimization improves language model quality."
  "lam005 has code generalization costs." (nur s0)
  "The LRU cache improvement translates directly to faster inference."

SAGEN:
  "Dense remains the quality upper bound;
   entropy-consolidated SR-Core improves the sparse model's memory behavior
   and creates controllable operating points under memory-bounded inference."

  "Mild consolidation (λ=0.003) consistently improves held-out code ratio across seeds;
   stronger consolidation (λ=0.005) consistently improves cache behavior,
   but its quality trade-off is seed-dependent."

  "The contribution of entmin is systems-oriented:
   reduced LRU working-set requirements within the sparse model family."
```

---

## Mapping: Section → Quelle in results_note_entmin_sweep.md

| Write-up Section | results_note Sections |
|---|---|
| 1 Motivation | Einleitung / Section 1 |
| 2 Problem | Section 1 (b64 Diagnose), Section 4 (Pareto ctrl-Punkt) |
| 3 Method | Section 3 (Entropy-Mechanismus) |
| 4 Negative Controls | Section 2 (Jaccard), Section 3 (noise_std) |
| 5 Pareto Sweep | Sections 4–8, Section 10 (Tabelle), Fig 1+2 |
| 6 Cross-Seed | Section 12, Tabelle |
| 7 Dense Baseline | Section 13, Fig 3 |
| 8 Discussion | Sections 7, 9 |
| 9 Limitations | Section 9, Messrauschen-Hinweise überall |
| 10 Next Work | Section 9 (Offene Fragen), next_steps.md |
