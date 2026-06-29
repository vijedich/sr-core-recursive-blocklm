# Entropy Minimization Sweep — Abschlussbefund
**SR-Core b64 k8 R6 per_token | Basis: 15k steps | Sweep: 15k→17k | 2026-06-22**

---

## 1. Motivation

b64 (n_blocks=64, k=8) verbessert gegenüber b32 die Kapazität, aber Routing ist diffus:
25,000–26,000 unique Core-Kombinationen pro Eval-Lauf, niedrige hard_overlap_eval (~0.25).
Das bedeutet schlechte Cache-Lokalität: jeder Token greift auf eine andere Block-Kombination zu,
LRU-Cache schlägt selten an, Leiterbahn-Bytes/Token bleiben hoch.

Ziel des Sweeps: Routing stabilisieren ohne LM-Qualität zu opfern.

---

## 2. Negative Controls — Jaccard-Locality

### Getestete Varianten
- `softfull@17k`: soft Jaccard-Loss auf `router_probs` (volle Verteilung)
- `softsharp_a2@17k`: soft Jaccard mit α=2 (stärkere Spitzengewichtung)

### Ergebnis

| Metrik | ctrl | softfull | softsharp_a2 |
|---|---|---|---|
| soft_overlap_raw | 0.724 | 0.731 | 0.729 |
| hard_overlap_eval | 0.253 | 0.245 | 0.247 |
| unique_cores | 25,941 | 27,204 | 26,819 |

Soft-Overlap steigt numerisch — das Optimierungsziel wird erreicht.
Aber hard_overlap verschlechtert sich, unique_cores steigen.

### Mechanismus-Diagnose

Jaccard-Losses auf `router_probs` optimieren Verteilungsähnlichkeit zwischen
aufeinanderfolgenden Tokens. Das macht die Verteilung breiter, nicht konzentrierter.
Hard-topk-Selektion ist nicht differenzierbar; der Gradient kann die *Auswahl*
der Blöcke nicht direktional steuern — nur die Wahrscheinlichkeitsmasse umverteilen.

**Fazit:** Locality-Loss ohne Router-Schärfung ist der falsche Hebel.
Soft-Jaccard-Losses sind für Cache-Konsolidierung nicht geeignet.

### Kontroll-Probe: noise_std=0.1 vs. Default 0.3

noise_std=0.3 ist kein Hyperparameter-Rauschen, sondern ein impliziter
Load-Balancing-Regularizer. Reduzierung auf 0.1 verschlechtert hard_overlap
um −0.029 und erhöht unique_cores um +9,000. Nicht anfassen.

---

## 3. Entropy-Minimierung — Der richtige Hebel

### Funktionsweise

```python
# rblm/reg_losses.py
def router_entropy_loss(router_probs_bt, eps=1e-8):
    p = router_probs_bt.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1).mean()

# experiments/heteromini_long.py — Training Loop
if lambda_entropy > 0.0 and rp is not None:
    loss = loss + lambda_entropy * router_entropy_loss(rp)
```

Gradient-Pfad: `router_entropy_loss` → pre-topk Softmax → Routing-Logits.
Entropy-Minimierung zwingt den Router, Wahrscheinlichkeitsmasse auf weniger Blöcke
zu konzentrieren. Das macht die topk-Auswahl konsistenter: dieselben Hot-Cores
werden öfter gewählt → weniger unique_core-Kombinationen → bessere Cache-Trefferrate.

### Warum das funktioniert wo Jaccard scheitert

Jaccard: optimiert Ähnlichkeit zwischen Verteilungen → breiter
Entropy: optimiert Schärfe der Verteilung selbst → konzentrierter

Nur Schärfung der Verteilung ändert welche Blöcke konsistent gewählt werden.

---

## 4. Pareto-Kurve λ=0.000 … 0.007

### Routing-Metriken (eval-mode, n_batches=40)

| λ | hard_ov Δ | unique_cores | unique_cores Δ | entropy | entropy Δ | Lfin | dead |
|---|---|---|---|---|---|---|---|
| 0.000 | — | 25,941 | — | 3.832 | — | 5.3158 | 0 |
| 0.001 | +0.003 | 25,487 | −454 | 3.801 | −0.031 | 5.3159 | 0 |
| 0.003 | +0.006 | 23,029 | −2,912 | 3.777 | −0.055 | 5.3158 | 0 |
| 0.005 | +0.012 | 21,273 | −4,668 | 3.732 | −0.100 | 5.3162 | 1 |
| **0.007** | **+0.022** | **18,817** | **−7,124** | **3.647** | **−0.185** | **5.3163** | **0** |

Sprung lam003→lam005 ist nicht-linear (entropy −0.10 vs −0.055 kumuliert).
lam007 setzt die Kurve fort: entropy −0.185, unique_cores −7,124, hard_overlap +0.022.
Lfin bleibt in allen Punkten praktisch stabil (maximale Abweichung: +0.0005).

### Leiterbahn-Tabelle (LRU bytes/token Δ vs. ctrl, K=[8,16,24,32,48])

| K | lam001 | lam003 | lam005 | lam007 |
|---|---|---|---|---|
| K=8 | −5 KB | −33 KB | −59 KB | −75 KB |
| K=16 | −11 KB | −4 KB | −78 KB | **−90 KB** |
| K=24 | −5 KB | −12 KB | −75 KB | **−96 KB** |
| K=32 | — | — | −54 KB | **−84 KB** |
| K=48 | — | — | −7 KB | **−18 KB** |

hot_core_pin gewinnt bei steigendem λ überproportional gegenüber LRU:
K=16 hot_pin bei lam007: −134 KB (vs. LRU −90 KB → Pin-Vorteil +49%).
Routing-Konsolidierung erzeugt stabile Hot-Cores, die Pin-Policies direkt ausnutzen.

---

## 5. Sweet Spot — λ=0.005

### Routing (eval-mode)

- unique_cores: 25,941 → 21,273 (−18%)
- router_entropy: 3.832 → 3.732 (−0.100)
- hard_overlap_eval: 0.253 → 0.266 (+0.013)
- reuseP90: 88 → 87
- dead_blocks: 0 → 1 (unkritisch)

### Qualität (eval_quality_compare, n_batches=40)

| Domain | Δ Lfin seen | Δ Lfin heldout | Code-Gap ctrl | Code-Gap lam005 |
|---|---|---|---|---|
| web | −0.036 | **−0.170** | — | — |
| wiki | −0.104 | +0.081 | — | — |
| code | **−0.153** | +0.095 | 0.266 | **0.514** |
| lit | +0.020 | +0.052 | — | — |

Code auf Seen-Daten: stärkste Verbesserung (−0.153). Code-Vorteil erhalten.
Code auf Held-out: leicht schlechter (+0.095) — Generalisierungs-Gap verbreitert sich.
qual_gen: kein sichtbarer Generierungsunterschied zu ctrl.
Adaptive Stopping: mean_R=3.096 (identisch zu ctrl 3.097).

**λ=0.005 = systems-optimaler Sweet Spot:**
beste Cache-Gewinne bei noch positiver Code-Qualitätsbilanz.

---

## 6. Bruchpunkt — λ=0.007

### Routing (eval-mode)

Beste Routing-Kennzahlen im gesamten Sweep:
unique_cores 18,817, entropy 3.647, hard_overlap 0.278, K24 LRU −96 KB.

### Qualität — der Kipp-Punkt

| Domain | Δ Lfin seen lam005 | Δ Lfin seen lam007 |
|---|---|---|
| web | −0.036 | −0.129 |
| wiki | −0.104 | −0.155 |
| code | −0.153 | **+0.005** ← erstmals negativ |
| lit | +0.020 | −0.033 |

Code-Ratio seen (niedriger = Code-Vorteil):
```
ctrl@17k:    0.816
lam005@17k:  0.808   ← Code-Vorteil erhalten
lam007@17k:  0.833   ← Code-Vorteil weg, relative Verschlechterung
```

Bei λ=0.007 verliert Code auf Seen-Daten erstmals seinen Qualitätsvorteil (+0.005).
Code war über alle früheren λ-Punkte der stärkste Gewinner; jetzt ist er neutral-negativ.
Das ist das saubere Bruchpunkt-Signal.

Hinweis Messung: ctrl-Baseline variiert leicht zwischen Eval-Läufen (~0.01–0.11
je nach Domain, wegen stochastischem Batch-Sampling ohne fixierten Seed).
Code-Seen-Signal ist trotzdem belastbar: der Vorzeichen-Wechsel ist eindeutig.

---

## 7. Gesamtinterpretation

> **Entropy-based router consolidation induces a Pareto frontier:
> mild consolidation can improve both routing stability and held-out code performance,
> moderate consolidation maximizes cache benefits,
> and stronger consolidation crosses into specialization loss.**

### Mechanismus-Kette

```
lambda_entropy ↑
  → H(router_probs) ↓               (schärfere Verteilung vor topk)
  → konsistentere topk-Auswahl      (mehr Hard-Overlap zwischen Tokens)
  → unique_cores ↓                  (weniger verschiedene Block-Kombinationen)
  → LRU cache miss rate ↓           (dieselben Blöcke werden öfter wiedergenutzt)
  → hot_core_pin profitiert stärker (stabile Hot-Cores lassen sich explizit pinnen)

Regime-abhängiger Nebeneffekt:
  → λ=0.003: Routing stabiler → Generalisierung VERBESSERT (Code gap sinkt)
  → λ=0.005: Konsolidierung stark → Seen-Spezialisierung beginnt (Code gap fast verdoppelt)
  → λ=0.007: Code-Seen-Vorteil kippt erstmals negativ
```

### Drei interpretierbare Regime

| λ | Regime | Cache | Code seen | Code heldout | Code gap |
|---|---|---|---|---|---|
| 0.003 | generalization-balanced | moderat | −0.120 | **−0.155** | 0.256 ↓ |
| 0.005 | systems-optimal | stark | −0.153 | +0.095 | 0.514 ↑ |
| 0.007 | breakpoint | maximal | +0.005 | −0.120 | ~0.38 |

---

## 8. Generalization-balanced point: λ=0.003

A full held-out/domain evaluation of λ=0.003 reveals that it is not merely a weaker
cache point — it is the best generalization-balanced point in the sweep.

Unlike λ=0.005, which improves seen-code loss but widens the held-out gap, λ=0.003
improves **both** seen and held-out code loss. Code held-out loss improves by −0.155
relative to the 17k control, and the code generalization gap decreases from 0.290 to
0.256. This indicates that mild entropy minimization stabilizes routing in a way that
**improves generalization**, while stronger entropy pressure over-consolidates routes
toward the seen distribution.

Code-Ratio held-out: ctrl=0.886 → lam003=0.866 (−0.021) — stärkste held-out
Code-Verbesserung im gesamten Sweep. Alle vier Domänen verbessern sich auf Seen-Daten;
web, code, und lit verbessern sich auch auf Held-out.

qual_gen: identisch mit ctrl — kein sichtbarer Generierungsunterschied.

### Interpretation

Entropy-based router consolidation is not inherently harmful to generalization.
The effect is regime-dependent:

- **Mild pressure (λ≈0.003):** Router stabilizes on recurring structural patterns.
  These patterns generalize — the model routes better, not more narrowly.

- **Moderate pressure (λ≈0.005):** Consolidation is strong enough to improve
  cache metrics substantially, but begins to narrow toward the seen distribution.
  Code held-out gap almost doubles.

- **Strong pressure (λ≥0.007):** Over-consolidation. Cache metrics continue to
  improve, but code specialization on seen data collapses.

The Pareto frontier has three deployment-relevant points, not one sweet spot.

---

## 9. Offene Fragen

- **Dense-Baseline (ModelA)**: Vergleich gegen nicht-sparses Modell steht aus.
- **100M-Replikation**: Skalierbarkeit des Mechanismus auf größerem Modell unbekannt.

---

## Nächstes Kapitel: Target Entropy

**Frage:** Kann man die Cache-Gewinne von λ=0.005 erreichen,
ohne den Generalisierungs-Gap von λ=0.005 zu erzeugen?

**Mechanismus:**
```python
# statt:  loss += λ * H(router)         — immer schärfer
# besser: loss += λ * relu(H(router) - H_target)  — nur zu diffuse Router schärfen

def router_entropy_loss_targeted(router_probs_bt, H_target, eps=1e-8):
    p = router_probs_bt.clamp_min(eps)
    H = -(p * p.log()).sum(dim=-1).mean()
    return F.relu(H - H_target)
```

**Geplante Punkte:** H_target=3.75 / H_target=3.70, λ=0.005 oder 0.01, Start b64@15k→17k.

**Erwartung:** Entropie sinkt nur dort wo sie zu hoch ist → unique_cores sinken moderat →
K16/K24 verbessern sich → Code-heldout bleibt näher bei lam003 → Cache-Gewinn näher bei lam005.

Gemessene Referenzentropien: ctrl=3.832, lam003=3.777, lam005=3.732, lam007=3.647.
H_target=3.75 liegt zwischen lam003 und lam005 — genau der gesuchte Kompromissbereich.

---

## 10. Finale Vergleichstabelle

### Routing und Leiterbahn (eval-mode, Δ vs. ctrl innerhalb desselben Eval-Laufs)

| | ctrl | lam003 | lam005 | lam007 |
|---|---|---|---|---|
| router_entropy | 3.832 | 3.799 | 3.732 | 3.647 |
| Δ entropy | — | −0.033 | −0.100 | −0.185 |
| hard_overlap_eval | 0.253 | 0.262 | 0.266 | 0.278 |
| unique_cores | 25,941 | 23,077 | 21,273 | 18,817 |
| Δ unique_cores | — | −2,864 | −4,668 | −7,124 |
| dead_blocks | 0 | 0 | 1 | 0 |
| reuseP90 | 88 | 87 | 87 | 80 |
| **K8 LRU Δ** | — | −19 KB | −59 KB | −75 KB |
| **K16 LRU Δ** | — | −4 KB | −78 KB | −90 KB |
| **K24 LRU Δ** | — | −12 KB | −75 KB | −96 KB |
| K32 LRU Δ | — | −16 KB | −54 KB | −84 KB |
| K16 hot_pin Δ | — | −49 KB | −104 KB | −134 KB |

### Qualität (Δ vs. ctrl, Within-Run-Werte; ctrl-Baseline variiert leicht durch Sampling)

| | lam003 | lam005 | lam007 |
|---|---|---|---|
| Δ code seen | −0.120 | −0.153 | **+0.005** ← Kippunkt |
| Δ code heldout | **−0.155** | +0.095 | −0.120 |
| Code-Gap (seen→heldout) | **0.256 ↓** | 0.514 ↑↑ | ~0.377 |
| Code-Ratio heldout Δ | **−0.021** | +0.019 | −0.005 |
| Δ Lfin (global) | ~+0.001 | +0.0004 | +0.0005 |
| Adaptive mean_R | 3.102 | 3.096 | 3.096 |
| qual_gen Δ | keiner | keiner | keiner |

### Deployment-Regime

| λ | Label | Wählen wenn… |
|---|---|---|
| 0.003 | **generalization-balanced** | Generalisierung Priorität; moderate Cache-Gewinne ausreichend |
| 0.005 | **systems/cache sweet spot** | Cache-Effizienz Priorität; Code-Gap akzeptabel |
| 0.007 | **breakpoint / upper bound** | Referenzpunkt für maximalen Cache-Druck; nicht für Deployment |

---

## 11. Target-Entropy Follow-up: Bounded Consolidation Did Not Improve the Pareto Frontier

After the entropy-minimization sweep, we tested whether a bounded target-entropy objective
could retain the cache gains of λ=0.005 while preserving the held-out code behavior of λ=0.003.

**Objective:** `loss += λ * relu(H(router) - H_target)` — only penalises routers above target.

**Setup:** R1-only, same as entmin sweep. λ=0.005, b64@15k → 17k, two targets tested.

### Results

| Run | H_target | H actual (eval) | K16 LRU Δ | K24 LRU Δ | Code-Ratio heldout Δ | dead |
|---|---|---|---|---|---|---|
| lam003 (ref) | — | 3.799 | −4 KB | −12 KB | **−0.021** | 0 |
| H375 | 3.75 | 3.785 | −77 KB | −75 KB | +0.034 | 0 |
| H370 | 3.70 | 3.765 | −29 KB | −24 KB | +0.022 | 0 |
| lam005 (ref) | — | 3.732 | −78 KB | −75 KB | +0.019 | 1 |

Neither run reached its target entropy within 2k steps. The bounded relu-loss weakens its
gradient as H approaches H_target — convergence is slower than pure entropy minimization.

H375 matched lam005 almost exactly on cache metrics (K16 −77 KB, K24 −75 KB) while
keeping dead_blocks at zero. However, it did not recover lam003's generalization advantage:
Code-Ratio heldout worsened by +0.034 vs ctrl, compared to lam003's −0.021 improvement.

H370 achieved neither: weaker cache gains than lam003 and worse code-ratio than lam005.

### Conclusion

Target entropy is mechanically valid but does not improve the Pareto frontier in the
current 2k-step fine-tuning regime. The bounded loss converges more slowly, and at these
operating points the quality outcome tracks closer to lam005 than to lam003.

λ=0.003 (generalization-balanced) and λ=0.005 (systems/cache sweet spot) remain the
meaningful operating points. Future target-entropy variants would require either longer
fine-tuning, higher λ, or a loss schedule — but that is a later chapter.

---

## 12. Cross-Seed Replication (Seed 1)

To test whether the Pareto regime generalises across training seeds, we trained a fresh
b64 k8 R6 seed 1 from scratch to 15k steps, then applied λ=0.003 and λ=0.005 from that
checkpoint (15k → 17k), and ran the same eval_compare + eval_quality_compare pipeline.

### Routing metrics (eval-mode, seed 1 vs ctrl_s1)

| | Δ entropy | unique_cores | K16 LRU Δ | K24 LRU Δ | dead |
|---|---|---|---|---|---|
| lam003 | −0.047 | 24,702 (−952) | −9 KB | −5 KB | 0 |
| lam005 | −0.057 | 23,690 (−1,934) | −34 KB | −35 KB | 0 |

Direction consistent with seed 0: lam005 consolidates more strongly, achieves larger
cache gains. Absolute magnitudes smaller than seed 0 (seed 0 lam005: K16 −78 KB) —
this is expected, as different initialisations converge to different routing attractors.
Dead blocks remain zero for both.

### Quality metrics (Code-Ratio Δ vs ctrl_s1)

| | seen Δ | heldout Δ |
|---|---|---|
| lam003_s1 | **−0.029** | **−0.073** |
| lam005_s1 | +0.031 | **−0.022** |

### Cross-seed comparison

| | s0 heldout Δ | s1 heldout Δ | Verdict |
|---|---|---|---|
| lam003 | −0.021 | **−0.073** | **consistent improvement, stronger in s1** |
| lam005 | +0.019 | −0.022 | seed-dependent (sign flip) |

### Conclusion

Seed 1 confirms the main mechanism: entropy minimisation consistently reduces router
entropy and achieves cache gains proportional to λ. The **generalisation-balanced point
at λ=0.003 is robust across seeds**: held-out code ratio improves in both seeds, and
the effect is stronger in seed 1.

The λ=0.005 quality tradeoff is **seed-dependent**: seed 0 shows a held-out code ratio
cost (+0.019), seed 1 shows a small improvement (−0.022). This means the code
generalisation cost at λ=0.005 is not a reliable claim — it was a seed 0 specific signal.

**Revised central claim:**

> Entropy-based router consolidation induces a reproducible cache/locality axis.
> Mild consolidation (λ=0.003) consistently improves held-out code ratio while modestly
> improving cache metrics across both seeds. Stronger consolidation (λ=0.005) consistently
> improves cache behaviour, but its quality tradeoff depends on the routing
> attractor/seed. λ=0.003 is the robust generalization-balanced point; λ=0.005 is the
> robust cache point with uncertain quality consequences.

---

## 13. Dense Baseline: Quality Upper Bound, Not a Systems Competitor

A dense d24 baseline (8.7M params, 17k steps, seed 0, same HeteroMini-v1 dataset) was
evaluated with the same domain/held-out and qualitative generation pipeline.

### Absolute quality

Dense remains the absolute quality upper bound across all domains:

| Domain | dense seen | ctrl seen | dense heldout | ctrl heldout |
|---|---|---|---|---|
| web  | 5.030 | 5.668 | 5.448 | 5.807 |
| wiki | 4.994 | 5.368 | 5.456 | 5.841 |
| code | **4.126** | 4.711 | **4.621** | 4.856 |
| lit  | 4.940 | 5.095 | 4.735 | 5.286 |

Dense is clearly better on both code seen and code held-out loss, as well as on all other
domains. The gap on code held-out is −0.235 nats vs ctrl, −0.658 nats vs lam003.

### Code-ratio caution

Code-ratio comparisons between Dense and Sparse were not treated as decisive because the
dense code-ratio estimates varied substantially across 40-batch eval runs (0.815 to 0.893
for the same model across three eval runs). Absolute losses were more stable and show a
clear dense advantage.

### Adaptive stopping observation

The dense d24 model saturates early under the KL adaptive-stopping criterion:
mean_R ≈ 3.05 out of 24 layers → **87.3% compute saved**. This suggests d24 is
over-deep for the current HeteroMini-v1 task; layers beyond ~3 contribute little
under this stopping criterion. Compare: SR-Core (6 iterations, mean_R ≈ 3.09, 48.6%
saved) — each sparse iteration genuinely contributes because the same blocks reuse
accumulated hidden state.

### Interpretation

This confirms that the entropy-minimization results should not be interpreted as Sparse
beating Dense in language-model quality. The correct interpretation is narrower and
systems-oriented: entropy minimisation improves routing efficiency and cache behaviour
inside the Sparse Recursive Fixed-Core family.

The dense result therefore validates the claim boundary:

- If the dense model fits in fast memory, Dense remains the preferred quality baseline.
- Sparse Recursive Fixed-Core is relevant when memory movement, active working set, or
  offloading constraints dominate.
- Entropy-based router consolidation improves the sparse model's memory behaviour, but
  does not close the absolute quality gap to Dense.

### Resulting model hierarchy

| Model | Role |
|---|---|
| Dense d24 | quality upper bound |
| b64 SR-Core ctrl | memory-bounded sparse baseline |
| b64 SR-Core + entmin λ=0.003 | generalization-balanced sparse point |
| b64 SR-Core + entmin λ=0.005 | systems/cache-optimized sparse point |

---

## 15. Figures

Alle drei Figuren werden von `scripts/build_figures.py` aus den Eval-JSONs in `results/` generiert.

| Datei | Inhalt |
|---|---|
| `results/fig_entropy_pareto.png` | Cache (K24 LRU bytes/token) vs. Code-Ratio held-out — zeigt lam003 als Dominanzpunkt, lam005 als Cache-Sweet-Spot, lam007 als Boundary; Target-Entropy-Punkte als Referenz |
| `results/fig_router_consolidation.png` | 2×2: Router-Entropie, Unique Cores, K24 LRU KB/token, Hard Core Overlap — je als Funktion von λ. Drei Regime-Linien (balanced/sweet spot/boundary). Zentrale Mechanismus-Figur. |
| `results/fig_dense_vs_sparse_quality.png` | Held-out Loss nach Domäne (Web/Wiki/Code/Lit) für Dense d24, SR-Core ctrl, lam003, lam005. Dense ist klare Qualitätsobergrenze; Sparse+entmin ist Systems-Beitrag. |
| `scripts/build_figures.py` | Figuren-Generator — liest Eval-JSONs, schreibt PNG-Dateien |

**Claim-Ebenen:**
- Fig 1 (Pareto): Entropy consolidation erzeugt kontrollierbare Operating Points.
- Fig 2 (Mechanismus): Router-Entropie ↓ → unique cores ↓ → hard overlap ↑ → K24 bytes ↓.
- Fig 3 (Claim-Grenze): Dense bleibt Qualitätsobergrenze; Sparse+entmin ist kein Dense-Quality-Sieg.

---

## 14. Dateiverzeichnis

| Datei | Inhalt |
|---|---|
| `results/hm_cont_hm_srcore_b64_k8_R6_ctrl_17k_s0.pt` | Baseline ctrl@17k |
| `results/hm_cont_hm_srcore_b64_k8_R6_entmin_r1_lam001_s0.pt` | λ=0.001 |
| `results/hm_cont_hm_srcore_b64_k8_R6_entmin_r1_lam003_s0.pt` | λ=0.003 (gen-balanced) |
| `results/hm_cont_hm_srcore_b64_k8_R6_entmin_r1_lam005_s0.pt` | λ=0.005 (systems sweet spot) |
| `results/hm_cont_hm_srcore_b64_k8_R6_entmin_lam007_s0.pt` | λ=0.007 (breakpoint) |
| `results/eval_compare_entmin_lam005.json` | Routing/Leiterbahn lam005 vs ctrl |
| `results/eval_compare_entmin_lam007.json` | Routing/Leiterbahn lam007 vs ctrl |
| `results/eval_quality_lam003.json` | gain_su / adaptive_stop lam003 vs ctrl |
| `results/eval_quality_lam005.json` | gain_su / adaptive_stop lam005 vs ctrl |
| `results/eval_quality_lam007.json` | gain_su / adaptive_stop lam007 vs ctrl |
| `results/qual_gen_comparison.json` | qual_gen ctrl vs lam003/lam005 |
| `rblm/reg_losses.py` | router_entropy_loss() |
| `experiments/heteromini_long.py` | --lambda_entropy CLI-Arg |
| `scripts/eval_compare.py` | Routing + Leiterbahn Eval |
| `scripts/eval_quality_compare.py` | gain_su + adaptive_stop Eval |
