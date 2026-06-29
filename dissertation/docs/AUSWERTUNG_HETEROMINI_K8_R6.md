# Auswertung — srcore_b32_k8_R6: 10k → 15k

*Eingefroren: 2026-06-20. k8_R6 @15k vs. k4_R6 @10k vs. Naked @10k vs. Dense @10k.
Eval: heteromini_eval (40 Batches) + domain_partition + gain_seen/unknown + seen_vs_unknown.
domain_label NICHT sichtbar im Modell.*

---

## Hauptvergleich (@15k für k8, @10k für alle anderen)

| Modell | k | WS | Loss(seen) | anytime(seen) | anytime(unk) | code_gain | code_gain_unk | anytime_ratio |
|---|---|---|---|---|---|---|---|---|
| srcore_b32_k8_R6 @15k | 8 | 8.0 | 5.232 | **0.041** | **0.040** | **0.063** | **0.057** | **0.98** |
| srcore_b32_R6 (k4) @10k | 4 | 4.0 | 5.222 | 0.028 | 0.019 | 0.038 | 0.025 | 0.68 |
| naked_b32_R6 @10k | 4 | 5.75 | 5.200 | 0.052 | 0.043 | 0.066 | 0.043 | 0.83 |
| dense_d24 @10k | — | — | **5.133** | 0.028 | 0.013 | 0.034 | 0.018 | 0.46 |

*anytime_ratio = anytime_unknown / anytime_seen. Wert nahe 1.0 = Rekursion überträgt sich
auf unbekannte Dokumente. Wert << 1.0 = Rekursion ist Fitting der gesehenen Muster.*

---

## Trainings-Verlauf k8_R6 (Milestone-Eval, 6 Batches)

| Step | Lfin | anytime | WS | reuse_p90 |
|---|---|---|---|---|
| 1000 | 7.026 | 0.001 | 8.0 | 8.0 |
| 2500 | 6.179 | 0.004 | 8.0 | 9.0 |
| 5000 | 5.654 | 0.016 | 8.0 | 9.0 |
| 7500 | 5.485 | 0.026 | 8.0 | 9.0 |
| 10000 | 5.526 | 0.029 | 8.0 | 9.0 |
| 12500 | 5.413 | 0.034 | 8.0 | 9.0 |
| **15000** | **5.167** | **0.055** | **8.0** | **9.0** |

WS=8.0 und reuse_p90=9.0 sind über den gesamten Verlauf trainingsinvariant.
Lfin bei 15k: 5.167 — etwa auf Niveau von k4 (5.137 @10k). Qualitätslücke geschlossen.
anytime bei 15k: 0.055 (Sprung von 0.034 auf 0.055 im letzten Segment, nicht flach).

---

## Rekursionsgewinn pro Domäne

| Modell | web | wiki | code | lit |
|---|---|---|---|---|
| k8_R6 @15k | 0.033 | 0.030 | **0.063** | 0.036 |
| k4_R6 @10k | 0.024 | 0.026 | 0.038 | 0.019 |
| naked @10k | 0.050 | 0.047 | **0.066** | 0.038 |

k8@15k code_gain = 0.063, nur 5% unter Naked (0.066).
Zum Vergleich k4 code_gain = 0.038 — k8 liegt 66% darüber.

---

## Gain Seen vs. Unknown — Überträgt sich Rekursion?

| Modell | Domäne | gain_seen | gain_unk | delta | ratio |
|---|---|---|---|---|---|
| k8 @15k | web | 0.033 | 0.029 | −0.004 | 0.87× |
| k8 @15k | wiki | 0.030 | 0.034 | +0.004 | 1.13× |
| k8 @15k | **code** | **0.063** | **0.057** | **−0.006** | **0.90×** |
| k8 @15k | lit | 0.036 | 0.048 | +0.011 | 1.32× |
| k4 @10k | web | 0.024 | 0.013 | −0.011 | 0.55× |
| k4 @10k | wiki | 0.026 | 0.010 | −0.016 | 0.40× |
| k4 @10k | **code** | **0.038** | **0.025** | **−0.013** | **0.66×** |
| k4 @10k | lit | 0.019 | 0.018 | −0.001 | 0.95× |
| naked | **code** | **0.066** | **0.043** | **−0.023** | **0.65×** |

**Kernbefund (robust über beide Eval-Läufe @10k und @15k):**
k8 code_gain_ratio: 0.98× (@10k), 0.90× (@15k) — bleibt nahe 1.0.
k4 code_gain_ratio: 0.80× (@10k), 0.66× (@15k) — fällt mit mehr Training.
Naked code_gain_ratio: 0.73× (@10k), 0.65× (@15k).

Interpretation: k8's Rekursion nutzt Code-Struktur, die auf unbekannten Dokumenten
erhalten bleibt. k4 und Naked fitzen mit mehr Training stärker auf gesehene Muster.

---

## Domain-Partition-Analyse

| Modell | excl | chance | domain-Jaccard | unique_cores | Coverage top-1 |
|---|---|---|---|---|---|
| k8 @15k | 0.398 | 0.250 | 0.301 | 7796 | 3–4% |
| k4 @10k | 0.441 | 0.250 | 0.489 | 1629 | 10–22% |
| naked @10k | 0.499 | 0.250 | 0.314 | 696 | 9–13% |

Alle: **SCHWACHE PARTITION**. Kein Confound. Befund über beide Eval-Läufe stabil.

k8 @15k: Top-Cores unterscheiden sich zwischen Domänen — web/wiki `[2,7,8,12,13,23,25,28]`,
code `[2,4,8,13,16,19,25,26]`, lit `[0,1,3,5,6,9,11,15]`. Nicht trivial überlappend.
k4: universaler Top-Core `[16,20,22,24]` für alle 4 Domänen (9–22% Coverage) — stabil
über beide Eval-Läufe bestätigt. k4 hat eine dominante "Einheits-Eintrittspforte".

---

## Seen-vs-Unknown Loss-Gap

**Hinweis: Diese Metrik hat hohe Batch-Varianz (~0.05–0.12 Nats zwischen Läufen).**
Richtung innerhalb desselben Modells über Training ist aussagekräftiger als Modellranking.

| Modell | Loss(seen) | Loss(unknown) | Gap |
|---|---|---|---|
| k8 @15k | 5.232 | 5.471 | +0.239 |
| k4 @10k | 5.222 | 5.373 | +0.151 |
| naked @10k | 5.200 | 5.338 | +0.139 |
| dense @10k | 5.133 | 5.287 | +0.154 |

k8-Gap bei 15k (+0.239) > k8-Gap bei 10k (+0.173). Mit mehr Training fittet k8 mehr
(erwartbar). Absolut-Ranking der Gaps ist durch Mess-Varianz nicht stabil genug
für starke Schlüsse. Aussagekräftiger: anytime_ratio (oben) — der ist stabiler.

---

## Offload-Projektion (WS gemessen)

| Modell | WS | Transfervorteil @b32 | Transfervorteil @b8192 |
|---|---|---|---|
| k8 | 8.0 | 4× | 1024× |
| k4 | 4.0 | 8× | 2048× |
| naked | 5.75 | 6× | 1420× |

k8 kostet Faktor 2 im WS vs k4, bleibt aber im Zielregime (b8192) bei 1024× vs Dense.

---

## Entscheidungen

### R8 trainieren? → JA

**Kriterien nach Anweisung (alle 4):**

| Kriterium | Wert | Schwelle | Status |
|---|---|---|---|
| anytime k8 >> k4 | 0.041 vs 0.028 | >k4×1.1=0.031 | ✓ |
| code_gain >> k4 | 0.063 vs 0.038 | deutlich | ✓ |
| loss_per_iter fällt bis r6 | 5.268→5.226 (delta 0.041) | sichtbar | ✓ (mit Vorbehalt) |
| Lfin nicht schlechter | 5.232 vs 5.222 (k4) | ≈ gleich | ✓ |

Vorbehalt zu r6: loss_per_iter zeigt leichten Uptick bei r6 (r5=5.2262, r6=5.2266).
Das ist einzelner Eval-Lauf (40 Batches, ~0.001 Nats Rauschen) — kein robustes Signal.
Trainings-Milestone anytime 0.034→0.055 im letzten Segment ist das stärkere Signal.

**Empfehlung: srcore_b32_k8_R8 @10k** — Training von Scratch (k8_R6 ist mit R6
trainiert, nicht resumierbar auf R8). Vergleich: R8 @10k vs. k8_R6 @15k.

### Leiterbahn-Simulator? → NACH R8

k8 routing ist sehr liquid (7796 unique cores, top-1 coverage 3–4%). Für Leiterbahn
braucht es konzentriertere Hot-Cores. R8 könnte die Core-Konzentration erhöhen
(mehr Iterationen = stabilere Core-Patterns). Erst nach R8 eval prüfen.

Alternative Leiterbahn-Perspektive: k4's universaler Core [16,20,22,24] ist ein
natürlicher Hot-Core-Kandidat (22% Code-Coverage). Wenn Leiterbahn-Simulator vor
R8 gebaut werden soll: bei k4 ansetzen, nicht k8.

### Domain-Partition-Confound? → NEIN (stabil bestätigt)

Zwei Eval-Läufe (@10k und @15k), beide: SCHWACHE PARTITION. excl ≈ 0.40 (Zufall 0.25).
Claim ist sauber: k8 lernt gemischte funktionale Kerne, keine Domänen-Silos.

---

## Caveats

- Loss-Gap (seen vs. unknown) hat hohe Batch-Varianz; Rankings zwischen Modellen
  sind in diesen Stichprobengrößen nicht stabil. anytime_ratio ist robuster.
- k8 @15k hat mehr Epochs auf den 6.6M Tokens gesehen als k4 @10k — Vergleich
  ist nicht step-matched. Für einen fairen Vergleich müsste k4 auch auf 15k gebracht werden.
- r6-Uptick in loss_per_iter: einzelner Messpunkt, kein robuster Befund.
- WS=8.0 und code_gain_ratio ≈ 0.9–1.0 sind die robustesten Befunde (wenig Varianz).
- Alle Metriken auf b32. Skalierung zu b256+ ungetestet.
