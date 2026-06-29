# GESAMTSTAND BENCHMARK — HeteroMini, alle Modelle

*Erstellt: 2026-06-20. Seen = HeteroMiniData (Trainingsdaten). Unknown = heteromini_v1_heldout.*

*gain = loss_r1 − loss_rR (tiefer = besser als r1). anytime_ratio = gain_unknown / gain_seen.*


## Haupttabelle

| Modell | Step | Lfin(seen) | WS | anytime(eval) | code_gain_seen | code_gain_unk | code_ratio | anytime_ratio |
|---|---|---|---|---|---|---|---|---|
| dense_d24@2000 | 2000 | 6.4733 | — | 0.0332 | 0.0675 | 0.0659 | 0.9760 | 1.1780 |
| dense_d8@2000 | 2000 | 6.5410 | — | 0.0057 | 0.0084 | 0.0209 | 2.4880 | 1.5280 |
| srcore_b32_R2@2000 | 2000 | 6.5606 | 4.0 | 0.0001 | 0.0001 | 0.0001 | 1.0000 | 1.5000 |
| naked_b32_R6@2000 | 2000 | 6.5683 | 7.5 | 0.0216 | -0.0051 | 0.0085 | — | — |
| srcore_b64_R2@2000 | 2000 | 6.6587 | 4.0 | 0.0002 | -0.0001 | -0.0004 | — | — |
| srcore_b64_R6@2000 | 2000 | 6.6869 | 4.0 | 0.0001 | 0.0001 | 0.0001 | 1.0000 | 1.0000 |
| srcore_b32_R6@2000 | 2000 | 6.7096 | 4.0 | 0.0215 | 0.0552 | 0.0438 | 0.7930 | 1.0110 |
| naked_b32_R2@2000 | 2000 | 6.7556 | 4.1 | 0.0001 | 0.0005 | 0.0013 | 2.6000 | 2.9620 |
| srcore_b32_R2@5000 | 5000 | 5.6659 | 4.0 | 0.0001 | 0.0002 | 0.0001 | 0.5000 | 0.5000 |
| srcore_b64_R6@5000 | 5000 | 5.7900 | 4.0 | 0.0004 | -0.0008 | -0.0003 | — | 2.5350 |
| srcore_b32_R6@10000 | 10000 | 5.1469 | 4.0 | 0.0283 | 0.0379 | 0.0249 | 0.6570 | 0.6390 |
| naked_b32_R6@10000 | 10000 | 5.2189 | 5.7 | 0.0494 | 0.0659 | 0.0431 | 0.6540 | 0.7780 |
| dense_d24@10000 | 10000 | 5.3186 | 24.0 | 0.0250 | 0.0340 | 0.0181 | 0.5320 | 0.4340 |
| srcore_b32_k8_R6@15000 | 15000 | 5.2266 | 8.0 | 0.0419 | 0.0629 | 0.0565 | 0.8980 | 1.0540 |

## Anytime nach Domäne (Seen)

| Modell | Step | web | wiki | code | lit |
|---|---|---|---|---|---|
| dense_d24@2000 | 2000 | 0.0357 | 0.0280 | 0.0675 | 0.0441 |
| dense_d8@2000 | 2000 | 0.0084 | 0.0090 | 0.0084 | 0.0042 |
| srcore_b32_R2@2000 | 2000 | 0.0000 | -0.0003 | 0.0001 | 0.0002 |
| naked_b32_R6@2000 | 2000 | -0.0104 | -0.0051 | -0.0051 | -0.0096 |
| srcore_b64_R2@2000 | 2000 | -0.0001 | -0.0001 | -0.0001 | -0.0001 |
| srcore_b64_R6@2000 | 2000 | -0.0004 | -0.0004 | 0.0001 | -0.0000 |
| srcore_b32_R6@2000 | 2000 | 0.0142 | 0.0084 | 0.0552 | 0.0259 |
| naked_b32_R2@2000 | 2000 | 0.0004 | 0.0007 | 0.0005 | -0.0020 |
| srcore_b32_R2@5000 | 5000 | -0.0004 | -0.0000 | 0.0002 | -0.0001 |
| srcore_b64_R6@5000 | 5000 | -0.0004 | 0.0002 | -0.0008 | 0.0007 |
| srcore_b32_R6@10000 | 10000 | 0.0237 | 0.0258 | 0.0379 | 0.0188 |
| naked_b32_R6@10000 | 10000 | 0.0495 | 0.0474 | 0.0659 | 0.0375 |
| dense_d24@10000 | 10000 | 0.0321 | 0.0265 | 0.0340 | 0.0228 |
| srcore_b32_k8_R6@15000 | 15000 | 0.0329 | 0.0303 | 0.0629 | 0.0361 |

## Anytime nach Domäne (Unknown)

| Modell | Step | web | wiki | code | lit |
|---|---|---|---|---|---|
| dense_d24@2000 | 2000 | 0.0423 | 0.0444 | 0.0659 | 0.0426 |
| dense_d8@2000 | 2000 | 0.0094 | 0.0146 | 0.0209 | 0.0037 |
| srcore_b32_R2@2000 | 2000 | -0.0001 | -0.0005 | 0.0001 | 0.0004 |
| naked_b32_R6@2000 | 2000 | -0.0015 | 0.0004 | 0.0085 | -0.0113 |
| srcore_b64_R2@2000 | 2000 | -0.0001 | -0.0003 | -0.0004 | 0.0001 |
| srcore_b64_R6@2000 | 2000 | -0.0002 | -0.0005 | 0.0001 | 0.0003 |
| srcore_b32_R6@2000 | 2000 | 0.0118 | 0.0097 | 0.0438 | 0.0328 |
| naked_b32_R2@2000 | 2000 | 0.0008 | 0.0030 | 0.0013 | -0.0010 |
| srcore_b32_R2@5000 | 5000 | -0.0003 | -0.0001 | 0.0001 | -0.0001 |
| srcore_b64_R6@5000 | 5000 | -0.0002 | 0.0009 | -0.0003 | 0.0004 |
| srcore_b32_R6@10000 | 10000 | 0.0130 | 0.0103 | 0.0249 | 0.0179 |
| naked_b32_R6@10000 | 10000 | 0.0463 | 0.0355 | 0.0431 | 0.0291 |
| dense_d24@10000 | 10000 | 0.0138 | 0.0040 | 0.0181 | 0.0142 |
| srcore_b32_k8_R6@15000 | 15000 | 0.0287 | 0.0343 | 0.0565 | 0.0475 |

## Transferqualität (ratio_u_s = gain_unk / gain_seen)

*1.0 = Rekursion überträgt vollständig auf unbekannte Dokumente. <0.5 = hauptsächlich Fitting. — = gain nahe 0 bei 2k (nicht interpretierbar).*

| Modell | Step | web | wiki | code | lit | mean |
|---|---|---|---|---|---|---|
| dense_d24@2000 | 2000 | 1.19× | 1.59× | 0.98× | 0.97× | 1.178 |
| dense_d8@2000 | 2000 | 1.12× | 1.62× | 2.49× | 0.88× | 1.528 |
| srcore_b32_R2@2000 | 2000 | — | — | 1.00× | 2.00× | 1.500 |
| naked_b32_R6@2000 | 2000 | — | — | — | — | — |
| srcore_b64_R2@2000 | 2000 | — | — | — | — | — |
| srcore_b64_R6@2000 | 2000 | — | — | 1.00× | — | 1.000 |
| srcore_b32_R6@2000 | 2000 | 0.83× | 1.16× | 0.79× | 1.27× | 1.011 |
| naked_b32_R2@2000 | 2000 | 2.00× | 4.29× | 2.60× | — | 2.962 |
| srcore_b32_R2@5000 | 5000 | — | — | 0.50× | — | 0.500 |
| srcore_b64_R6@5000 | 5000 | — | 4.50× | — | 0.57× | 2.535 |
| srcore_b32_R6@10000 | 10000 | 0.55× | 0.40× | 0.66× | 0.95× | 0.639 |
| naked_b32_R6@10000 | 10000 | 0.94× | 0.75× | 0.65× | 0.78× | 0.778 |
| dense_d24@10000 | 10000 | 0.43× | 0.15× | 0.53× | 0.62× | 0.434 |
| srcore_b32_k8_R6@15000 | 15000 | 0.87× | 1.13× | 0.90× | 1.32× | 1.054 |

## Interpretation

**Robuste Befunde (> 1 Eval-Lauf bestätigt):**

- k8_R6@15k: code_ratio=0.90 — höchste Generalisierung aller trainierten Modelle
- k8_R6@15k code_gain_seen=0.063 ≈ Naked 0.066 — fast gleicher absoluter Gewinn
- Dense@10k: anytime_ratio=0.43 — schlechteste Generalisierung (tief ≠ rekursiv)
- Naked@10k: code_ratio=0.65 — mehr Gain als k4, aber schlechtere Generalisierung
- k4_R6@10k: code_ratio=0.66 — ähnlich wie Naked, aber niedrigerer Gain-Betrag

**2k-Smoke-Modelle:**
- Gains nahe 0 bei den meisten 2k-Modellen → Ratio nicht interpretierbar (als — markiert)
- Ausnahmen: dense_d24 (Schicht-Vorteil, nicht Rekursion) und srcore_b32_R6@2k
- srcore_b32_R6 hat bereits bei 2k erkennbaren code_gain=0.055 — Rekursion setzt früh ein

**Methodische Grenzen:**
- Lfin: hm_eval-JSONs (40 Batches) vs. Trajectory-Milestone (6 Batches) — 0.1–0.2 Nats Unterschied
- anytime_ratio für 2k-Modelle: nicht verwertbar (gain-Werte im Rauschen)
- k8@15k vs. andere @10k: nicht step-gematcht (k8 hat mehr Training gesehen)

---

*anytime_ratio < 1.0 = Modell fittet gesehene Daten. Nahe 1.0 = Rekursion nutzt Textstruktur (generalisiert).*