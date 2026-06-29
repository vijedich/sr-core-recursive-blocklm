# Auswertung — HeteroMini 5k-Lauf, Verlauf (hält die SR-Core-Geometrie?)

*Eingefroren: 2026-06-19. `experiments/heteromini_long.py` (5k-Segmente, fortsetzbare
Snapshots), `scripts/heteromini_long_matrix.py`. HeteroMini-v1 (6,6M Tokens, 4 Domänen),
seed 0, k=4, End-Gewichtung, Milestones 1k/2,5k/5k inline ausgewertet. Loss bei 5k noch
fallend (nicht geplateaut) — Snapshots erlauben +5k-Fortsetzung.*

## Verlaufstabelle (K=8 Cache, contiguous; rel = K8-Bytes/Token relativ zu Dense bzw. Naked)

| step | model | Lfin | WS | reuse_p90 | K8 KB | rel_dense | rel_naked | anytime | dead |
|---|---|---|---|---|---|---|---|---|---|
| 1000 | srcore_b32_R6 | 7.002 | 4.0 | 4 | 158.9 | 0.013 | 0.124 | 0.003 | 6 |
| 1000 | naked_b32_R6 | 6.946 | 7.8 | 24 | 1284 | 0.104 | 1.000 | 0.017 | 8 |
| 1000 | dense_d24 | 6.819 | 24 | 24 | 12348 | 1.000 | – | 0.120 | 0 |
| 1000 | srcore_b64_R6 | 7.217 | 4.0 | 4 | 42.0 | 0.003 | 0.033 | 0.001 | 49 |
| 1000 | srcore_b32_R2 | 7.009 | 4.0 | 21 | 422 | 0.034 | 0.329 | 0.001 | 5 |
| 2500 | srcore_b32_R6 | 6.082 | 4.0 | 4 | 295.7 | 0.024 | 0.198 | 0.008 | 8 |
| 2500 | naked_b32_R6 | 6.106 | 6.7 | 31 | 1494 | 0.121 | 1.000 | 0.008 | 0 |
| 2500 | dense_d24 | 5.996 | 24 | 24 | 12348 | 1.000 | – | 0.009 | 0 |
| 2500 | srcore_b64_R6 | 6.316 | 4.0 | 5 | 687 | 0.056 | 0.460 | 0.000 | 24 |
| 2500 | srcore_b32_R2 | 6.175 | 4.0 | 29 | 643 | 0.052 | 0.431 | 0.000 | 5 |
| 5000 | srcore_b32_R6 | 5.515 | 4.0 | 4 | 504.8 | 0.041 | 0.281 | 0.013 | 3 |
| 5000 | naked_b32_R6 | 5.559 | 6.3 | 48 | 1797 | 0.146 | 1.000 | 0.014 | 1 |
| 5000 | dense_d24 | 5.435 | 24 | 24 | 12348 | 1.000 | – | 0.009 | 0 |
| 5000 | srcore_b64_R6 | 5.790 | 4.0 | 5 | 790 | 0.064 | 0.439 | 0.000 | 19 |
| 5000 | srcore_b32_R2 | 5.666 | 4.0 | 36 | 844 | 0.068 | 0.469 | 0.000 | 2 |

## Go/No-Go @5k: GO bleibt stark — alle drei Kriterien erfüllt

- srcore_b32_R6 K8 = 0.041 von Dense → **24× besser** (>20× ✓)
- = 0.281 von Naked → **3,6× besser** (>3× ✓)
- Loss: srcore_b32_R6 (5.515) **schlägt Naked** (5.559); zu Dense nur 0.08 dahinter ✓

Mechanismus beim trainierten Modell sichtbar: srcore hält **reuse p90 = 4 konstant**, Naked
explodiert auf **48** → daher der Byte-Vorteil. **WS = 4 hält über alle Milestones** → die
Geometrie ist KEIN frühes Trainingsartefakt. Primäre Frage positiv beantwortet.

## Warnsignale (mild, gemischt)

- **Vorteil schrumpft mit Training:** srcore_b32_R6 vs Naked 8×→5×→3,6× (rel_dense 77×→42×→24×).
  Per-Token bleibt perfekt (WS=4, reuse=4), aber die Kern-Wahlen streuen über den Token-Strom
  breiter → ein GETEILTER K8-Cache sieht mehr distinkte Blöcke. Noch klar über Schwelle,
  Trendrichtung = Warnsignal. Beobachten.
- **b64 schwach:** dead 49→19 (Blöcke wachen auf) → K8 steigt → nur 16× < Dense (<20×). Größere
  Bank hilft fürs Offloading bei diesem Budget NICHT.
- Loss bei 5k noch fallend → +5k würde Quality verbessern UND klären, ob der Schrumpf-Trend
  stabilisiert oder weiterläuft.

## Empfehlung

GO ist bestätigt; vor dem realen RAM→VRAM-Prototyp wäre EIN +5k-Segment (von den Snapshots,
MAX_STEPS=10000) sinnvoll, um zu sehen, ob srcore_b32_R6 vs Naked bei ~3× stabilisiert oder
weiter abbaut. Bleibt es ≥3× → Prototyp klar gerechtfertigt.
