# Auswertung — CPU-Benchmark: dense vs. sparse (entscheidender Befund)

*Eingefroren: 2026-06-18. `scripts/cpu_benchmark.py`, CPU, 8 Threads, bs=8 seq=128
(1024 Tokens/Forward), Median über 15 Läufe. Kern-Zeit = ab h0, OHNE Deep-Supervision-
Readouts (für alle Modelle gleich gemessen — fair).*

## Frage

Lässt sich das gemessene kleine aktive Working Set (~7-8 Blöcke/Token) schon OHNE Streaming
in einen realen CPU-Speedup gegenüber einem vergleichbaren dichten Modell übersetzen?

## Ergebnis (gemessen, Kern-Zeit)

| Modell | Block-Anwendungen/Token | Qualität Lfin | Params | Kern-Zeit | tok/s |
|---|---|---|---|---|---|
| Sparse ModelC b64, R=6 | 24 | 3.720 | ~19M | 158 ms | 6 471 |
| Sparse ModelC b64, R=2 | 8 | ~3.72 (flach) | ~19M | 57 ms | 18 057 |
| Dense ModelA d24 (compute-matched) | 24 | **3.481** | 8.7M | **56 ms** | 18 185 |
| Dense ModelA d8 | 8 | **3.582** | 4.5M | **18 ms** | 58 016 |

Zusatz (ModelC-Bank, R=6): A_dense_all (alle 64 Blöcke, 384 Anw.) 823 ms; D_fixed_core
(7 Blöcke dicht) 92 ms; Router-Kosten B−C = 7 ms (**5 %**); reiner Dispatch C = 151 ms.

## Antwort: NEIN

**Das kleine Working Set übersetzt sich NICHT in einen CPU-Speedup.** Im Gegenteil:
- Sparse R=6 ist **2,8× langsamer** als compute-matched Dense d24 (158 vs 56 ms) — bei
  IDENTISCHEN 24 Block-Anwendungen — UND qualitativ schlechter (3.720 vs 3.481) bei mehr Params.
- Sparse R=2 (57 ms) ≈ Dense d24 (56 ms) in Zeit, aber Dense klar besser in Qualität.
- Dense d8 (18 ms) schlägt Sparse R=2 in Zeit (3×) UND Qualität.

**Ursache (gemessen):** Gleiche FLOPs, aber Sparse macht viele kleine Matmuls + Gather/Scatter
(Dispatch-Tax: 151 ms vs 56 ms für dieselben 24 Anwendungen ≈ 2,7×). Dense macht wenige große
BLAS-effiziente Matmuls. Der Router ist NICHT der Engpass (5 %), der Dispatch ist es.

**Korrektur eines früheren Fehlers:** Der erste „Sparse 5,4× schneller"-Wert verglich gegen das
dichte Rechnen ALLER 64 Bankblöcke (384 Anwendungen) — kein reales Modell. Gegen ein echtes
compute-matched Dense-Modell verliert Sparse.

## Konsequenz fürs Projekt

Der Wert der Architektur ist NICHT Rechengeschwindigkeit, sondern AUSSCHLIESSLICH Speicher:
Sparse liest pro Token nur ~7-8 eindeutige Blöcke statt aller Gewichte. Das zahlt sich NUR im
Streaming-/Offloading-Regime aus, wo das Modell nicht in den schnellen Speicher passt. Bei
allem-im-RAM gewinnt Dense auf der CPU klar.

→ Der **Offloading-Simulator** (gemessen in bewegten Bytes, RAM→VRAM, gegen Layer-Offloading-
Baseline) ist damit der einzige verbleibende Test, der die Architektur rechtfertigen könnte.
Compute-Speedup als Argument ist gemessen-widerlegt.

## Caveats (Fairness)

- Sparse-Dispatch ist UNOPTIMIERT (Python-Schleife über eindeutige Blöcke, kleine Matmuls).
  Ein gruppierter GEMM-Kernel könnte die 2,7×-Tax verkleinern, aber bei matched FLOPs Dense
  NICHT schlagen (gleiche Rechenmenge, Dense hat optimale BLAS-Lokalität).
- Eine Config (b64, CPU, 8 Threads). GPU-Bild kann abweichen, aber die FLOP-Logik gilt dort genauso.
- Sparse-Qualität leidet hier zusätzlich an Bank-Unternutzung (plain, hoher Gini). Besseres
  Training/Diversity könnte die QUALITÄTS-Lücke schließen, nicht die COMPUTE-Lücke.
