# Gesamtanalyse — HeteroMini 10k: Offload-Trend + Rekursionsgewinn pro Domäne

*Eingefroren: 2026-06-19. Resume 5k→10k (3 Modelle), seed 0. Trajectory 1k/2,5k/5k/7,5k/10k.
Gain-Analyse (`gain = loss_r1 − loss_rR`) auf 10k-Snapshots, per Domäne + Top-Beispiele.*

## 1. Offload-Geometrie: per-Token stabil, Byte-Vorteil erodiert

srcore_b32_R6 Trajectory:

| step | Lfin | WS | reuse_p90 | K8 KB | ×Dense | ×Naked |
|---|---|---|---|---|---|---|
| 1000 | 7.002 | 4.0 | 4 | 159 | 77× | 8.1× |
| 2500 | 6.082 | 4.0 | 4 | 296 | 42× | 5.0× |
| 5000 | 5.515 | 4.0 | 4 | 505 | 24× | 3.6× |
| 7500 | 5.325 | 4.0 | 5 | 675 | 18× | 2.0× |
| 10000 | 5.356 | 4.0 | 5 | 746 | 16.7× | 1.85× |

naked_b32_R6 @10k: Lfin 5.383, WS 5.7, reuse_p90 31, K8 1377. dense_d24 @10k: Lfin 5.319, K8 12348.

- **WS=4.0 + reuse_p90≈4–5 halten perfekt** über das Training → Per-Token-Geometrie ist
  trainingsinvariant (das war die Primärfrage; robust positiv).
- **Byte-Vorteil schrumpft monoton:** vs Naked 3,6×→1,85×, vs Dense 24×→16,7×. Beide strikten
  GO-Schwellen (>20× Dense, >3× Naked) bei 10k GERISSEN. Das Warnsignal hat sich materialisiert.
- Ursache: srcore K8 steigt (Kerne streuen über Token-Strom), Naked wird cache-effizienter
  (K8 1797→1377) → Konvergenz gegeneinander.
- Loss plateaut ~5.3–5.4 (Dense minimal vorn; srcore knapp < Naked).

**Einordnung:** Der robuste Teil ist die Within-Token-Reuse (reuse_p90≈4 → K≥4-Cache hält den
Kern über die R Iterationen; ohne das wären es k·R=24 Loads/Token wie Dense). Die Cross-Token-
Cache-Teilung erodiert mit Training. Vs Dense bleibt 16,7× (stark, „sparse schlägt dense"); der
EXTRA-Vorteil des festen Kerns gegenüber freiem Sparse-Routing (Naked) schrumpft auf ~1,85×.

## 2. Rekursionsgewinn pro Domäne @10k — Hypothese bestätigt

mean gain (loss_r1 − loss_rR):

| Domäne | srcore | naked | dense |
|---|---|---|---|
| code | +0.039 | +0.060 | +0.037 |
| wiki | +0.023 | +0.050 | +0.025 |
| web  | +0.020 | +0.051 | +0.037 |
| lit  | +0.021 | +0.029 | +0.022 |

- **Code = höchster Rekursionsgewinn in allen 3 Modellen; Top-Gewinn-Beispiele durchweg Code**
  (strukturierte Syntax, Lizenz-Header). Robust, modellübergreifend.
- **Wächst mit Training:** srcore code-gain 0.019(5k)→0.039(10k), verdoppelt. „Mehr Training
  macht Rekursion nutzbarer" — bestätigt.
- lit (narrativ) am wenigsten → „lokale Sprachmodellierung = wenig Rekursionsgewinn".

## 3. Architektur-Tradeoff (neuer Befund)

Naked extrahiert MEHR Rekursionsgewinn als SR-Core (anytime @10k: naked 0.052 vs srcore 0.021).
Der feste Kern erzwingt Reuse, BEGRENZT dadurch aber die Verfeinerung — freies Routing holt
mehr aus den Iterationen. SR-Core tauscht Recursion-Gewinn gegen ein garantiert kleines,
vorhersagbares Working Set (WS=4 hart). Das ist der zentrale Tradeoff, jetzt belegt.

## GO-Neubewertung

- Strikt (5k-Schwellen >20× Dense, >3× Naked): bei 10k NICHT mehr erfüllt (16,7× / 1,85×).
- Real: sparse-recursive bewegt weiterhin ~17× weniger Bytes als Dense (der echte Offload-
  Baseline-Vergleich) — das bleibt stark. Der Mehrwert des FESTEN Kerns gegenüber Naked ist bei
  Konvergenz aber nur noch ~2× (Bytes), und Naked hat sogar mehr Recursion-Gewinn.

## Empfehlung

Der Prototyp-Case ist schwächer als der 5k-Snapshot suggerierte. Vor RAM→VRAM-Prototyp klären:
1. Ist der Vergleichsmaßstab Dense (16,7×, stark) oder Naked (1,85×, schwach)? Fürs Offloading
   zählt Dense — dann GO; geht es um „bringt der feste Kern was gegenüber simplem Sparse", ist
   die Antwort bei Konvergenz „wenig".
2. Cross-Token-Streuung untersuchen/begrenzen (z.B. Kern pro Sequenz statt pro Token pinnen,
   oder Kern-Cache-Loss) — könnte den Byte-Vorteil wieder ausbauen.
3. Recursion-Achse: code-spezifischer Gewinn ist das stärkste neue Signal → ein code-lastigerer
   Datensatz / längeres Training würde die Rekursionsfrage am direktesten weitertreiben.
