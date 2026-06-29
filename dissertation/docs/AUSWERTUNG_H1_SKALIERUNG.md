# Auswertung — H1-Skalierungstest (Working Set vs. Bankgröße)

*Eingefroren: 2026-06-18. `scripts/scaling_sweep.py`. n_blocks ∈ {64,128,256}, k=4, R=6,
3000 Schritte, seed 0, from-scratch, OHNE Diversity (Phase-2-Rezept) — nur n_blocks variiert.*

## Frage (die eigentliche VRAM-Hypothese)

Bleibt das aktive Working Set pro Token klein (~7-8), wenn die GESAMTE Blockbank wächst?
Oder wächst es mit der Bank mit?

## Ergebnis (gemessen)

| n_blocks | Working Set | aktiv/Bank | Gini | tote Blöcke | L1 | Lfin | Cache-Miss @50 % gelernt | vs. Zufall |
|---|---|---|---|---|---|---|---|---|
| 64  | 7.07 | 11.0 % | 0.575 | 0 | 3.714 | 3.720 | 0.077 | 6.7× |
| 128 | 7.52 | 5.9 %  | 0.655 | 0 | 3.628 | 3.630 | 0.059 | 8.7× |
| 256 | 7.38 | 2.9 %  | 0.814 | 1 | 3.638 | 3.650 | 0.044 | 11.7× |

Working Set = mittlere einzigartige Blöcke/Token (Maximum R·k=24). aktiv/Bank = WorkSet/n_blocks.
Cache-Miss @50 % = LRU-Miss-Rate bei Cache-Kapazität n_blocks/2.

## Befunde

**KERN — gemessen, nicht hochgerechnet:**
- **Working Set bleibt ~7.1–7.5** über die vierfache Bankgröße. p90=8 konstant. Das aktive Set
  pro Token ist ~bankgrößen-UNABHÄNGIG.
- **Aktiver Anteil halbiert sich pro Verdopplung** (11 → 5.9 → 2.9 %). Absolut ~konstant.
- **Transfer-Reduktion (n/WorkSet) skaliert linear:** 9.1× (64) → 17.0× (128) → 34.7× (256)
  — jetzt als MESSPUNKTE, nicht mehr Projektion.
- **Loss stabil:** leicht besser 64→128 (3.72→3.63), Plateau bei 256. Keine Verschlechterung.
- **Cache-Miss sinkt** mit wachsender Bank (0.077→0.044 @50 %), Vorsprung vs. Zufall wächst.

**HAKEN (plain/ohne Diversity):**
- **Konzentration steigt mit der Bank:** Gini 0.575→0.814, erster toter Block bei 256. Ein plain
  trainiertes Modell nutzt die große Bank NICHT voll aus — es verhält sich wie ein kleineres
  Modell plus untergenutzte Blöcke; die 256-Kapazität wird bei 3000 Schritten nicht in Qualität
  umgesetzt (Loss-Plateau nach 128).
- Hier greift der bereits validierte Diversity-Hebel (b64: Gini 0.57→0.12). Offene Folgefrage:
  hält Diversity bei großen Bänken das Working Set klein UND verteilt die Nutzung?

**Compute-Zeit (Beobachtung):** Wall-Time wuchs ~linear (35→62→118 Min). Trotz sparser
k=4-Compute kostet mehr Bank mehr Zeit (Router scort alle n; O(n²)-Coord-Term läuft auch bei
coord_w=0). Der Streaming-Gewinn ist VRAM, nicht automatisch Geschwindigkeit.

## Was das zeigt / nicht zeigt

- ZEIGT (gemessen bis n=256): aktives Working Set ~konstant → aktiver Anteil ~1/n → die zentrale
  Voraussetzung der VRAM-Hypothese hält.
- ZEIGT NICHT: Konstanz bis n=1000 (Trend klar, aber Extrapolation); reale Streaming-Durchsatz-
  Zahlen (Bytes/s, Tokens/s mit ausgelagerten Blöcken) — das trennt erst der Offloading-Simulator/
  Mikrobenchmark; ob große Bänke ohne Diversity sinnvoll nutzbar sind (Konzentration spricht dagegen).
