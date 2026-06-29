# Auswertung — Skalen-Projektion (lohnt sich Offloading bei großen Modellen?)

*Eingefroren: 2026-06-20. `offload_sim.py --project`. PROJIZIERT die gemessene Routing-
Geometrie (Working Set) auf Zielmodellgrößen. Annahmen klar markiert: WS regime-stabil
(gemessen bis 256 Blöcke), transfer-limitiert (Roofline, Compute ignoriert), sparse lädt
WS Blöcke/Token kalt (konservativ, ohne Cross-Token-Cache), Dense-Baseline = Layer-Offload
(ganzes Modell/Token). 16 GB/s, fp16, 6 GB VRAM.*

## Kernbefund: Verhältnis = n_blocks / WS (blockgrößen-unabhängig)

**Ziel 6B:** Dense = 12 GB, passt NICHT in 6 GB → Layer-Offload = 1,3 Tok/s (einzige Option).

| Modell (WS) | n_blocks | Blockgröße | sparse Tok/s | × vs Dense-Offload |
|---|---|---|---|---|
| srcore (WS=4) | 32 | 375 MB | 11 | 8× |
| srcore (WS=4) | 256 | 47 MB | 85 | 64× |
| srcore (WS=4) | 1024 | 12 MB | 341 | 256× |
| srcore (WS=4) | 8192 | 1,5 MB | 2731 | 2048× |
| naked (WS=5.7) | 32 | 375 MB | 7 | 6× |
| naked (WS=5.7) | 8192 | 1,5 MB | 1910 | 1433× |

**Ziel 13B:** Dense 0,6 Tok/s (26 GB); srcore Regime B = 1260 Tok/s (2048×).

## Aussagen

1. Bei 6B/13B LÄUFT Dense nicht ohne Quantisierung; sparse läuft in jedem Regime →
   Ermöglichung, nicht Optimierung. (Bestätigt die Gründungsthese.)
2. Transfer-Verhältnis = n_blocks / WS:
   - Regime B (viele kleine Blöcke, n_blocks groß): bis 2048× — "lächerlich wenig" Transfer.
     k/WS praktisch egal.
   - Regime A (wenige große Blöcke): 8× — Transfer real (GB/Token), k/WS matters; aber
     immer noch läuft-vs-läuft-nicht.
3. Die eigentliche Wette ist das BLOCK-REGIME (Anzahl×Größe), nicht k. Schmaler Stream +
   viele kleine Blöcke = 2000×-Welt, falls das zu echter Fähigkeit skaliert (UNBEWIESEN).

## Caveats

- Projektion, keine Messung. WS-Stabilität bis 8192 Blöcke nicht verifiziert.
- Nur transfer-bound: bei Regime A binden große Block-Matmuls (k×R) evtl. den Compute →
  die 11 Tok/s sind obere Schranke. Regime B: Compute trivial, transfer-bound gilt.
- Quantisierung orthogonal: int8 verdoppelt beide Tok/s, Verhältnis n_blocks/WS bleibt.

## Konsequenz für die Forschung

Nächste Frage ist nicht k=4 vs 8, sondern: skaliert das Regime "schmaler Stream + viele
kleine Blöcke" zu Fähigkeit? Falls ja, ist der Offload-Gewinn riesig und k egal. Ein
k-/block_hidden-Sweep sollte nach SKALEN-PROJIZIERTEM Durchsatz bewertet werden, nicht nach
Toy-Ratios.
