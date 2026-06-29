# Auswertung — Offloading-Simulator (bewegte Bytes, HeteroMini-Traces)

*Eingefroren: 2026-06-19. `experiments/offload_sim.py` auf den HeteroMini-Smoke-Checkpoints.
Block = ResidualMLPBlock(256,512) ≈ 263K Params = 526 KB fp16 / 263 KB int8. Bandbreite
16 GB/s (PCIe RAM→VRAM). Dense d24 = Layer-Offloading-Baseline (alle 24 Blöcke/Token).
Annahme: Modell passt NICHT in den schnellen Speicher (Offloading-Prämisse) → Cache K ≪ Bank.*

## Haupttabelle — contiguous, bytes/token fp16 (KB) | relativ zu Dense

| Modell | K=4 | K=8 | K=16 | reuse p90 |
|---|---|---|---|---|
| dense d24 (Baseline) | 12348 \| 1.00 | 12348 \| 1.00 | 12348 \| 1.00 | 24 |
| naked b32 R6 | 3876 \| 0.314 | 1940 \| 0.157 | 432 \| 0.035 | 33 |
| naked b32 R2 | 1247 \| 0.101 | 664 \| 0.054 | 182 \| 0.015 | 29 |
| srcore b64 R2 | 1340 \| 0.108 | 680 \| 0.055 | 249 \| 0.020 | 29 |
| srcore b64 R6 | 1101 \| 0.089 | 536 \| 0.043 | 83 \| 0.007 | 4 |
| srcore b32 R2 | 1151 \| 0.093 | 535 \| 0.043 | 147 \| 0.012 | 28 |
| srcore b32 R6 | 758 \| 0.061 | 215 \| 0.017 | 10 \| 0.001 | 4 |

(K=32/64 weggelassen: dort sind beide resident → rel.dense bedeutungslos. int8 = fp16/2.)

Roofline @16 GB/s, K=8: dense 790 µs/token (~1 270 tok/s) vs srcore_b32_R6 13,8 µs/token
(~72 000 tok/s) → 57× höheres Transferlimit. Volle Tabelle (alle Modelle/Modi/K, Belady-
Floor, stall_bytes, reuse p50/90/99) in results/offload_sim.json.

## Befunde

- **SR-Core gewinnt im Offloading-Regime massiv.** Bei K=8: srcore_b32_R6 = 0,017 von Dense
  (59× weniger Bytes/token) und 9× weniger als naked_R6. Bei K=4: 16× vs Dense; naked_R6 nur
  3,2× (WS 7,5 > Cache 4 → Thrashing). Der feste Kern (WS=4, reuse p90=4) passt in einen
  Mini-Cache.
- **Echte Bytes, nicht nur Miss-Raten:** der Vorteil ist in bytes/token quantifiziert.
- **Dokument-Scheduling zählt:** contiguous vs shuffled ~1,8× bei R2 (WS>Cache); bei srcore-R6
  order-invariant (WS=4 passt immer). Shuffled zerstört nicht, aber contiguous gewinnt solange
  WS>Cache → Batch-/Doc-Scheduling gehört in die Architektur-Story.
- **Dense muss bei K<24 das ganze Modell pro Token streamen** (Miss=1.0) — die Entkopplung
  „große Bank, kleiner aktiver Satz" ist hier in bewegten Bytes belegt.

## Go/No-Go: GO

Bedingung „SR-Core contiguous bei K=4/8 massiv weniger Bytes/token als Naked und Dense" ist
klar erfüllt → ein realer RAM→VRAM-Offloading-Prototyp lohnt sich.

## Caveats

- Beruht auf UNTERTRAINIERTEN Smoke-Modellen (HeteroMini 2000 Steps). Die Routing-GEOMETRIE
  (WS, Reuse, Miss) ist robust; Routing könnte sich mit mehr Training noch verschieben.
- Transfer-only-Roofline: ignoriert Compute-Zeit/Overlap. stall_bytes = bytes (LRU ohne
  Prefetch). Ein Prefetch-Oracle (Traces sind bekannt) könnte Stalls weiter verstecken.
- block_bytes identisch über Modelle (gleicher Block) → fairer Per-Block-Vergleich.

## Nächste Schritte (Viktors Reihenfolge)

1. Offloading-Sim ✓ (GO).
2. Längerer HeteroMini-Lauf (belastbare Quality/Domänen/anytime; prüfen ob Routing-Geometrie hält).
3. Realer RAM→VRAM-Prototyp (Prefetch, echte Bandbreite/Latenz, Compute-Overlap).
