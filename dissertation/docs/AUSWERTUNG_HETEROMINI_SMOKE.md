# Auswertung — HeteroMini-v1 Smoke (Offloading-/Working-Set-Hypothese)

*Eingefroren: 2026-06-19. `scripts/heteromini_matrix.py`. HeteroMini-v1 (4 Domänen: web/wiki/
code/lit, 6,6M Tokens), 2000 Steps, seed 0, k=4, End-Gewichtung, R direkt trainiert.
WICHTIG: stark UNTERTRAINIERT (Lfin ~6.5 vs. ln(8000)=8.99; TinyStories erreichte ~3.5).
Routing-Geometrie-Metriken (WS, Reuse, Cache-Lokalität, Bytes) sind robust; Quality/anytime
vorläufig.*

## Tabelle 1 — Working Set / Domänen / Konzentration

| Config | Par | Lfin | anytime | WS_c | WS_s | Gini | dead | CLF/Zufall | Jacc-off |
|---|---|---|---|---|---|---|---|---|---|
| dense d8  | 4.5M | 6.541 | – | – | – | – | – | – | – |
| dense d24 | 8.7M | 6.473 | – | – | – | – | – | – | – |
| naked b32 R2 | 10.8M | 6.756 | 0.000 | 4.1 | 4.1 | 0.610 | 0 | 0.454/0.25 | 0.470 |
| naked b32 R6 | 10.8M | 6.568 | 0.022 | 7.5 | 7.4 | 0.558 | 0 | 0.494/0.25 | 0.633 |
| srcore b32 R2 | 10.8M | 6.561 | 0.000 | 4.0 | 4.0 | 0.612 | 1 | 0.473/0.25 | 0.422 |
| srcore b32 R6 | 10.8M | 6.710 | 0.021 | 4.0 | 4.0 | 0.792 | 8 | 0.450/0.25 | 0.889 |
| srcore b64 R2 | 19.3M | 6.659 | 0.000 | 4.0 | 4.0 | 0.746 | 5 | 0.469/0.25 | 0.377 |
| srcore b64 R6 | 19.3M | 6.687 | 0.000 | 4.0 | 4.0 | 0.823 | 31 | 0.437/0.25 | 0.554 |

## Tabelle 2 — Cache (contiguous/shuffled), Reuse, Bytes/Token

| Config | Miss@4 | Miss@8 | Miss@16 | Miss@32 | reuse p50/90/99 | fp16·int8 |
|---|---|---|---|---|---|---|
| naked b32 R2 | .30/.33 | .16/.20 | .05/.07 | .02/.03 | 4/28/212 | 2.18·1.09 MB |
| naked b32 R6 | .32/.31 | .16/.17 | .04/.06 | .01/.01 | 4/33/296 | 3.95·1.97 MB |
| srcore b32 R2 | .29/.36 | .13/.24 | .04/.08 | .02/.03 | 4/28/205 | 2.11·1.05 MB |
| srcore b32 R6 | .06/.06 | .02/.02 | .005 | .005 | 4/4/148 | 2.11·1.05 MB |
| srcore b64 R2 | .32/.37 | .17/.24 | .07/.11 | .03/.04 | 4/36/252 | 2.11·1.05 MB |
| srcore b64 R6 | .09/.12 | .04/.07 | .01/.02 | .006 | 4/4/268 | 2.11·1.05 MB |

## Kriterien-Verdikt

1. **WS≈4 auf Hetero-Daten? JA.** SR-Core exakt 4.0, unabhängig von R und Bankgröße. Naked
   wächst auf 7.5 (R6). Kernverhalten überträgt sich. ✓
2. **Domänenspezifischer als TinyStories? Teilweise.** Domäne aus Routing dekodierbar
   ~1.8–2.0× Zufall (TinyStories ~2.3×). Größere Bank bei R2 etwas distinkter
   (Jacc-off 0.377 b64 < 0.422 b32 < 0.470 naked). Nicht stärker; untertrainiert. ~
3. **contiguous CacheMiss << shuffled? JA (R2).** srcore b32 R2: Miss@16 0.04 vs 0.08 (2×),
   @8 0.13 vs 0.24. Dokument-Lokalität → Cache-Lokalität. ✓
4. **Größere Bank mehr Nutzen ohne WS-Anstieg? WS bleibt 4 ✓, Loss NICHT besser** (b64 < b32,
   5–31 tote Blöcke). Fixed-Core unternutzt große Bank bei diesem Budget. ✗ (Loss)
5. **anytime stabil? Gemischt.** b32 R6 = 0.021 (wie TinyStories ✓), b64 R6 = 0.000
   (verschwunden). Naked zeigt hier auch Signal (0.022) — nicht mehr SR-exklusiv. ~

## Stärkster Befund (robust trotz Untertraining)

SR-Cores fester Kern erzeugt extrem enge Reuse-Lokalität (R6: p50=p90=4) → Cache K=4 reicht
für ~6% Miss, Naked R6 braucht K=16+. Plus ~halbe Bytes/Token (2.1 vs 3.95 MB fp16, halbiert
bei int8). Genau die Eigenschaft fürs Offloading-Regime — und sie hält auf heterogenen Daten.

## Nächste Schritte

- Längeres Training (mehr Steps / mehr Tokens) für belastbare Loss/Domänen/anytime-Aussagen —
  der Smoke ist zu kurz für Quality-Schlüsse.
- Der klare contiguous-vs-shuffled-Cache-Vorteil + die enge Reuse-Lokalität motivieren jetzt
  direkt den Offloading-Simulator (bewegte Bytes statt FLOPs).
- Offen warum SR-Core b64 das anytime-Signal verliert (Konzentration/Untertraining?).
