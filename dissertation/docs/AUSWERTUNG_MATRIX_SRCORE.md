# Auswertung — Faire Modellmatrix: Dense vs. Naked Sparse vs. SR-Fixed-Core

*Eingefroren: 2026-06-19. `scripts/matrix_sweep.py`. Alle: TinyStories, 3000 Schritte, k=4,
seed 0, End-Gewichtung, R direkt trainiert (nicht gekürzt). EIN Seed — Streuung beachten.*

## Frage

Kann ein kleiner rekursiv wiederverwendeter Kern (SR-Fixed-Core) bei konstanter aktiver
Breite die Tiefe eines dichten Modells teilweise ersetzen?

## Ergebnisse

**Dense (ModelA) — saubere Tiefenskalierung:**

| Modell | Params | Lfin |
|---|---|---|
| d4  | 3.4M | 3.611 |
| d8  | 4.5M | 3.582 |
| d12 | 5.5M | 3.555 |
| d24 | 8.7M | 3.481 |

**Sparse (b16 ~6.5M, b32 ~10.7M):**

| Modell | Lfin (R2/R4/R6) | Working Set | Gini | anytime (Within-Run-Tiefe) |
|---|---|---|---|---|
| naked_b16 | 3.713 / 3.711 / 3.741 | ~6 | 0.18–0.25 | ~0.001 |
| naked_b32 | 3.875 / 3.729 / 3.693 | 4.9–6.7 | ~0.34 | ~0.001 |
| srcore_b32 (per_token) | **3.663** / 3.911 / 3.701 | **4.0** | 0.43–0.70 | 0.024 / 0.012 |
| srsat_b32 (core_satellite) | 3.737 / 3.818 / 3.715 | 5.0 | ~0.60 | ~0.005 |

## Antworten

1. **Fixed-Core > Naked?** Teilweise, nicht robust: bei R2 klar (3.663 vs 3.875), bei R6
   gleichauf, bei R4 schlechter (3.911-Ausreißer). Bei einem Seed im Rauschen. ABER: SR-Core
   erzeugt echte Within-Run-Verfeinerung (anytime 0.012–0.024), Naked ist flach (~0.001).
2. **Bringt R Tiefe?** SR-Core innerhalb eines Laufs ja (loss_per_iter monoton fallend), über
   R nicht monoton (R2 am besten). Dense skaliert sauber mit Tiefe; Naked flach. → Rekursion
   gibt ein kleines Tiefensignal, ersetzt Dense-Tiefe NICHT.
3. **Working Set klein?** Ja, SR-Core glänzt: WS = 4.0 = k, EXAKT und unabhängig von R (nur
   die r1-Kernblöcke). Minimal möglicher aktiver Satz. srsat 5.0, naked 5–7.
4. **Sparse näher an Dense?** NEIN. Bestes Sparse (srcore R2 = 3.663, ~10.7M) < Dense d4
   (3.611, 3.4M) — ein Drittel der Params. Dense d24 (3.481) dominiert. Dense besitzt die
   Qualität-pro-Parameter-Front.
5. **Core+Satellit?** Nein — srsat verbessert nichts ggü. srcore, vergrößert nur WS (5.0 vs 4.0).

## Bottom Line

Auf TinyStories ersetzt rekursive Kern-Wiederverwendung die Dense-Tiefe NICHT. Dense dominiert
die Qualität-pro-Parameter-Front. Der einzige echte SR-Core-Vorteil ist kein Qualitäts-,
sondern ein SPEICHER-Vorteil: garantiertes Working Set = k = 4, unabhängig von R, plus ein
schwaches echtes Verfeinerungssignal (das Naked fehlt). Konsistent mit dem CPU-Benchmark
([[experiment-results]]): Der Wert der Architektur ist Speicher/Streaming, nicht Compute/Qualität
— und SR-Core ist dafür die beste Variante, weil es den aktiven Satz hart auf k pinnt.

## Caveats (Claim-Disziplin)

- EIN Seed. Der srcore-R4-Ausreißer (3.911) zeigt: Run-zu-Run-Streuung > die meisten
  Unterschiede. Für belastbare Aussagen über SR-Core vs. Naked: Cross-Seed nötig.
- TinyStories ist tiefen-arm (`TESTFELD_ADAPTIVE_TIEFE.md`) — dieser Datensatz KANN den
  Tiefenwert der Rekursion nicht zeigen. Der faire Tiefentest braucht härtere/heterogenere Daten.
- Quality-Befund (Dense dominiert) ist robust; die feinen Sparse-Unterschiede nicht.
