# Auswertung — Cross-Seed Curriculum C vs. No-Diversity-Kontrolle

*Eingefroren: 2026-06-18. Datengrundlage: Overnight-Pipeline (Seeds 0/1/2/3 Curriculum C
`diverse_from_iter=2`, plus No-Diversity-Warmstart als Kontrolle). Alle Modelle: warm
gestartet aus `tinystories_phase2/seed_0/step_3000`, 3000 Schritte, n_blocks=64, k=4, R=6.*

## Frage

Kommt die breitere Blocknutzung und (vermutete) stärkere Kategorieinformation vom
**iterationsabhängigen Diversity-Druck** oder nur von **zusätzlicher Trainingszeit /
Warmstart**? Die No-Diversity-Warmstart-Kontrolle (gleiche Basis, gleiche Schritte, kein
Diversity-Druck) isoliert das.

## Ergebnistabelle

| Modell | val_loss | Gini | Working Set | CLF (Zufall 0.167) | Kat-Jaccard off-diag |
|---|---|---|---|---|---|
| Seed 0 Curr | 3.127 | 0.127 | 7.58 | 0.424 [0.391, 0.451] | 0.282 |
| Seed 1 Curr | 3.069 | 0.128 | 7.63 | 0.389 [0.381, 0.429] | 0.336 |
| Seed 2 Curr | 3.158 | 0.114 | 7.59 | 0.363 [0.361, 0.427] | 0.249 |
| Seed 3 Curr | 3.103 | 0.118 | 7.44 | 0.385 [0.362, 0.404] | 0.262 |
| **Seed 0 NoDiv (Kontrolle)** | 3.124 | **0.509** | 7.48 | 0.383 [0.360, 0.402] | **0.792** |

CLF-CIs = Bootstrap 95%. Kat-Jaccard = Mittel der Off-Diagonale der 6×6-Kategorien-Matrix
(causality, coreference, dialogue, emotion, scene_shift, temporal); niedrig = Kategorien
routen weniger überlappend.

## Befunde (gemessen)

**Vom Diversity-Druck verursacht — robust über alle 4 Seeds:**
- **Gini 0.11–0.13 vs. 0.509** der Kontrolle. Saubere kausale Zuschreibung: identische
  Basis und Trainingsdauer, einziger Unterschied ist der Diversity-Druck → er verbreitert
  die Blocknutzung. Ohne ihn bleibt sie konzentriert (Hub-Struktur, wie Phase 2 Gini 0.62).
- Kategorien-Jaccard 0.25–0.34 vs. 0.792.

**NICHT vom Diversity-Druck verursacht:**
- **CLF statistisch ununterscheidbar von der Kontrolle** (Curr 0.363–0.424 vs. NoDiv 0.383;
  CIs überlappen, Seed 2 liegt drunter). Dekodierbare Kategorieinformation kommt aus
  Warmstart + Training, nicht aus dem Diversity-Druck.

**Quasi-invariant (alle Modelle):**
- **Working Set ~7.4–7.6** unabhängig von Diversity — nur bei n_blocks=64 gemessen.
- **val_loss 3.07–3.16 ≈ Kontrolle 3.124.** Curriculum C ist qualitäts-neutral gegen die
  echte Kontrolle (anders als Full-Diversity-from-start: 3.891). Der frühere
  „Qualitäts-Diversitäts-Zielkonflikt" war ein Artefakt der falschen Baseline (Phase 2).

## Interpretation (vorsichtig)

**Breitere Nutzung ≠ Kompetenzzentren.** Der niedrige Kategorien-Jaccard unter Curriculum
ist höchstwahrscheinlich ein mechanischer Nebeneffekt der uniformen Nutzung (mehr Blöcke
im Spiel → weniger Überlappung per Konstruktion), kein Beleg für kategorie-ausgerichtete
Zentren. Der unkonfundierte Test ist die CLF — und die zeigt keinen Vorteil gegenüber der
Kontrolle.

**Antwort auf die Forschungsfrage:** Unter diesem Setup KEIN Nachweis natürlich
entstehender, kategorie-ausgerichteter Kompetenzzentren. Der Diversity-Druck steuert *wo*
Last verteilt wird (Gini), aber nicht, dass die Verteilung kategorie-bedeutsam wird (CLF).
Konsistent mit Exp4 („B-schwach": schwaches verteiltes Kategoriesignal ~2,3× Zufall +
universaler Kern, kein starkes MoE-Verhalten).

## Was das NICHT zeigt (offen)

- Streambarkeit als reale Inferenz: weiter offen — keine gemessenen Bytes/s, Transfers,
  Tokens/s mit ausgelagerten Blöcken (Simulator mit Layer-Offloading-Baseline, Phase 5).
- Working-Set-Konstanz bei n_blocks=128/256/1000: weiter Hochrechnung (H1-Skalierungstest).
- Leiterbahn-Index/Predictor: nicht gebaut.
