# Auswertung — Gesehen vs. Unbekannt (Memorisierung vs. Generalisierung)

*Eingefroren: 2026-06-20. `experiments/seen_vs_unknown.py`. SEEN = Sequenzen aus dem
Trainingskorpus (data/heteromini_v1, 6,6M Tok). UNKNOWN = frische, nie gesehene Dokumente
derselben 4 Domänen (Held-out, gestreamt mit Offset skip=3500, gleicher Tokenizer; 1,7M Tok).
contiguous-Sampling, n_batches=30. Metriken: Loss(CE), PPL, Top-1-Next-Token-Acc, loss_per_iter.*

## Ergebnis

| Modell | Set | Loss | PPL | Top-1 | Gap Δloss | PPL-Ratio |
|---|---|---|---|---|---|---|
| dense_d24@10k | seen | 5.129 | 169 | 0.184 | | |
| dense_d24@10k | unknown | 5.333 | 207 | 0.168 | +0.204 | ×1.23 |
| naked_b32_R6@10k | seen | 5.180 | 178 | 0.179 | | |
| naked_b32_R6@10k | unknown | 5.394 | 220 | 0.159 | +0.214 | ×1.24 |
| srcore_b32_R6@10k | seen | 5.133 | 169 | 0.187 | | |
| srcore_b32_R6@10k | unknown | 5.383 | 218 | 0.164 | +0.250 | ×1.28 |
| srcore_b32_R2@5k | seen→unk | 5.69→5.80 | 295→331 | 0.136→0.132 | +0.117 | ×1.12 |
| srcore_b64_R6@5k | seen→unk | 5.81→5.88 | 333→358 | 0.130→0.118 | +0.072 | ×1.08 |

## Pro-Domäne-Generalisierungslücke (UNKNOWN−SEEN Loss)

| Modell | web | wiki | code | lit |
|---|---|---|---|---|
| dense_d24@10k | +0.22 | +0.26 | +0.31 | +0.03 |
| naked_b32_R6@10k | +0.19 | +0.15 | +0.48 | +0.09 |
| srcore_b32_R6@10k | +0.21 | +0.27 | +0.31 | +0.06 |
| srcore_b32_R2@5k | +0.07 | +0.13 | +0.12 | −0.03 |
| srcore_b64_R6@5k | +0.10 | +0.01 | +0.23 | +0.00 |

## Befunde

1. **Alle Modelle generalisieren** — moderate Lücke (Loss +0.07…+0.25, PPL ×1.08…1.28). KEINE
   Memorisierung-Katastrophe. Auf 6,6M Tok / 5–10k Steps wurde Struktur gelernt, nicht auswendig.
2. **Lücke wächst mit Training:** 10k-Modelle (+0.20…0.25) > 5k-Modelle (+0.07…0.12). Mehr
   Epochen = engeres Fitten der gesehenen Fenster, aber kein Overfitting-Blowup.
3. **srcore_b32_R6 größte Lücke** (+0.25) der 10k-Modelle — fester Kern fittet seen minimal enger.
4. **Rekursionsgewinn kleiner auf UNKNOWN** (anytime srcore 0.026→0.020, dense 0.029→0.012) —
   Iterationen verfeinern teils die gesehene Verteilung.
5. **Code = am stärksten memorisiert** (größte Lücke in allen Modellen, naked +0.48); lit am
   besten generalisiert. Dieselbe Domäne (Code) mit dem HÖCHSTEN Rekursionsgewinn hat auch die
   GRÖSSTE Memorisierungslücke — repetitive/strukturierte Inhalte werden eng gefittet.

## Hinweise

- Top-1-Acc ~0.16–0.19 (kleine, untertrainierte Modelle; Zufall ~1/8000). PPL-Niveau hoch, weil
  HeteroMini heterogen + Modelle klein/kurz trainiert.
- Reine Vorhersage-Prüfung (Loss/PPL/Acc); qualitative Generierung (Prompt→Fortsetzung) waere ein
  optionaler Zusatz, aber automatisch schwer zu scoren.
- Wiederverwendbar: `python -m experiments.seen_vs_unknown --glob "results/hm_cont_*.pt"`.
