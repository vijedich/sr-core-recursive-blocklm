# Rekursive Block-Sparse Sprachmodelle: Architektur, Skalierung und Routing-Konsolidierung

**Dissertation — Viktor Jedich**  
**Status:** Artefakte eingefroren (2026-06-23).

## Inhalt

```
dissertation/
├── chapters/              LEER — für Fließtext (LaTeX oder Markdown)
│
├── figures/               Alle Figuren
│
├── data/
│   ├── checkpoints/
│   │   ├── phase1/        TinyStories-Matrix + A/B/C Synthetik (b16–b256)
│   │   ├── heteromini/    HeteroMini Long-Run (b32 k8 R6 s0–s2, k8 R8, naked, dense)
│   │   └── entmin/        Entmin-Sweep b64 k8 R6 (ctrl, lam001–007, H375/H370, neg-ctrl, dense)
│   │
│   └── eval/
│       ├── phase1/        A/B/C-Vergleich, TinyStories-Ergebnisse, CPU-Benchmark, Offload-Sim
│       ├── heteromini/    HM-Matrix, Long-Run, Seen/Unknown, Cross-Seed, Anytime, Domain-Partition
│       └── entmin/        eval_compare_*, eval_quality_*, hm_traj_*b64*, qual_gen
│
├── scripts/               Alle 29 Reproduktionsskripte
│
└── docs/                  Alle Analyse-Dokumente
    ├── AUSWERTUNG_*.md    (12 Auswertungsdokumente)
    ├── EXP*.md            (Phase-1 Experimente)
    ├── Theorie.md
    ├── FINDINGS.md
    ├── results_note_entmin_sweep.md
    └── writeup_skeleton_entmin.md
```

## Kapitelstruktur

| Kapitel | Inhalt | Hauptdaten |
|---|---|---|
| 1 | Motivation & Problemstellung | — |
| 2 | SR-Core Architektur, WS-Garantie | `data/eval/phase1/A_*.json`, `B_*.json`, `C_*.json` |
| 3 | Skalierung: WS bankgrößen-unabhängig | `data/eval/phase1/tinystories_b*.json` |
| 4 | CPU-Benchmark: Dispatch-Tax | `data/eval/phase1/cpu_benchmark.json` |
| 5a | HeteroMini Multi-Domain-Matrix | `data/eval/heteromini/heteromini_hm_*.json` |
| 5b | Offloading-Simulation | `data/eval/phase1/offload_sim.json` |
| 5c | Cross-Seed-Robustheit k8 R6 | `data/eval/heteromini/routing_analysis_crossseed_k8_R6.json` |
| 6 | Entropy-Minimierung | `data/eval/entmin/eval_compare_*.json` |
| 7 | Diskussion & Ausblick | — |

## Modell-Übersicht

**phase1/** (TinyStories + Synthetik, Phase 1–2):
- Synthetik: `model_C_routed_s0.pt` (Routing-Attraktor-Nachweis)
- TinyStories: b16/b32/b64/b128/b256 × k4 × R2/R4/R6, diverse Seeds
- Dense: `dense_d4/d8/d12/d24_s0_model.pt`

**heteromini/** (HeteroMini Long-Run Phase 3):
- HM-Smoke: `heteromini_hm_dense_d8/d24_s0_model.pt`, `hm_naked_b32_R2/R6_s0_model.pt`, etc.
- Long-Run b32: `hm_cont_hm_srcore_b32_k8_R6_s0/s1/s2.pt`, `hm_cont_hm_naked_b32_R6_s0.pt`
- Long-Run b64: `hm_cont_hm_srcore_b64_k8_R6_s0/s1.pt`, `hm_cont_hm_srcore_b64_R6_s0.pt`

**entmin/** (Phase 4 Entmin-Sweep):
- Ctrl: `hm_cont_hm_srcore_b64_k8_R6_ctrl_17k_s0/s1.pt`
- Dense: `hm_cont_hm_dense_d24_17k_s0.pt`
- Sweep: lam001, lam003 (s0/s1), lam005 (s0/s1), lam007
- Target-Entropy: H375, H370
- Negativ-Kontrollen: softfull, softsharp_a2, noise_0p1, coreloc_r1

**Hinweis:** .pt-Dateien (7.2 GB gesamt) sollten via Git LFS oder HuggingFace Hub gehostet werden.
