# Publikationsplan: Recursive Block-Sparse LM

Stand: 2026-06-23  
Autor: Viktor Jedich

---

## Kurzfassung des Projekts

Dieses Projekt untersucht, ob ein Sprach-Modell durch blockweise Sparse-Aktivierung so aufgebaut werden
kann, dass pro Token nur eine kleine, vorhersagbare Menge von Gewichten aktiv ist (Working Set = k genau).
Das Ziel ist kein schnelleres Dense-Modell, sondern ein Modell, das strukturell für RAM→VRAM-Offloading
geeignet ist: der aktive Block-Satz (Leiterbahn) ist bei Inferenz vorab bekannt.

---

## Status (2026-06-23)

### Abgeschlossene experimentelle Kapitel

| Phase | Inhalt | Status |
|---|---|---|
| Phase 1 — Synthetik | A/B/C-Modelle, Tunnel, Jaccard-Overlap, WS-Garantie | ✅ Abgeschlossen |
| Phase 2 — Skalierung | TinyStories b16/b32/b64/b128/b256 × R2/R4/R6 | ✅ Abgeschlossen |
| CPU-Benchmark | Dispatch-Tax (Sparse 2.8× langsamer als Dense) | ✅ Abgeschlossen |
| HeteroMini-Smoke | 4-Domain-Matrix, Offloading-Sim, 2k-Steps | ✅ Abgeschlossen |
| HeteroMini-Long | b32 k8 R6 @15k, 3 Seeds, Cross-Seed-Robustheit | ✅ Abgeschlossen |
| Seen/Unknown-Split | Domänen-Partition, anytime_gain, Routing-Analyse | ✅ Abgeschlossen |
| Entmin-Sweep | λ=0.001–0.007, Negativ-Kontrollen, Target-Entropy | ✅ Abgeschlossen |
| Dense-Baseline | d24 @17k, 3 gepaarte Qualitäts-Evals | ✅ Abgeschlossen |

### Noch offen (nicht geblockt)

| Aufgabe | Priorität | Aufwand | Beschreibung |
|---|---|---|---|
| **Offload-Sim @17k** | HOCH | ~5 Min CPU | `offload_sim.py` auf ctrl/lam003/lam005/dense_d24 @17k — überbrückt K24-Metrik → Bytes-in-Motion |
| **eval_compare softfull** | MITTEL | ~10 Min GPU | Methodologische Konsistenz: softfull über eval_compare.py statt Trajectory-Stats |
| **Fließtext schreiben** | — | Wochen | Kapitel 1–7 auf Basis von writeup_skeleton_entmin.md und AUSWERTUNG_*.md |

### Nicht mehr vorgesehen

- Kein weiteres Training (Phase abgeschlossen)
- Keine neuen Modelle (außer oben genannte Evals auf bestehenden Checkpoints)
- Die `paper/` und `dissertation/` Ordner sind ab jetzt **eingefroren** (keine Überschreibungen)

---

## Empfohlene Ausarbeitungsstruktur

### Konferenz-Paper (aus `paper/`)

**Scope:** Entropy-Minimierung als Instrument zur Routing-Konsolidierung in Block-Sparse LMs.  
**Umfang:** ~8 Seiten (z.B. EMNLP Workshop, ICLR Workshop).  
**Kernbeitrag:** Entropy-based router consolidation induces a reproducible cache/locality axis within the sparse model family. Dense remains the quality upper bound; the contribution is controllable memory locality under a fixed active working set.

**Architektonische Vorgeschichte (F2 + F7 aus `RESEARCH_NOTE.md`):**

- **F2 — Two-Phase Routing:** SR-Core-Routing trennt sich spontan in einen token-spezifischen ersten Schritt (r=1, Jaccard=0.20) und einen universalen kollabierenden Kern (r=2..6, Jaccard≈0.984). Diese Struktur entsteht ohne explizite Supervision aus dem Language-Modeling-Loss. Sie ist die mechanistische Grundlage für das Working Set — und definiert gleichzeitig den Eingriffsort für Entropie-Regularisierung.
- **F7 — Reuse vs. Novelty:** Forced-Diversity-Ablationen zeigen, dass reiner Reuse-Kollaps nicht optimal ist: erzwungene Diversität verbessert tiefe Iterationen (r=6: −0.104 Nats im Vergleich zu normalem Routing). Die Routerstruktur ist also kausal relevant — und kontrollierbar.
- **Verbindung:** Entropy-Minimierung ist deshalb kein beliebiger Regularizer, sondern ein gezielter Eingriff in eine bekannte Schwäche der SR-Core-Routing-Dynamik. Der Beitrag ist nicht "Entropie ist gut", sondern: Entropie-Druck macht die Router-Konsolidierung zur steuerbaren Systems-Achse.

*Nur F2 und F7 werden im Paper genutzt. Kompetenzzentren, Curriculum A/C und Phase-3-Qualitätspreis bleiben Dissertation.*

**Paper-Struktur:**

```
1. Intro              — Offloading-Motivation, SR-Core und WS=k in 1 Absatz
2. Architectural Motivation
   2.1 Two-Phase Routing (F2): token-spezifischer Gatekeeper + kollabierender Kern
   2.2 Reuse vs. Novelty (F7): Routing-Kollaps als identifizierter Trainingsdefekt
3. Method             — router_entropy_loss (formal), WS-Invariante
4. Setup              — HeteroMini-v1, b64 k8 R6, Negative Controls
5. Results
   5.1 Router Consolidation (Fig 2: 2×2 Mechanismus)
   5.2 Cache/Quality Pareto (Fig 1: Pareto-Scatter)
   5.3 Dense Baseline (Fig 3: Heldout Bars)
6. Discussion         — Tradeoff, Grenzen, fehlende Timing-Validierung
7. Related Work
```

**Geschützte Claim-Formulierungen:**

- SAGEN: "Entropy-based consolidation induces a reproducible cache/locality axis."
- SAGEN: "Forced-diversity ablations show that unconstrained reuse collapse is not always optimal, motivating controlled interventions on router structure."
- NICHT SAGEN: "Sparse beats Dense", "speedup demonstrated", "inference is faster"
- NICHT SAGEN: "Entropy minimization solves the forced-diversity problem." — Entmin erzeugt Konsolidierung/Schärfung, nicht Diversity-Erhöhung. F7 motiviert den Eingriff; Entmin ist eine spezifische Variante dieses Eingriffs.

---

### Dissertation (aus `dissertation/`)

**Scope:** Vollständige Arbeit inkl. Motivation, Architektur, Skalierung, Systembeweis.  
**Umfang:** ~80–120 Seiten.

```
Kap 1 — Motivation & Problemstellung
         — Offloading-Rechnung, Abgrenzung zu MoE/Quantisierung

Kap 2 — Architektur: SR-Core und WS-Garantie
         Modelle: Dense d8/d16, Shared-Block, Naked b32, SR-Core b32/b64
         Eval:    Tunnel-Jaccard, streaming_results.json, A/B/C-Vergleich

Kap 3 — Skalierungseigenschaften: WS ist bankgrößen-unabhängig
         Modelle: tinystories_b64/b128/b256k4R6
         Eval:    AUSWERTUNG_H1_SKALIERUNG.md, Tabelle WS vs n_blocks

Kap 4 — Compute ist nicht der Gewinn: CPU-Benchmark
         Modelle: Sparse b64 R6, Dense d24 (compute-matched)
         Eval:    cpu_benchmark.json, AUSWERTUNG_CPU_BENCHMARK.md

Kap 5 — Multi-Domain: HeteroMini-v1
         5a. Modellmatrix (Dense/Naked/SR-Core b32/b64 × R2/R6)
             Eval: heteromini_hm_*.json, gain_*.json, seen_vs_unknown.json
         5b. Offloading-Simulation
             Eval: offload_sim.json, AUSWERTUNG_OFFLOAD_SIM.md
         5c. Cross-Seed-Robustheit (k8 R6, 3 Seeds)
             Eval: routing_analysis_crossseed_k8_R6.json,
                   anytime_inference_srcore_b32_k8_R6@15000_s*.json,
                   gain_seen_unknown_srcore_b32_k8_R6@15000_s*.json

Kap 6 — Router-Konsolidierung: Entropy-Minimierung
         → Direkt aus writeup_skeleton_entmin.md
         Modelle: ctrl/lam001/003/005/007/H375/H370 + Dense d24 @17k
         Eval:    eval_compare_*.json, eval_quality_*.json, hm_traj_*_b64*.json
         Figuren: fig_entropy_pareto.png, fig_router_consolidation.png,
                  fig_dense_vs_sparse_quality.png

Kap 7 — Diskussion, Grenzen, Ausblick
         — Was gezeigt (WS-Garantie, Transfer-Reduktion, Cache-Achse)
         — Was nicht gezeigt (Compute-Speedup, Dense-Qualität, reales Timing)
         — Offene Fragen: Leiterbahn-Index, Timing-Prototyp, 100M-Skala
```

---

## Fehlende Artefakte (für spätere Fertigstellung)

### Kritisch für Paper-Submission

| Artefakt | Status | Kernergebnis |
|---|---|---|
| `offload_sim_17k.json` | ✅ **Fertig (2026-06-23)** | Sparse ctrl **4.1× besser** als Dense bei K=8 (3035 vs 12348 KB/Token); lam005 spart weitere 62 KB/Token (−2.0%) vs ctrl. Crossover zu Dense bei K≥32 (dense_d24 passt komplett in Cache, sparse b64 nicht — bekanntes Skalenverhalten). |
| `eval_compare_softfull.json` | ✅ **Fertig (2026-06-23)** | softfull hat +17.8% unique_cores und niedrigere hard_overlap vs ctrl — Routing wird diffuser, nicht konsolidierter. Valide Negativ-Kontrolle bestätigt. |

### Nice-to-have für Dissertation

| Artefakt | Was fehlt | Fix |
|---|---|---|
| `eval_quality_lam001.json` | Code-Ratio für lam001 (kein Pareto-Punkt) | `python scripts/eval_quality_compare.py --b lam001` |
| `eval_quality_dense_vs_naked.json` | Dense vs Naked im Quality-Format | `python scripts/eval_quality_compare.py --a dense_d24_17k --b naked` |
| Offload-Sim Dense @17k | Dense Bytes-in-Motion für Fig-3-Systems-Kontext | wie oben, in offload_sim_17k mitlaufen |
| Adaptive-Stopping entmin | mean_R für lam005 vs ctrl | `python scripts/adaptive_stopping.py --ckpts ctrl lam005` |

---

## Ordnerstruktur

```
recursive-blocklm/
├── PUBLICATION_PLAN.md        ← diese Datei
├── paper/                     ← Konferenz-Paper Artefakte (EINGEFROREN)
│   ├── README.md
│   ├── .gitattributes         (Git LFS für *.pt)
│   ├── figures/               3 Paper-Figuren
│   ├── data/
│   │   ├── checkpoints/       14 .pt Dateien, ~3.0 GB
│   │   └── eval/              eval_compare + eval_quality + hm_traj b64 JSONs
│   ├── scripts/               build_figures.py + eval_*.py
│   └── docs/                  results_note_entmin_sweep.md, writeup_skeleton_entmin.md,
│                              RESEARCH_NOTE.md (F2/F7 als Architectural Motivation)
│
├── dissertation/              ← Vollständige Dissertation Artefakte (EINGEFROREN)
│   ├── README.md
│   ├── .gitattributes
│   ├── chapters/              LEER — für Fließtext
│   ├── figures/               alle Figuren
│   ├── data/
│   │   ├── checkpoints/
│   │   │   ├── phase1/        TinyStories A/B/C + Scaling-Matrix
│   │   │   ├── heteromini/    alle hm_cont_*.pt + heteromini_*.pt
│   │   │   └── entmin/        alle b64 k8 R6 Sweep-Modelle
│   │   └── eval/
│   │       ├── phase1/        A/B/C, TinyStories, CPU-Benchmark
│   │       ├── heteromini/    HM-Matrix, Long-Run, Seen/Unknown, Cross-Seed
│   │       └── entmin/        eval_compare + eval_quality + trajectories
│   ├── scripts/               alle 29 Skripte
│   └── docs/                  alle AUSWERTUNG_*.md + Theorie.md + FINDINGS.md
│
└── results/                   ← ORIGINAL (unverändert, eingefroren)
    ├── *.pt                   Originalcheckpoints
    └── *.json                 Originalergebnisse
```

---

## Nächste Schritte (sortiert nach Priorität)

1. [x] **Offload-Sim @17k** — `results/offload_sim_17k.json` ✅
2. [x] **eval_compare softfull** — `results/eval_compare_softfull.json` ✅
3. [ ] **Paper: Intro + Sec 2 (F2/F7) + Method** schreiben (Fließtext, ~3 Seiten) — narrativer Bogen: SR-Core WS-Garantie → Two-Phase-Routing → Routing-Kollaps als Trainingsdefekt → Entmin als gezielter Eingriff
4. [ ] **Paper: Results** schreiben (Fig 1/2/3 beschriften, ~2 Seiten)
5. [ ] **Dissertation Kap 2** — Architektur formal beschreiben
6. [ ] **Dissertation Kap 6** — direkt aus writeup_skeleton_entmin.md expandieren

---

*Bestehende Artefakte unter `paper/` und `dissertation/` sind eingefroren — keine direkten Änderungen.
Neue Analysen (z.B. offload_sim_17k, eval_compare_softfull) laufen zuerst in `results/` und werden
erst nach Prüfung manuell als neue Version in die Publikationsordner übernommen.*
