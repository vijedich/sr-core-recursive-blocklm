# Ausführungsplan — A-Sweep + fehlende Checkpoints

**Stand:** angelegt 2026-06-24. Hardware: RTX 2060 6GB (heiß — Sweep auf 2 Nächte gesplittet).

## Ziel

1. **A-Sweep:** Sweet-Spot des Aktiv-Brains A=k·Blockgröße finden, ab dem Rekursionsgewinne
   bis r≈4 positiv bleiben. Gate für die 3B-Upcycle-Config (n/k/R).
2. **Fehlende Checkpoints:** Dissertationslücken schließen (compute-matched Dense d48,
   SR-Core Seeds s2/s3).

Korpus = HeteroMini (kein FineWeb-Flag im Trainer). Reicht fürs *qualitative* Band;
Feinkalibrierung fürs 3B später ggf. auf reicherem Korpus.

## Wichtige Trainer-Fallen (verifiziert in experiments/heteromini_long.py)

- **`--core_mode per_token` ist PFLICHT** für SR-Core. CLI-Default ist `None` → das ergäbe
  ein *naked*-Modell (freies Routing, WS≠k). Ohne diesen Flag ist der Lauf wertlos.
- **k=4 bekommt keinen `k_tag`** → exp_base = `hm_srcore_b64_R6` (nicht `..._k4_...`).
  Alle anderen k: `hm_srcore_b64_k{k}_R6`.
- From-scratch ok: ohne existierenden `hm_cont_*`-Checkpoint startet es bei step 0,
  `max_steps` = Gesamt-Steps. Trajektorie wird ab step 1000 geloggt.
- Checkpoint-Name: `results/hm_cont_{exp_base}{exp_tag}_s{seed}.pt`.

## Zeitplan (Kosten ∝ k; Referenz k8 ≈ 560 ms/step → 15k Steps ≈ 2,3 h)

### Nacht 1 — `queues/asweep_night1.json`  (~8,75 h) ✅ FERTIG (ok=4)
- [x] asw_k2   → hm_cont_hm_srcore_b64_k2_R6_asw_s0.pt
- [x] asw_k4   → hm_cont_hm_srcore_b64_R6_asw_s0.pt (Name OHNE k_tag!)
- [x] asw_k8   → hm_cont_hm_srcore_b64_k8_R6_asw_s0.pt
- [x] asw_k16  → hm_cont_hm_srcore_b64_k16_R6_asw_s0.pt
- [x] ΔL(r)-Eval (scripts/anytime_inference.py --ckpt ... --r_list 1 2 3 4 5 6)
      → anytime_inference_*.json. BEFUND: nutzbare Tiefe sauber bei r≈4 (k2 stirbt ab r≈2-3,
      = A-Floor zwischen k2/k4 = 4-8M). Magnitude single-seed-verrauscht (k8-Dip = Rauschen,
      Seed-Varianz 0.58 nats > Effekt). R=4 bestätigt.

### Nacht 2 — `queues/asweep_night2.json`  (~9,3 h)
- [ ] asw_k32  (der lange Pol; allein eine Nacht — Hitze) → danach ΔL(r)-Eval anhängen

### Nacht 3 — `queues/missing_checkpoints.json`  (~6,5–7 h)  [angepasst 2026-06-26]
- s2 ENTFERNT: `hm_cont_hm_srcore_b64_k8_R6_s2.pt` ist Kopie von `..._asw_s2.pt` (identische
  Config, schon trainiert) — kein Re-Training. Kopie erledigt.
- [ ] dense_d48 s0       (compute-matched Baseline, 17k, ~2,5 h)
- [ ] srcore_b64 s3      (15k, ~2,5 h — gemessen: k8/R6/15k = 9091s)
- [ ] entmin lam003 s2   (cont von s2-Basis, 2k Steps, ~20 min)
- [ ] entmin lam005 s2   (~20 min)
- [ ] entmin lam003 s3   (cont von s3-Basis — läuft NACH srcore_b64 s3, ~20 min)
- [ ] entmin lam005 s3   (~20 min)
- Damit: 4-Seed-Qualität (s0/s1 vorhanden + s2-Kopie + s3 neu) UND λ-Pareto über 4 Seeds.

### Nacht 4 — `queues/kseed_k8.json`  (~4,7 h)
- [ ] asw_k8 s1  → bestätigt/widerlegt den k8-Dip als Rauschen
- [ ] asw_k8 s2
- [ ] danach ΔL(r)-Eval auf s1/s2 → k8 mit 3 Seeds (s0/s1/s2) gegen k4/k16-Nachbarn lesen

## Nach dem Training (kein/wenig GPU)

- [x] **Plot-Skript fertig:** `scripts/plot_asweep.py` → `results/fig_asweep_depth.png` (3 Panels:
      kumul. Gewinn, Marginal+Schwelle, Qualität vs A). Greift k32/Seeds automatisch mit —
      nach jeder Trainingsnacht + ΔL(r)-Eval einfach neu laufen lassen.
- [ ] Plot ΔL(r) vs. r (eine Linie pro k) + "nutzbare Tiefe vs. A" über alle Punkte (k2..k32).
- [ ] Sweet-Spot = KLEINSTES A oberhalb des Floors mit lebendiger Rekursion + akzeptabler
      Qualität (NICHT das größte A — Ziel ist Streambarkeit auf Consumer-HW, nicht Peak-Qualität).
- [ ] λ-Sweep s2/s3 anhängen (braucht srcore s2/s3 als Basis: `--cont_src <base>` +
      `--lambda_entropy 0.003 / 0.005`).

## Gated / später (NICHT auf der 2060)

- **A-Sweep Phase 2** (Blockgröße bei festem A): `--d_model` / `--block_hidden`-Flags sind
  jetzt in heteromini_long (Default 256/512 = namensgleich zu Alt-Checkpoints; nicht-Default
  => Namens-Suffix `_d{d}h{h}`, keine Kollision). Queue erst bauen, wenn Sweet-Spot-A aus
  Phase 1 steht — dann gleiches A über (d_model, k)-Kombis halten und ΔL(r) vergleichen.
- **3B-Konversion (Upcycle):** Cloud-Job (~60 GB VRAM für 3B+Adam, passt nicht in 6 GB).
  Config (n/k/R) = Ergebnis des A-Sweeps. Parallel vorbereitbar: Qwen-Config zerlegen,
  FFN-Split-Code, Cloud-Env. Qwen vor Gemma (kleineres Vokab → mehr Budget in FFN-Blöcken).
  R=2 starten und hochwachsen lassen (Rekursion mappt nicht sauber aus Dense).

## Start-Kommandos

```powershell
# Validieren (kein GPU):
python scripts/run_queue.py queues/asweep_night1.json --dry-run
# Detached starten (per etabliertem Muster, NICHT als Harness-Task):
.\scripts\launch_detached.ps1 -LogName queue_night1 -CmdArgs @('scripts/run_queue.py','queues/asweep_night1.json')
# Monitoring: results/_asw_k2.log ... und results/_queue_*.log
```
