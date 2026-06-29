"""Vollstaendige Evaluation fuer k8_R6 (und optional alle anderen HeteroMini-Modelle).

Laeuft NACH dem Training. Fuhrt alle Benchmarks durch und schreibt
AUSWERTUNG_HETEROMINI_K8_R6.md mit klarer Entscheidung:
  R8 ja/nein?
  Leiterbahn-Simulator ja/nein?
  Domain-Partition-Confound ja/nein?

Nutzung (nach Training):
  python scripts/k8_eval.py
  python scripts/k8_eval.py --device cuda --n_batches 60
  python scripts/k8_eval.py --only results/hm_cont_hm_srcore_b32_k8_R6_s0.pt
"""
from __future__ import annotations
import argparse, json, os, sys, time
import torch

# Projekt-Root dem Pfad hinzufuegen
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rblm.heteromini import HeteroMiniData, DATA_ROOT
from rblm import model_io
from experiments.tinystories_exp import RESULTS
from experiments.heteromini_eval import evaluate
from experiments.domain_partition import run as dp_run, analyze as dp_analyze, print_report as dp_print
from experiments.gain_analysis import run as gain_run, run_seen_unknown, gain_analysis
from experiments.seen_vs_unknown import run as svu_run
from experiments.offload_sim import run as offload_run, measure_ws, project_scale

HELDOUT_DIR = DATA_ROOT + "_heldout"

# Baselines fuer Vergleich
BASELINE_GLOBS = [
    "results/hm_cont_hm_srcore_b32_R6_s0.pt",    # k4_R6
    "results/hm_cont_hm_naked_b32_R6_s0.pt",      # naked k4_R6
    "results/hm_cont_hm_dense_d24_s0.pt",         # dense
]


def find_checkpoints(only=None):
    """Gibt [(label, path)] fuer alle relevanten Checkpoints."""
    cks = []
    if only:
        for p in only:
            if os.path.exists(p):
                cks.append(p)
            else:
                print(f"[k8_eval] Warnung: {p} nicht gefunden.", flush=True)
        return cks
    # k8 zuerst
    k8_path = os.path.join(RESULTS, "hm_cont_hm_srcore_b32_k8_R6_s0.pt")
    if os.path.exists(k8_path):
        cks.append(k8_path)
    else:
        print(f"[k8_eval] FEHLER: k8_R6-Snapshot nicht gefunden: {k8_path}")
        print("[k8_eval] Training noch nicht abgeschlossen?")
    for p in BASELINE_GLOBS:
        fp = os.path.join(ROOT, p)
        if os.path.exists(fp):
            cks.append(fp)
    return cks


def load_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def run_all(checkpoints, device="cpu", n_batches=40, bs=16, seq_len=128):
    data = HeteroMiniData()
    heldout_ok = os.path.isdir(HELDOUT_DIR)
    results = {}

    for path in checkpoints:
        if not os.path.exists(path):
            print(f"[k8_eval] Uebersprungen (nicht gefunden): {path}")
            continue
        model, arch, step = model_io.load_checkpoint(path, data.vocab_size, device)
        name = model_io.label(arch, step)
        is_sparse = not model_io.is_dense(arch)
        print(f"\n{'='*60}")
        print(f"[k8_eval] {name}  |  {os.path.basename(path)}")
        print(f"{'='*60}", flush=True)

        row = {"name": name, "step": step, "path": path}

        # 1. HeteroMini-Eval (WS, cache miss, loss_per_iter, domain_clf ...)
        if is_sparse:
            print(f"[k8_eval] heteromini_eval ...", flush=True)
            t0 = time.time()
            ev = evaluate(model, data, n_batches=n_batches, bs=bs, seq_len=seq_len, device=device)
            ev["experiment"] = name
            ev_out = os.path.join(RESULTS, f"hm_eval_{name}_s0.json")
            with open(ev_out, "w", encoding="utf-8") as f:
                json.dump(ev, f, indent=2)
            row["eval"] = ev
            print(f"  -> WS={ev['working_set']['contiguous']:.2f}  "
                  f"anytime={ev['anytime']:.4f}  "
                  f"dead={ev['dead_blocks']}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

        # 2. Domain-Partition-Analyse (nur Sparse)
        if is_sparse:
            print(f"[k8_eval] domain_partition ...", flush=True)
            t0 = time.time()
            dp_res = dp_analyze(model, data, n_batches, bs, seq_len, device)
            dp_res["experiment"] = name; dp_res["step"] = step
            dp_out = os.path.join(RESULTS, f"domain_partition_{name}_s0.json")
            with open(dp_out, "w", encoding="utf-8") as f:
                json.dump(dp_res, f, indent=2)
            dp_print(dp_res, name)
            row["domain_partition"] = dp_res
            print(f"  -> excl={dp_res['mean_block_exclusivity']:.3f}  "
                  f"jaccard={dp_res['domain_jaccard_offdiag_mean']:.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

        # 3. Gain-Analyse (seen)
        print(f"[k8_eval] gain_analysis (seen) ...", flush=True)
        t0 = time.time()
        g_res = gain_analysis(model, data, n_batches, bs, seq_len, device)
        g_res["experiment"] = name; g_res["step"] = step
        g_out = os.path.join(RESULTS, f"gain_{name}_s0.json")
        with open(g_out, "w", encoding="utf-8") as f:
            json.dump(g_res, f, indent=2, ensure_ascii=False)
        row["gain_seen"] = g_res["domain_gain"]
        print(f"  -> gain: " + " ".join(f"{d}={v['mean_gain']:.4f}"
              for d, v in g_res["domain_gain"].items()) + f"  ({time.time()-t0:.0f}s)",
              flush=True)

        # 4. Gain-Analyse (unknown)
        if heldout_ok:
            print(f"[k8_eval] gain_analysis (unknown) ...", flush=True)
            t0 = time.time()
            data_unk = HeteroMiniData(HELDOUT_DIR)
            g_unk = gain_analysis(model, data_unk, n_batches, bs, seq_len, device)
            g_unk["experiment"] = name; g_unk["step"] = step
            # Vergleich
            comparison = {}
            for dom in data.domains:
                s = g_res["domain_gain"].get(dom)
                u = g_unk["domain_gain"].get(dom)
                if s and u:
                    comparison[dom] = {
                        "gain_seen": round(s["mean_gain"], 4),
                        "gain_unknown": round(u["mean_gain"], 4),
                        "delta": round(u["mean_gain"] - s["mean_gain"], 4),
                        "ratio_u_s": round(u["mean_gain"] / max(0.0001, s["mean_gain"]), 3),
                    }
            su_out = os.path.join(RESULTS, f"gain_seen_unknown_{name}_s0.json")
            with open(su_out, "w", encoding="utf-8") as f:
                json.dump({"experiment": name, "step": step,
                           "seen": g_res, "unknown": g_unk, "comparison": comparison},
                          f, indent=2, ensure_ascii=False)
            row["gain_comparison"] = comparison
            print(f"  -> Seen-Unknown gain delta: " +
                  " ".join(f"{d}={v['delta']:+.4f}" for d, v in comparison.items()) +
                  f"  ({time.time()-t0:.0f}s)", flush=True)

        # 5. Seen-vs-Unknown Loss (PPL, Top-1-Acc)
        if heldout_ok:
            print(f"[k8_eval] seen_vs_unknown (loss/PPL) ...", flush=True)
            t0 = time.time()
            try:
                svu = svu_run([path], n_batches=n_batches, bs=bs, seq_len=seq_len, device=device)
                row["seen_vs_unknown"] = svu
            except Exception as e:
                print(f"  [WARN] seen_vs_unknown fehlgeschlagen: {e}", flush=True)
            print(f"  ({time.time()-t0:.0f}s)", flush=True)

        # 6. Offload-Sim + Skalen-Projektion (nur Sparse)
        if is_sparse:
            print(f"[k8_eval] offload_sim + scale_projection ...", flush=True)
            t0 = time.time()
            ws = measure_ws(model, arch, data, n_batches // 4, bs, seq_len, device)
            row["ws_measured"] = round(ws, 3)
            print(f"  -> WS_measured={ws:.2f}  ({time.time()-t0:.0f}s)", flush=True)

        results[name] = row

    return results


def make_report(results, device):
    """Erstellt AUSWERTUNG_HETEROMINI_K8_R6.md mit Entscheidungen."""
    k4_key = next((k for k in results if "k4" in k or ("srcore" in k and "k8" not in k)), None)
    k8_key = next((k for k in results if "k8" in k), None)
    naked_key = next((k for k in results if "naked" in k), None)
    dense_key = next((k for k in results if "dense" in k), None)

    k4 = results.get(k4_key, {}) if k4_key else {}
    k8 = results.get(k8_key, {}) if k8_key else {}
    naked = results.get(naked_key, {}) if naked_key else {}

    def g(d, *keys, default="N/A"):
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, {})
            else:
                return default
        return d if d != {} else default

    def gain_val(row, dom):
        gs = g(row, "gain_seen", dom, "mean_gain")
        return f"{gs:.4f}" if isinstance(gs, float) else "N/A"

    def gain_unk_val(row, dom):
        v = g(row, "gain_comparison", dom, "gain_unknown")
        return f"{v:.4f}" if isinstance(v, float) else "N/A"

    # Entscheidungslogik
    def decide_r8():
        lpi_k8 = g(k8, "eval", "loss_per_iter")
        any_k8 = g(k8, "eval", "anytime", default=0.0)
        any_k4 = g(k4, "eval", "anytime", default=0.0)
        any_naked = g(naked, "eval", "anytime", default=0.0)
        gains = []
        if isinstance(lpi_k8, list) and len(lpi_k8) >= 2:
            last_drop = lpi_k8[-2] - lpi_k8[-1]
            gains.append(last_drop)
        yes_signals = 0
        if isinstance(any_k8, float) and isinstance(any_k4, float) and any_k8 > any_k4 * 1.1:
            yes_signals += 1
        if gains and gains[0] > 0.01:
            yes_signals += 1
        if isinstance(any_k8, float) and isinstance(any_naked, float) and any_k8 > any_naked * 0.85:
            yes_signals += 1
        return yes_signals >= 2, yes_signals

    def decide_leiterbahn():
        dp_k8 = k8.get("domain_partition", {})
        excl = dp_k8.get("mean_block_exclusivity", 0)
        joff = dp_k8.get("domain_jaccard_offdiag_mean", 1)
        cov = dp_k8.get("core_coverage", {})
        top10_vals = [v.get("top_10", 0) for v in cov.values() if isinstance(v, dict)]
        low_concentration = all(v < 0.5 for v in top10_vals) if top10_vals else True
        stable_geom = g(k8, "eval", "working_set", "contiguous", default=0)
        ok = (isinstance(stable_geom, float) and stable_geom <= 8.5 and
              excl < 0.9 and low_concentration)
        return ok

    def decide_partition():
        dp_k8 = k8.get("domain_partition", {})
        excl = dp_k8.get("mean_block_exclusivity", 0)
        joff = dp_k8.get("domain_jaccard_offdiag_mean", 1)
        return excl > 0.85 and joff < 0.15

    r8_ok, r8_signals = decide_r8()
    leiterbahn_ok = decide_leiterbahn()
    partition = decide_partition()

    lines = []
    lines.append("# Auswertung — srcore_b32_k8_R6 @10k")
    lines.append("")
    lines.append(f"*Eingefroren: {__import__('datetime').date.today()}. "
                 "Vergleich: k8_R6 vs. k4_R6 vs. Naked_k4_R6 vs. Dense_d24.*")
    lines.append("")

    # Haupttabelle
    domains = ["web", "wiki", "code", "lit"]
    lines.append("## Hauptvergleich")
    lines.append("")
    lines.append("| Modell | k | Lfin | anytime | gain_code | gain_code_unk | WS | excl |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for label, row, ki in [(k8_key, k8, g(k8, "eval", "n_blocks", default="?")),
                            (k4_key, k4, 4),
                            (naked_key, naked, 4),
                            (dense_key, results.get(dense_key, {}), "-")]:
        if not label:
            continue
        lpi = g(row, "eval", "loss_per_iter")
        lfin = f"{lpi[-1]:.4f}" if isinstance(lpi, list) else "N/A"
        anyt = g(row, "eval", "anytime")
        anyt_s = f"{anyt:.4f}" if isinstance(anyt, float) else "N/A"
        gc = gain_val(row, "code")
        gc_u = gain_unk_val(row, "code")
        ws = g(row, "eval", "working_set", "contiguous")
        ws_s = f"{ws:.2f}" if isinstance(ws, float) else "N/A"
        excl = g(row, "domain_partition", "mean_block_exclusivity")
        excl_s = f"{excl:.3f}" if isinstance(excl, float) else "N/A"
        lines.append(f"| {label} | {ki} | {lfin} | {anyt_s} | {gc} | {gc_u} | {ws_s} | {excl_s} |")
    lines.append("")

    # Domain-Gewinn-Tabelle
    lines.append("## Rekursionsgewinn pro Domaene (gain = loss_r1 - loss_rR)")
    lines.append("")
    lines.append("| Modell | web | wiki | code | lit |")
    lines.append("|---|---|---|---|---|")
    for label, row in [(k8_key, k8), (k4_key, k4), (naked_key, naked)]:
        if not label:
            continue
        cols = " | ".join(gain_val(row, d) for d in domains)
        lines.append(f"| {label} | {cols} |")
    lines.append("")

    # Gain Seen vs. Unknown
    lines.append("## Gain Seen vs. Unknown (Generalisiert Rekursion?)")
    lines.append("")
    lines.append("| Modell | Domaene | gain_seen | gain_unknown | delta | ratio |")
    lines.append("|---|---|---|---|---|---|")
    for label, row in [(k8_key, k8), (k4_key, k4), (naked_key, naked)]:
        if not label:
            continue
        comp = row.get("gain_comparison", {})
        for dom in domains:
            v = comp.get(dom)
            if v:
                lines.append(f"| {label} | {dom} | {v['gain_seen']:.4f} | "
                             f"{v['gain_unknown']:.4f} | {v['delta']:+.4f} | {v['ratio_u_s']:.3f}x |")
    lines.append("")

    # Domain-Partition-Diagnose
    dp_k8 = k8.get("domain_partition", {})
    dp_k4 = k4.get("domain_partition", {})
    lines.append("## Domain-Partition-Analyse")
    lines.append("")
    lines.append(f"**k=8, b=32, Domaenen=4**: k*Domaenen = 32 = Bankgroesse. "
                 f"Perfekte Partition theorie-moeglich.")
    lines.append("")
    lines.append("| Modell | excl | chance_excl | domain-Jaccard | unique_cores | Diagnose |")
    lines.append("|---|---|---|---|---|---|")
    for label, dp_res, ki in [(k8_key, dp_k8, 8), (k4_key, dp_k4, 4)]:
        if not label or not dp_res:
            continue
        excl = dp_res.get("mean_block_exclusivity", 0)
        chance = dp_res.get("chance_exclusivity_uniform", 0)
        joff = dp_res.get("domain_jaccard_offdiag_mean", 0)
        nu = dp_res.get("n_unique_cores_total", 0)
        if excl > 0.85 and joff < 0.15:
            diag = "STARKE PARTITION"
        elif excl > 0.65:
            diag = "MODERATE PARTITION"
        else:
            diag = "SCHWACH / GEMISCHT"
        lines.append(f"| {label} | {excl:.3f} | {chance:.3f} | {joff:.3f} | {nu} | {diag} |")
    lines.append("")

    # Offload-Projektion
    lines.append("## Offload-Projektion (WS gemessen)")
    lines.append("")
    lines.append("| Modell | WS | n_blocks=32 8x | n_blocks=8192 2048x |")
    lines.append("Verhaeltnis = n_blocks/WS. Fuer 6B fp16 bei 16 GB/s.")
    lines.append("")
    for label, row in [(k8_key, k8), (k4_key, k4), (naked_key, naked)]:
        if not label:
            continue
        ws = row.get("ws_measured")
        if ws:
            r32 = round(32 / ws)
            r8192 = round(8192 / ws)
            lines.append(f"| {label} | {ws:.2f} | {r32}x | {r8192}x |")
    lines.append("")

    # Entscheidungen
    lines.append("## Entscheidungen")
    lines.append("")
    lines.append(f"### R8 trainieren? {'JA' if r8_ok else 'NEIN'}")
    lines.append("")
    lines.append(f"Signale: {r8_signals}/3")
    lpi = g(k8, "eval", "loss_per_iter")
    if isinstance(lpi, list):
        lpi_str = " → ".join(f"{v:.4f}" for v in lpi)
        lines.append(f"loss_per_iter k8: {lpi_str}")
    anyt_k8 = g(k8, "eval", "anytime")
    anyt_k4 = g(k4, "eval", "anytime")
    if isinstance(anyt_k8, float) and isinstance(anyt_k4, float):
        lines.append(f"anytime k8={anyt_k8:.4f} vs k4={anyt_k4:.4f} "
                     f"({'k8 besser' if anyt_k8 > anyt_k4 else 'k4 besser oder gleich'})")
    lines.append("")
    if r8_ok:
        lines.append("Naechster Lauf: `srcore_b32_k8_R8 @10k`")
    else:
        lines.append("R8 weggelassen. Limit liegt nicht an Tiefe, "
                     "sondern an Core-Groesse oder Fixed-Core-Prinzip.")
    lines.append("")

    lines.append(f"### Leiterbahn-Simulator? {'JA' if leiterbahn_ok else 'NEIN'}")
    lines.append("")
    ws_k8 = k8.get("ws_measured", "?")
    lines.append(f"WS={ws_k8}  Partition-Status: {'stark' if partition else 'schwach/mittel'}")
    if leiterbahn_ok:
        lines.append("WS stabil und keine extreme Konzentration auf wenige Cores → "
                     "Leiterbahn-Index sinnvoll: Core-Transitionsgraph aufbauen.")
    else:
        lines.append("WS instabil oder Partition zu stark → erst Partition-Confound klaeren.")
    lines.append("")

    lines.append(f"### Domain-Partition-Confound? {'JA' if partition else 'NEIN'}")
    lines.append("")
    excl_k8 = dp_k8.get("mean_block_exclusivity", "?")
    joff_k8 = dp_k8.get("domain_jaccard_offdiag_mean", "?")
    lines.append(f"excl={excl_k8}  domain-Jaccard={joff_k8}")
    if partition:
        lines.append("CONFOUND AKTIV: k8 bildet hauptsaechlich 4 Domaenen-Macro-Cores.")
        lines.append("Claim muss lauten: 'k8 lernt Domaenen-Experten', nicht 'funktionale Kompetenzkerne'.")
        lines.append("Loesung: mit b128 testen (dann k*Domaenen << Bankgroesse), "
                     "oder Domaenen-Mischbatch (shuffled) als Training.")
    else:
        lines.append("Kein starker Confound: Bloecke werden domaenenuebergreifend genutzt.")
        lines.append("k8 entwickelt gemischte oder funktionale Cores, nicht reine Domaenen-Partition.")
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append("- Alle Metriken auf b32 (kleines Testfeld). Generalisierung zu b256+ unbewiesen.")
    lines.append("- gain_code_unknown: falls Rekursion hauptsaechlich Memorization, faellt delta.")
    lines.append("- Domain-Partition: domain_label NICHT sichtbar fuer Modell/Router (nur Metadatum).")
    lines.append("  Modell kann Domaene aus Textstatistik ableiten, hat aber keinen expliziten Schluessel.")
    lines.append("")

    md = "\n".join(lines)
    out = os.path.join(ROOT, "AUSWERTUNG_HETEROMINI_K8_R6.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nAuswertung gespeichert: {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="Nur diese Checkpoints (statt automatische Suche)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n_batches", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=128)
    a = ap.parse_args()

    cks = find_checkpoints(a.only)
    if not cks:
        raise SystemExit("[k8_eval] Keine Checkpoints gefunden.")
    print(f"[k8_eval] Evaluiere {len(cks)} Checkpoints: {[os.path.basename(p) for p in cks]}")

    results = run_all(cks, device=a.device, n_batches=a.n_batches, bs=a.bs, seq_len=a.seq_len)
    out = make_report(results, a.device)
    print(f"\n[k8_eval] Fertig. Bericht: {out}")


if __name__ == "__main__":
    main()
