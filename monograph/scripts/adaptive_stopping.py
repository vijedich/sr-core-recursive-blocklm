"""Adaptive Stopping fuer SR-Core k8_R6 — Anytime-Inferenz mit token-adaptivem R.

Simuliert drei Stopping-Kriterien (offline, kein Batch-Masking):
  top1_stable  — stoppe wenn argmax(logits[r]) == argmax(logits[r-1]) N-mal in Folge
  entropy_drop — stoppe wenn |H(r) - H(r-1)| < theta
  kl_div       — stoppe wenn KL(p_r || p_{r-1}) < theta

Plus fixe Baselines: fixed_R = 2, 3, 4, 6

Fuer jeden Run:
  Lfin, code_gain_seen, code_gain_unk, code_ratio
  mean_R, std_R, compute_saved, R_histogram
  per_domain_mean_R

Nutzung:
  python scripts/adaptive_stopping.py --device cuda --seeds 0 1
  python scripts/adaptive_stopping.py --device cuda --seeds 0 1 2
"""
from __future__ import annotations
import argparse, json, os, sys, re as _re
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rblm.heteromini import HeteroMiniData, DATA_ROOT
from rblm import model_io

RESULTS = os.path.join(ROOT, "results")
HELDOUT = DATA_ROOT + "_heldout"

CKPTS = {
    0: "results/BASELINE_srcore_b32_k8_R6_15k.pt",
    1: "results/hm_cont_hm_srcore_b32_k8_R6_s1.pt",
    2: "results/hm_cont_hm_srcore_b32_k8_R6_s2.pt",
}

# Alle Konfigurationen
CONFIGS = []

# Fixe Baselines
for r in [2, 3, 4, 6]:
    CONFIGS.append({"kind": "fixed", "fixed_R": r, "min_R": r,
                    "label": f"fixed_R{r}", "theta": None, "consecutive": None})

# top1_stable: stoppe wenn argmax stabil fuer N aufeinanderfolgende Iterationen
for min_R in [2, 3]:
    for consec in [1, 2]:
        CONFIGS.append({"kind": "top1_stable", "min_R": min_R, "consecutive": consec,
                        "theta": None,
                        "label": f"top1_stable_minR{min_R}_c{consec}"})

# entropy_drop: |H(r) - H(r-1)| < theta
for min_R in [2, 3]:
    for theta in [0.001, 0.005, 0.01, 0.02]:
        CONFIGS.append({"kind": "entropy_drop", "min_R": min_R, "theta": theta,
                        "consecutive": None,
                        "label": f"entropy_drop_minR{min_R}_t{theta}"})

# kl_div: KL(p_r || p_{r-1}) < theta
for min_R in [2, 3]:
    for theta in [0.001, 0.005, 0.01, 0.05]:
        CONFIGS.append({"kind": "kl_div", "min_R": min_R, "theta": theta,
                        "consecutive": None,
                        "label": f"kl_div_minR{min_R}_t{theta}"})


def entropy(logits):
    """Entropie pro Token. logits: (B, T, V) -> (B, T)"""
    p = torch.softmax(logits, dim=-1)
    return -(p * torch.log(p + 1e-9)).sum(dim=-1)


def kl_div_pt(logits_new, logits_old):
    """KL(p_new || p_old) pro Token. -> (B, T)"""
    log_p = F.log_softmax(logits_new, dim=-1)
    log_q = F.log_softmax(logits_old, dim=-1)
    p = torch.exp(log_p)
    return (p * (log_p - log_q)).sum(dim=-1)


def compute_r_stop(logits_stack, cfg, R_max, device="cpu"):
    """
    logits_stack: (R, B, T, V) tensor
    Gibt r_stop (B, T) zurueck — 1-basiert, in [min_R, R_max].
    """
    R, B, T, V = logits_stack.shape
    min_R = cfg["min_R"]
    kind  = cfg["kind"]

    if kind == "fixed":
        r_val = min(cfg["fixed_R"], R_max)
        return torch.full((B, T), r_val, dtype=torch.long, device=device)

    r_stop = torch.full((B, T), R_max, dtype=torch.long, device=device)

    if kind == "top1_stable":
        consec_needed = cfg["consecutive"]
        top1_prev  = logits_stack[0].argmax(dim=-1)
        stable_cnt = torch.zeros(B, T, dtype=torch.long, device=device)
        stopped    = torch.zeros(B, T, dtype=torch.bool, device=device)

        for r_idx in range(1, R):
            r_1based = r_idx + 1
            top1_cur = logits_stack[r_idx].argmax(dim=-1)
            same     = (top1_cur == top1_prev)
            stable_cnt = torch.where(same, stable_cnt + 1, torch.zeros_like(stable_cnt))
            trigger = same & (stable_cnt >= consec_needed) & (r_1based >= min_R) & ~stopped
            r_stop  = torch.where(trigger, torch.full_like(r_stop, r_1based), r_stop)
            stopped = stopped | trigger
            top1_prev = top1_cur

    elif kind == "entropy_drop":
        theta   = cfg["theta"]
        H_prev  = entropy(logits_stack[0])
        stopped = torch.zeros(B, T, dtype=torch.bool, device=device)

        for r_idx in range(1, R):
            r_1based = r_idx + 1
            H_cur   = entropy(logits_stack[r_idx])
            drop    = (H_prev - H_cur).abs() < theta
            trigger = drop & (r_1based >= min_R) & ~stopped
            r_stop  = torch.where(trigger, torch.full_like(r_stop, r_1based), r_stop)
            stopped = stopped | trigger
            H_prev  = H_cur

    elif kind == "kl_div":
        theta   = cfg["theta"]
        stopped = torch.zeros(B, T, dtype=torch.bool, device=device)

        for r_idx in range(1, R):
            r_1based = r_idx + 1
            kl      = kl_div_pt(logits_stack[r_idx], logits_stack[r_idx - 1])
            trigger = (kl < theta) & (r_1based >= min_R) & ~stopped
            r_stop  = torch.where(trigger, torch.full_like(r_stop, r_1based), r_stop)
            stopped = stopped | trigger

    return r_stop


@torch.no_grad()
def eval_adaptive_on_data(model, data, cfgs, n_batches, bs, seq_len, device):
    """
    Laeuft alle Configs in einem Datensatz-Sweep.
    Gibt pro Config: Lfin, code_gain, domain stats, R-Statistiken.
    """
    model.eval()
    R_max = model.cfg.routed_iters

    # Akkumulatoren pro Config
    accum = {c["label"]: {
        "ce_sum":    0.0, "ce_n":     0,
        "gain_sum":  0.0, "gain_n":   0,
        "r_list":    [],
        "domain_ce_sum":   {d: 0.0 for d in data.domains},
        "domain_ce_n":     {d: 0   for d in data.domains},
        "domain_gain_sum": {d: 0.0 for d in data.domains},
        "domain_r_sum":    {d: 0.0 for d in data.domains},
        "domain_r_n":      {d: 0   for d in data.domains},
    } for c in cfgs}

    for _ in range(n_batches):
        toks, tgt, mask, dom = data.batch(bs, seq_len, device=device, mode="contiguous")
        logits_all, _ = model(toks)        # list of R tensors (B, T, V)

        R_actual = len(logits_all)
        logits_stack = torch.stack(logits_all, dim=0)   # (R, B, T, V)
        B, T = toks.shape
        V    = logits_stack.shape[-1]

        # CE nach Iteration r1 (fuer Gain-Messung)
        ce_r1 = F.cross_entropy(
            logits_stack[0].reshape(-1, V), tgt.reshape(-1),
            reduction="none").reshape(B, T)             # (B, T)
        doms_np = dom.cpu().numpy()

        for cfg in cfgs:
            lbl   = cfg["label"]
            ac    = accum[lbl]
            r_stop = compute_r_stop(logits_stack, cfg, R_actual, device=device)

            # Token-adaptive CE: fuer jedes (b,t) logits[r_stop[b,t]-1]
            # Baue Gather-Index: (B, T) -> index ins R-dim
            r_idx = (r_stop - 1).clamp(0, R_actual - 1)   # 0-basiert
            r_idx_exp = r_idx.unsqueeze(-1).expand(B, T, V).to(device)
            chosen_logits = logits_stack.gather(
                0, r_idx_exp.unsqueeze(0).expand(1, B, T, V)
            ).squeeze(0)                                  # (B, T, V)

            ce_chosen = F.cross_entropy(
                chosen_logits.reshape(-1, V), tgt.reshape(-1),
                reduction="none").reshape(B, T)           # (B, T)

            gain_seq = (ce_r1 - ce_chosen).mean(dim=1)   # (B,)
            r_mean_seq = r_stop.float().mean(dim=1)       # (B,)

            ac["r_list"].extend(r_stop.reshape(-1).cpu().tolist())

            for b in range(B):
                d = int(doms_np[b])
                dname = data.domains[d] if 0 <= d < data.n_domains else None

                ce_b = ce_chosen[b].mean().item()
                ac["ce_sum"] += ce_b; ac["ce_n"] += 1
                ac["gain_sum"] += gain_seq[b].item(); ac["gain_n"] += 1

                if dname:
                    ac["domain_ce_sum"][dname]   += ce_b
                    ac["domain_ce_n"][dname]     += 1
                    ac["domain_gain_sum"][dname] += gain_seq[b].item()
                    ac["domain_r_sum"][dname]    += r_mean_seq[b].item()
                    ac["domain_r_n"][dname]      += 1

    # Ergebnisse zusammenstellen
    results = {}
    for cfg in cfgs:
        lbl = cfg["label"]
        ac  = accum[lbl]
        rs  = ac["r_list"]
        r_arr = np.array(rs)
        hist = {int(r): int((r_arr == r).sum()) for r in range(1, R_max + 1) if (r_arr == r).sum() > 0}
        mean_R = float(np.mean(r_arr)) if len(r_arr) > 0 else float("nan")
        std_R  = float(np.std(r_arr))  if len(r_arr) > 0 else float("nan")

        dom_res = {}
        for dname in data.domains:
            n = ac["domain_ce_n"][dname]
            if n > 0:
                dom_res[dname] = {
                    "Lfin":      round(ac["domain_ce_sum"][dname] / n, 4),
                    "gain":      round(ac["domain_gain_sum"][dname] / n, 4),
                    "mean_R":    round(ac["domain_r_sum"][dname] / n, 3),
                }

        Lfin     = ac["ce_sum"] / max(ac["ce_n"], 1)
        gain     = ac["gain_sum"] / max(ac["gain_n"], 1)
        code_gain = dom_res.get("code", {}).get("gain", 0.0)

        results[lbl] = {
            "config":        cfg,
            "Lfin":          round(Lfin, 4),
            "gain":          round(gain, 4),
            "code_gain":     round(code_gain, 4),
            "mean_R":        round(mean_R, 3),
            "std_R":         round(std_R, 3),
            "compute_saved": round(1.0 - mean_R / R_max, 3),
            "R_histogram":   hist,
            "domain":        dom_res,
        }
    return results


def combine_seen_unknown(seen_res, unk_res):
    """Fuege Unknown-Daten zu Seen-Ergebnissen hinzu."""
    combined = {}
    for lbl, s in seen_res.items():
        u = unk_res.get(lbl, {})
        entry = dict(s)
        entry["Lfin_unk"]      = u.get("Lfin")
        entry["code_gain_unk"] = u.get("domain", {}).get("code", {}).get("gain")
        cgs = s.get("code_gain", 0) or 0
        cgu = entry["code_gain_unk"] or 0
        entry["code_ratio"]    = round(cgu / cgs, 3) if cgs > 0.001 else None
        entry["domain_unk"]    = u.get("domain", {})
        combined[lbl] = entry
    return combined


def print_table(combined, R_max=6):
    print(f"\n{'Label':<35} {'Lfin':>6} {'L_unk':>6} {'cg_s':>6} {'cg_u':>6} {'ratio':>6} {'mean_R':>7} {'saved':>6}")
    print("-" * 90)
    for lbl, r in combined.items():
        ratio_s = f"{r['code_ratio']:.3f}" if r.get("code_ratio") is not None else "  —  "
        cgu = f"{r['code_gain_unk']:.4f}" if r.get("code_gain_unk") is not None else "  —  "
        print(f"{lbl:<35} {r['Lfin']:>6.4f} {str(r.get('Lfin_unk','—')):>6} "
              f"{r['code_gain']:>6.4f} {cgu:>6} {ratio_s:>6} "
              f"{r['mean_R']:>7.3f} {r['compute_saved']:>6.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",    default="cuda")
    ap.add_argument("--n_batches", type=int, default=40)
    ap.add_argument("--bs",        type=int, default=16)
    ap.add_argument("--seq_len",   type=int, default=128)
    ap.add_argument("--seeds",     nargs="+", type=int, default=[0, 1])
    a  = ap.parse_args()

    data_seen = HeteroMiniData()
    data_unk  = HeteroMiniData(HELDOUT) if os.path.isdir(HELDOUT) else None

    all_results = {}

    for seed in a.seeds:
        ck_rel = CKPTS.get(seed)
        if not ck_rel:
            print(f"[skip] Kein Checkpoint fuer seed={seed}"); continue
        ck = os.path.join(ROOT, ck_rel)
        if not os.path.exists(ck):
            print(f"[skip] seed={seed}: {ck} nicht gefunden"); continue

        print(f"\n{'='*60}\n=== Seed {seed} ===\n{'='*60}", flush=True)
        model, arch, step = model_io.load_checkpoint(ck, data_seen.vocab_size, a.device)
        R_max = model.cfg.routed_iters
        print(f"  {model_io.label(arch, step)}  R={R_max}", flush=True)

        print("  Evaluiere Seen ...", flush=True)
        seen_res = eval_adaptive_on_data(
            model, data_seen, CONFIGS, a.n_batches, a.bs, a.seq_len, a.device)

        unk_res = {}
        if data_unk:
            print("  Evaluiere Unknown ...", flush=True)
            unk_res = eval_adaptive_on_data(
                model, data_unk, CONFIGS, a.n_batches, a.bs, a.seq_len, a.device)

        combined = combine_seen_unknown(seen_res, unk_res)
        print_table(combined, R_max)

        # Per-Domain mean_R Tabelle
        print(f"\n  Per-Domain mean_R (Seen):")
        ref_lbl = "fixed_R6"
        for lbl, r in combined.items():
            if r["config"]["kind"] in ("top1_stable", "entropy_drop", "kl_div") \
               and r["config"].get("min_R") == 3:
                dom_r = {d: v.get("mean_R") for d, v in r.get("domain", {}).items()}
                print(f"  {lbl:<35}: {dom_r}")

        all_results[seed] = combined

    # Speichern — seed-spezifisch damit parallele Runs sich nicht ueberschreiben
    seed_tag = "_".join(str(s) for s in sorted(all_results.keys()))
    out = os.path.join(RESULTS, f"adaptive_stopping_k8_R6_s{seed_tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nGespeichert: {out}")

    # Seed-Mittelwert-Tabelle fuer Seeds 0+1
    main_seeds = [s for s in [0, 1] if s in all_results]
    if len(main_seeds) >= 2:
        print(f"\n{'='*60}")
        print(f"=== Seed-Mittelwert Seeds {main_seeds} ===")
        print(f"{'='*60}")
        avg = {}
        for lbl in all_results[main_seeds[0]]:
            vals = [all_results[s][lbl] for s in main_seeds if lbl in all_results[s]]
            def _mean(key):
                vs = [v.get(key) for v in vals if v.get(key) is not None]
                return round(float(np.mean(vs)), 4) if vs else None
            avg[lbl] = {
                "Lfin": _mean("Lfin"), "code_gain": _mean("code_gain"),
                "code_gain_unk": _mean("code_gain_unk"),
                "code_ratio": _mean("code_ratio"),
                "mean_R": _mean("mean_R"), "compute_saved": _mean("compute_saved"),
                "config": vals[0]["config"],
                "Lfin_unk": _mean("Lfin_unk"),
                "domain": {}, "domain_unk": {},
            }

        print(f"\n{'Label':<35} {'Lfin':>6} {'L_unk':>6} {'cg_s':>6} {'cg_u':>6} {'ratio':>6} {'mean_R':>7} {'saved':>6}")
        print("-" * 90)
        for lbl, r in avg.items():
            ratio_s = f"{r['code_ratio']:.3f}" if r.get("code_ratio") is not None else "  —  "
            cgu = f"{r['code_gain_unk']:.4f}" if r.get("code_gain_unk") is not None else "  —  "
            lunk = f"{r['Lfin_unk']:.4f}" if r.get("Lfin_unk") is not None else "  —  "
            print(f"{lbl:<35} {r['Lfin']:>6.4f} {lunk:>6} "
                  f"{r['code_gain']:>6.4f} {cgu:>6} {ratio_s:>6} "
                  f"{r['mean_R']:>7.3f} {r['compute_saved']:>6.3f}")


if __name__ == "__main__":
    main()
