"""Qualitaets-Eval: ctrl@17k vs entmin_lam005@17k.

Abdeckung:
  1. gain_su        -- Recursion-Gain seen vs. unknown, pro Domain und R
  2. loss_per_iter  -- LM-Loss R1..R6, pro Domain (code_ratio: L_code/L_other)
  3. adaptive_stop  -- mittlere R-Nutzung bei KL-Threshold
  4. Vergleichstabelle

Nutzung:
  python scripts/eval_quality_compare.py --device cuda
  python scripts/eval_quality_compare.py --ckpt_a A.pt --ckpt_b B.pt
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rblm.heteromini import HeteroMiniData, DATA_ROOT
from rblm import model_io

RESULTS      = os.path.join(ROOT, "results")
HELDOUT_ROOT = DATA_ROOT + "_heldout"

DEFAULT_A = os.path.join(RESULTS, "hm_cont_hm_srcore_b64_k8_R6_ctrl_17k_s0.pt")
DEFAULT_B = os.path.join(RESULTS, "hm_cont_hm_srcore_b64_k8_R6_entmin_r1_lam005_s0.pt")


# ─── Checkpoint laden ────────────────────────────────────────────────────────

def _load_any(ck_path, device):
    raw = torch.load(ck_path, map_location=device, weights_only=False)
    if "arch" in raw:
        return model_io.load_checkpoint(ck_path, device=device)
    cfg  = raw["config"]
    step = int(raw.get("step", 0))
    vocab = HeteroMiniData().vocab_size
    if cfg.get("variant") == "dense":
        from experiments.train_dense import make_dense
        depth = cfg.get("depth", 24)
        model, _ = make_dense(vocab, depth, device=device)
        model.load_state_dict(raw["model"])
        arch = {"variant": "dense", "depth": depth, "k": depth, "R": depth}
    else:
        from experiments.tinystories_exp import make_model
        model, _ = make_model(vocab, n_blocks=cfg["n_blocks"], k=cfg["k"],
                              R=cfg["R"], device=device, core_mode=cfg["core_mode"])
        model.load_state_dict(raw["model"])
        arch = {"n_blocks": cfg["n_blocks"], "k": cfg["k"], "R": cfg["R"],
                "core_mode": cfg["core_mode"]}
    return model, arch, step


# ─── Gain Seen vs. Unknown ────────────────────────────────────────────────────

@torch.no_grad()
def gain_su(model, data_seen, data_heldout, domain_names, n_batches, bs, seq_len, device):
    R = model.n_iters

    def _losses(data):
        ls = np.zeros((R, len(domain_names)))
        ns = np.zeros(len(domain_names))
        for _ in range(n_batches):
            toks, tgt, mask, dom = data.batch(bs, seq_len, device=device, mode="contiguous")
            logits_all, _ = model(toks)
            for r, lg in enumerate(logits_all):
                nll = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1),
                                      reduction="none").reshape(toks.shape)
                nll = (nll * mask).sum(-1) / mask.sum(-1).clamp(min=1)
                for d in range(len(domain_names)):
                    md = (dom == d)
                    if md.any():
                        ls[r, d] += nll[md].mean().item()
                        if r == 0: ns[d] += md.sum().item()
        return ls / max(1, n_batches)

    ls = _losses(data_seen)
    lh = _losses(data_heldout)
    gs = ls[0] - ls[-1]
    gu = lh[0] - lh[-1]

    return {
        "loss_per_iter_seen":    {domain_names[d]: [round(float(ls[r, d]), 4) for r in range(R)]
                                   for d in range(len(domain_names))},
        "loss_per_iter_heldout": {domain_names[d]: [round(float(lh[r, d]), 4) for r in range(R)]
                                   for d in range(len(domain_names))},
        "gain_seen":  {domain_names[d]: round(float(gs[d]), 4) for d in range(len(domain_names))},
        "gain_unk":   {domain_names[d]: round(float(gu[d]), 4) for d in range(len(domain_names))},
        "ratio":      {domain_names[d]: round(float(gu[d] / max(gs[d], 1e-6)), 3)
                       for d in range(len(domain_names))},
        "Lfin_seen":  {domain_names[d]: round(float(ls[-1, d]), 4) for d in range(len(domain_names))},
        "Lfin_heldout": {domain_names[d]: round(float(lh[-1, d]), 4) for d in range(len(domain_names))},
    }


# ─── Adaptive Stopping ────────────────────────────────────────────────────────

@torch.no_grad()
def adaptive_stopping(model, data, n_batches, bs, seq_len, device,
                      min_R=3, theta=0.005):
    R = model.n_iters
    r_sums = np.zeros(R + 1)
    n_tok  = 0
    for _ in range(n_batches):
        toks, _, _, _ = data.batch(bs, seq_len, device=device, mode="contiguous")
        logits_all, _ = model(toks)
        B, T = toks.shape
        r_stop = torch.full((B, T), R, dtype=torch.long, device=device)
        stopped = torch.zeros(B, T, dtype=torch.bool, device=device)
        p_prev = F.softmax(logits_all[0], -1)
        for r in range(1, R):
            p_cur = F.softmax(logits_all[r], -1)
            kl = (p_prev * (p_prev.log() - p_cur.log())).sum(-1)
            if r >= min_R:
                fire = (~stopped) & (kl < theta)
                r_stop[fire] = r
                stopped |= fire
            p_prev = p_cur
        for r in range(1, R + 1):
            r_sums[r] += (r_stop == r).sum().item()
        n_tok += B * T
    dist = {r: round(float(r_sums[r] / max(n_tok, 1)), 4) for r in range(1, R + 1)}
    mean_R = sum(r * v for r, v in dist.items())
    return {"mean_R": round(mean_R, 3), "compute_saved": round(1 - mean_R / R, 3),
            "r_distribution": dist}


# ─── Print ────────────────────────────────────────────────────────────────────

def print_comparison(label_a, label_b, qa, qb, domain_names):
    R = len(next(iter(qa["loss_per_iter_seen"].values())))

    print(f"\n{'='*65}")
    print(f"  QUALITAETS-VERGLEICH: {label_a}  vs  {label_b}")
    print(f"{'='*65}")

    # Loss per iter (seen)
    print(f"\n  LM-Loss R1..R{R} pro Domain (Seen):")
    print(f"  {'Domain':<10}", end="")
    for r in range(R): print(f"  R{r+1:<5}", end="")
    print()
    print("  " + "-" * (12 + 7 * R))
    for model_label, q in [(label_a, qa), (label_b, qb)]:
        print(f"  [{model_label[:8]:<8}]")
        for d in domain_names:
            vals = q["loss_per_iter_seen"][d]
            print(f"    {d:<10}" + "".join(f"  {v:.4f}" for v in vals))

    # Gain Seen vs Unknown
    print(f"\n  Recursion-Gain (Lfin_R1 - Lfin_R6) | Seen vs Held-out:")
    print(f"  {'Domain':<10}  {'gain_s_A':>9}  {'gain_s_B':>9}  {'gain_u_A':>9}  {'gain_u_B':>9}  {'ratio_A':>8}  {'ratio_B':>8}")
    print("  " + "-" * 70)
    for d in domain_names:
        print(f"  {d:<10}  {qa['gain_seen'][d]:>9.4f}  {qb['gain_seen'][d]:>9.4f}"
              f"  {qa['gain_unk'][d]:>9.4f}  {qb['gain_unk'][d]:>9.4f}"
              f"  {qa['ratio'][d]:>8.3f}  {qb['ratio'][d]:>8.3f}")

    # Lfin vergleich
    print(f"\n  Lfin (R={R}, Seen | Held-out):")
    print(f"  {'Domain':<10}  {'Lfin_s_A':>9}  {'Lfin_s_B':>9}  {'delta_s':>8}  {'Lfin_u_A':>9}  {'Lfin_u_B':>9}  {'delta_u':>8}")
    print("  " + "-" * 75)
    for d in domain_names:
        ds = qb["Lfin_seen"][d]    - qa["Lfin_seen"][d]
        du = qb["Lfin_heldout"][d] - qa["Lfin_heldout"][d]
        print(f"  {d:<10}  {qa['Lfin_seen'][d]:>9.4f}  {qb['Lfin_seen'][d]:>9.4f}  {ds:>+8.4f}"
              f"  {qa['Lfin_heldout'][d]:>9.4f}  {qb['Lfin_heldout'][d]:>9.4f}  {du:>+8.4f}")

    # Code-Ratio: L_code / mean(L_others)
    def code_ratio(q, split):
        key = f"Lfin_{split}"
        lc = q[key].get("code", float("nan"))
        others = [v for d, v in q[key].items() if d != "code"]
        lo = float(np.mean(others)) if others else float("nan")
        return round(lc / max(lo, 1e-6), 4)

    print(f"\n  Code-Ratio (L_code / mean_L_others) — niedriger = Code-Vorteil:")
    print(f"  {'':18}  {label_a:<14}  {label_b:<14}  Delta")
    print("  " + "-" * 58)
    for split in ["seen", "heldout"]:
        ra = code_ratio(qa, split)
        rb = code_ratio(qb, split)
        print(f"  {split:<18}  {ra:>14.4f}  {rb:>14.4f}  {rb-ra:>+.4f}")

    # Adaptive Stopping
    print(f"\n  Adaptive Stopping (KL theta=0.005, minR=3):")
    for model_label, q in [(label_a, qa["adaptive_stop"]), (label_b, qb["adaptive_stop"])]:
        print(f"  {model_label:<20}  mean_R={q['mean_R']:.3f}  "
              f"compute_saved={q['compute_saved']:.1%}")


# ─── Main ────────────────────────────────────────────────────────────────────

def eval_one(ck_path, data, data_heldout, n_batches, bs, seq_len, device, label):
    print(f"\n=== Qualitaets-Eval: {label} ===", flush=True)
    model, arch, step = _load_any(ck_path, device)
    if arch.get("variant") == "dense":
        print(f"  step={step}  dense_depth={arch['depth']}", flush=True)
    else:
        print(f"  step={step}  k={arch['k']}  R={arch['R']}", flush=True)
    model.eval()

    domain_names = data.domains
    print("  gain_su...", flush=True)
    gsu = gain_su(model, data, data_heldout, domain_names, n_batches, bs, seq_len, device)
    print("  adaptive_stopping...", flush=True)
    astop = adaptive_stopping(model, data, n_batches, bs, seq_len, device)
    gsu["adaptive_stop"] = astop
    return gsu, domain_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_a",    default=DEFAULT_A)
    ap.add_argument("--ckpt_b",    default=DEFAULT_B)
    ap.add_argument("--label_a",   default="ctrl@17k")
    ap.add_argument("--label_b",   default="lam005@17k")
    ap.add_argument("--device",    default="cuda")
    ap.add_argument("--n_batches", type=int, default=40)
    ap.add_argument("--bs",        type=int, default=16)
    ap.add_argument("--seq_len",   type=int, default=128)
    ap.add_argument("--out", default=os.path.join(RESULTS, "eval_quality_lam005.json"))
    a = ap.parse_args()

    data         = HeteroMiniData()
    data_heldout = HeteroMiniData(out_dir=HELDOUT_ROOT)

    qa, domain_names = eval_one(a.ckpt_a, data, data_heldout,
                                a.n_batches, a.bs, a.seq_len, a.device, a.label_a)
    qb, _            = eval_one(a.ckpt_b, data, data_heldout,
                                a.n_batches, a.bs, a.seq_len, a.device, a.label_b)

    print_comparison(a.label_a, a.label_b, qa, qb, domain_names)

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"label_a": a.label_a, "label_b": a.label_b,
                   "metrics_a": qa, "metrics_b": qb}, f, indent=2)
    print(f"\nGespeichert: {a.out}", flush=True)


if __name__ == "__main__":
    main()
