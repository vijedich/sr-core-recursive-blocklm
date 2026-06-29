"""Anytime-Inferenz — forciertes R fuer srcore_b32_k8_R6@15k.

Evaluiert das Modell so, als ob es nach R=1,2,3,4,6 Iterationen stoppen wuerde.
Verwendet logits[r-1] als Ausgabe-Logit der Iteration r (kein Re-Training noetig).
Misst:  Lfin(seen), Lfin(unknown), code_gain(seen), code_gain(unknown),
        tokens/s-Proxy (Forward-Time pro Token mit vollem R).

Nutzung:
  python scripts/anytime_inference.py --device cuda
  python scripts/anytime_inference.py --device cpu --n_batches 20
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rblm.heteromini import HeteroMiniData, DATA_ROOT
from rblm import model_io

RESULTS  = os.path.join(ROOT, "results")
HELDOUT  = DATA_ROOT + "_heldout"
CKPT     = os.path.join(RESULTS, "hm_cont_hm_srcore_b32_k8_R6_s0.pt")

import re as _re
def _seed_from_path(p: str) -> int:
    m = _re.search(r"_s(\d+)\.pt$", p)
    return int(m.group(1)) if m else 0


@torch.no_grad()
def eval_at_r(model, data, r_use, n_batches=40, bs=16, seq_len=128, device="cpu"):
    """
    r_use: 1-basiert (1..R). Verwendet logits[r_use-1] als Ausgabe.
    Gibt dict mit Lfin, anytime (r1→r_use), code_gain, domain_loss zurueck.
    """
    model.eval()
    losses_by_r_sum = None
    losses_by_r_n   = 0
    domain_last = {d: [] for d in range(data.n_domains)}
    domain_first = {d: [] for d in range(data.n_domains)}

    for _ in range(n_batches):
        toks, tgt, mask, dom = data.batch(bs, seq_len, device=device, mode="contiguous")
        logits_all, _ = model(toks)
        R_actual = len(logits_all)
        r_idx = min(r_use - 1, R_actual - 1)

        B, T = toks.shape
        V = logits_all[0].shape[-1]
        flat_mask = mask.reshape(-1)
        tgt_flat = tgt.reshape(-1)

        # Losses alle Iterationen r=1..r_use (fuer loss_per_iter Kurve)
        iter_losses = []
        for lg in logits_all[:r_idx + 1]:
            l = F.cross_entropy(lg.reshape(-1, V)[flat_mask], tgt_flat[flat_mask]).item()
            iter_losses.append(l)
        if losses_by_r_sum is None:
            losses_by_r_sum = [0.0] * len(iter_losses)
        for i, l in enumerate(iter_losses):
            losses_by_r_sum[i] += l
        losses_by_r_n += 1

        # code_gain: gain_r1 − gain_rR pro Sequenz
        ce_first = F.cross_entropy(logits_all[0].reshape(-1, V), tgt.reshape(-1),
                                   reduction="none").reshape(B, T)
        ce_last  = F.cross_entropy(logits_all[r_idx].reshape(-1, V), tgt.reshape(-1),
                                   reduction="none").reshape(B, T)
        gain_seq = (ce_first - ce_last).mean(dim=1).cpu().numpy()
        doms_np  = dom.cpu().numpy()
        for b in range(B):
            d = int(doms_np[b])
            if d >= 0 and d < data.n_domains:
                domain_last[d].append(
                    float(ce_last[b].mean().item()))
                domain_first[d].append(
                    float(ce_first[b].mean().item()))

    loss_per_iter = [s / losses_by_r_n for s in losses_by_r_sum] if losses_by_r_sum else []
    Lfin = loss_per_iter[-1] if loss_per_iter else float("nan")
    anytime = (loss_per_iter[0] - Lfin) if len(loss_per_iter) > 1 else 0.0

    # Per-Domäne Gain und Loss
    domain_results = {}
    for d in range(data.n_domains):
        name = data.domains[d]
        lasts  = domain_last[d]
        firsts = domain_first[d]
        if lasts:
            domain_results[name] = {
                "Lfin":      round(float(np.mean(lasts)), 4),
                "code_gain": round(float(np.mean([f-l for f,l in zip(firsts,lasts)])), 4),
            }

    return {
        "r_use":       r_use,
        "Lfin":        round(Lfin, 4),
        "anytime":     round(anytime, 4),
        "loss_per_iter": [round(l, 4) for l in loss_per_iter],
        "domain":      domain_results,
        "code_gain":   round(domain_results.get("code", {}).get("code_gain", 0), 4),
    }


@torch.no_grad()
def measure_speed(model, device, seq_len=128, bs=1, n_runs=20):
    """Tokens/s-Proxy: Zeit fuer einen Forward-Pass mit vollem R."""
    model.eval()
    dummy = torch.zeros(bs, seq_len, dtype=torch.long, device=device)
    # Warmup
    for _ in range(3):
        model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tokens_per_s = (n_runs * bs * seq_len) / elapsed
    ms_per_token = elapsed * 1000 / (n_runs * bs * seq_len)
    return round(tokens_per_s, 1), round(ms_per_token, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",    default="cuda")
    ap.add_argument("--n_batches", type=int, default=40)
    ap.add_argument("--bs",        type=int, default=16)
    ap.add_argument("--seq_len",   type=int, default=128)
    ap.add_argument("--ckpt",      default=CKPT)
    ap.add_argument("--r_list",    nargs="+", type=int, default=[1, 2, 3, 4, 6])
    ap.add_argument("--seed",      type=int, default=None,
                    help="Seed-Suffix fuer Output-Datei (default: aus Checkpoint-Pfad)")
    a = ap.parse_args()

    if not os.path.exists(a.ckpt):
        raise SystemExit(f"Checkpoint nicht gefunden: {a.ckpt}")

    data_seen = HeteroMiniData()
    data_unk  = HeteroMiniData(HELDOUT) if os.path.isdir(HELDOUT) else None

    print(f"Lade {os.path.basename(a.ckpt)} ...", flush=True)
    model, arch, step = model_io.load_checkpoint(a.ckpt, data_seen.vocab_size, a.device)
    R_full = arch["R"]
    name = model_io.label(arch, step)
    print(f"Modell: {name}  R={R_full}  device={a.device}")

    toks_per_s, ms_per_tok = measure_speed(model, a.device)
    print(f"Speed (volles R={R_full}): {toks_per_s:.0f} tok/s  |  {ms_per_tok:.3f} ms/tok")

    rows = []
    for r in a.r_list:
        if r > R_full:
            print(f"  [SKIP] R={r} > R_full={R_full}")
            continue
        print(f"\nEvaluiere R={r} (seen) ...", flush=True)
        seen = eval_at_r(model, data_seen, r, a.n_batches, a.bs, a.seq_len, a.device)
        unk  = None
        if data_unk:
            print(f"Evaluiere R={r} (unknown) ...", flush=True)
            unk = eval_at_r(model, data_unk, r, a.n_batches, a.bs, a.seq_len, a.device)

        row = {
            "r":            r,
            "block_apps":   r * arch.get("k", 4),
            "Lfin_seen":    seen["Lfin"],
            "anytime_seen": seen["anytime"],
            "code_gain_seen": seen["code_gain"],
            "Lfin_unk":     unk["Lfin"]   if unk else None,
            "code_gain_unk": unk["code_gain"] if unk else None,
            "toks_per_s":   toks_per_s,
            "ms_per_tok":   ms_per_tok,
        }
        if unk:
            row["code_ratio"] = round(unk["code_gain"] / max(0.0001, seen["code_gain"]), 3)
        rows.append(row)

        print(f"  R={r}: Lfin={seen['Lfin']:.4f}  anytime={seen['anytime']:.4f}  "
              f"code_gain={seen['code_gain']:.4f}", end="")
        if unk:
            print(f"  | unk: Lfin={unk['Lfin']:.4f}  code_gain={unk['code_gain']:.4f}  "
                  f"ratio={row.get('code_ratio','?')}", end="")
        print()

    # Ergebnis speichern
    result = {"model": name, "step": step, "R_full": R_full,
              "toks_per_s": toks_per_s, "ms_per_tok": ms_per_tok,
              "rows": rows}
    seed_tag = a.seed if a.seed is not None else _seed_from_path(a.ckpt)
    out_json = os.path.join(RESULTS, f"anytime_inference_{name}_s{seed_tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Tabelle ausgeben
    print(f"\n{'R':>4} {'BlockApps':>10} {'Lfin(s)':>9} {'anytime(s)':>11} "
          f"{'code_s':>7} {'code_u':>7} {'ratio':>6} {'ms/tok':>8}")
    print("-" * 75)
    for row in rows:
        print(f"{row['r']:>4} {row['block_apps']:>10} {row['Lfin_seen']:>9.4f} "
              f"{row['anytime_seen']:>11.4f} {row['code_gain_seen']:>7.4f} "
              f"{row.get('code_gain_unk') or 0:>7.4f} "
              f"{row.get('code_ratio') or 0:>6.3f} "
              f"{row['ms_per_tok']:>8.3f}")

    print(f"\nGespeichert: {out_json}")


if __name__ == "__main__":
    main()
