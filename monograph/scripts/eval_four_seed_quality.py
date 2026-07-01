"""Held-out final-loss eval across seeds — produces an auditable summary JSON.

Re-evaluates the trained SR-Core b64 k8 R6 checkpoints (and, optionally, the entropy
lam003/lam005 variants) on the held-out HeteroMini split, under one fixed protocol, so the
per-seed quality tables in Chapters 5 and 6 have a single reproducible source.

Protocol (identical for every checkpoint):
  - corpus:  data/heteromini_v1_heldout  (the held-out split, not the training corpus)
  - mode:    contiguous (within-document windows; domain well-defined)
  - metric:  L_final = mean cross-entropy at the final recursion step r=R
  - sampling: deterministic — numpy seed fixed per checkpoint, fixed batch count
  - tokens:  n_batches * bs * seq_len  (default 60 * 16 * 128 = 122,880 held-out tokens)

Run from the repository root (needs rblm/ + experiments/ importable and the checkpoints
under monograph/data/checkpoints/; checkpoints are on Hugging Face):

    python monograph/scripts/eval_four_seed_quality.py
    python monograph/scripts/eval_four_seed_quality.py --entropy   # also lam003/lam005
"""
from __future__ import annotations
import argparse, json, os, sys

import numpy as np
import torch
import torch.nn.functional as F

MONO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # monograph/
REPO = os.path.dirname(MONO)
sys.path.insert(0, REPO)

from rblm import model_io
from rblm.heteromini import HeteroMiniData

CKPT = os.path.join(MONO, "data", "checkpoints")
OUT  = os.path.join(MONO, "data", "eval", "heteromini", "eval_quality_four_seed_summary.json")
HELDOUT = os.path.join(REPO, "data", "heteromini_v1_heldout")

# Checkpoint locations (relative to data/checkpoints/). Plain b64 k8 R6 seeds live in two
# subdirs because they were trained in two campaigns; the architecture is identical.
PLAIN = {
    0: "entmin/hm_cont_hm_srcore_b64_k8_R6_s0.pt",
    1: "entmin/hm_cont_hm_srcore_b64_k8_R6_s1.pt",
    2: "heteromini/hm_cont_hm_srcore_b64_k8_R6_s2.pt",
    3: "heteromini/hm_cont_hm_srcore_b64_k8_R6_s3.pt",
}
def _seeds(prefix):
    return {s: f"entmin/hm_cont_hm_srcore_b64_k8_R6_entmin_r1_{prefix}_s{s}.pt" for s in range(4)}

ENTROPY = {"lam003": _seeds("lam003"), "lam005": _seeds("lam005")}

# Quality-comparison baselines for Section 5a.5 (same protocol → auditable ~0.5-nat gap).
# Dense d24 reference = the 17k snapshot (the text's 4.81 figure). The 10k snapshot measures
# seen 5.14 / held 5.34 under this protocol — recorded in the protocol note for context; both
# models are still descending at their respective stops (undertraining note in 5a.5).
DENSE_D24 = {0: "entmin/hm_cont_hm_dense_d24_17k_s0.pt"}
PARAM_MATCHED = {s: f"heteromini/hm_cont_hm_srcore_b64_k16_R4_d256h192_s{s}.pt" for s in range(3)}


@torch.no_grad()
def held_out_final_loss(model, data, eval_seed, n_batches, bs, seq_len, device):
    """Mean cross-entropy at the final recursion step r=R, deterministic sampling."""
    model.eval()
    np.random.seed(eval_seed)
    tot_loss, tot_tok = 0.0, 0
    for _ in range(n_batches):
        toks, tgt, mask, _ = data.batch(bs, seq_len, device=device, mode="contiguous")
        logits_all, _ = model(toks)
        V = logits_all[-1].shape[-1]
        fm = mask.reshape(-1)
        lg = logits_all[-1].reshape(-1, V)[fm]
        tg = tgt.reshape(-1)[fm]
        m = int(fm.sum())
        tot_loss += F.cross_entropy(lg, tg, reduction="mean").item() * m
        tot_tok += m
    return tot_loss / max(1, tot_tok)


def _agg(d):
    vals = [v for k, v in d.items() if k.startswith("s")]
    return {**d, "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4)} if vals else d


def evaluate_set(name, ckpt_map, data_seen, data_held, args):
    seen, held = {}, {}
    for seed, rel in sorted(ckpt_map.items()):
        path = os.path.join(CKPT, rel)
        if not os.path.exists(path):
            print(f"  [{name}] s{seed}: MISSING {rel} — skipped")
            continue
        model, arch, step = model_io.load_checkpoint(path, data_seen.vocab_size, args.device)
        eval_seed = 1000 + seed   # deterministic, reproducible; differs per model seed
        Lh = held_out_final_loss(model, data_held, eval_seed, args.n_batches, args.bs,
                                 args.seq_len, args.device)
        Ls = held_out_final_loss(model, data_seen, eval_seed, args.n_batches, args.bs,
                                 args.seq_len, args.device)
        held[f"s{seed}"] = round(Lh, 4)
        seen[f"s{seed}"] = round(Ls, 4)
        print(f"  [{name}] s{seed}: held={Lh:.4f}  seen={Ls:.4f}  (step={step}, {rel})")
    return {"heldout": _agg(held), "seen": _agg(seen)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_batches", type=int, default=60)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--entropy", action="store_true", help="also eval lam003/lam005 s2/s3")
    args = ap.parse_args()

    data_held = HeteroMiniData(out_dir=HELDOUT)
    data_seen = HeteroMiniData()   # default = training corpus
    print(f"Held-out: {HELDOUT}  ({data_held.n_domains} domains, vocab {data_held.vocab_size})")
    print(f"Protocol: contiguous, r=R final loss, {args.n_batches}x{args.bs}x{args.seq_len} "
          f"= {args.n_batches*args.bs*args.seq_len} tokens, deterministic per seed\n")

    summary = {
        "protocol": {
            "heldout_corpus": "data/heteromini_v1_heldout",
            "seen_corpus": "data/heteromini_v1 (training corpus)",
            "mode": "contiguous",
            "metric": "mean cross-entropy at final recursion step r=R",
            "n_batches": args.n_batches, "bs": args.bs, "seq_len": args.seq_len,
            "tokens": args.n_batches * args.bs * args.seq_len,
            "eval_seed": "1000 + model_seed (fixed, reproducible)",
            "generated_by": "monograph/scripts/eval_four_seed_quality.py",
            "dense_d24_note": "reference is the 17k snapshot; the 10k snapshot measures "
                              "seen 5.14 / held 5.34 under this same protocol (still descending)",
        },
        "srcore_b64_k8_R6": evaluate_set("b64_k8_R6", PLAIN, data_seen, data_held, args),
        "dense_d24": evaluate_set("dense_d24", DENSE_D24, data_seen, data_held, args),
        "param_matched_k16_R4": evaluate_set("param_matched", PARAM_MATCHED,
                                             data_seen, data_held, args),
    }
    if args.entropy:
        for name, m in ENTROPY.items():
            summary[f"entmin_{name}"] = evaluate_set(name, m, data_seen, data_held, args)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
