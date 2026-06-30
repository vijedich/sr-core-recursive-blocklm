"""Nachberechnung der Ablations-Daten fuer Seeds 1-3 (TinyStories b64k4R6).

Laedt jeweils den passendenCheckpoint und fuehrt den Ablationstest durch.
Ergebnisse werden als JSON in dissertation/data/eval/phase1/ gespeichert.

Nutzung:
    python -m scripts.competence_ablation_crossseed
    python -m scripts.competence_ablation_crossseed --device cpu
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.competence_centers_exp import (
    load_with_text,
    categorized_windows,
    flat_ids,
    FlatData,
    make_model,
    collect_routing,
    compute_lift,
    ablation_test,
    _load_checkpoint,
)

RESULTS = os.path.join(ROOT, "results")
DATA_OUT = os.path.join(ROOT, "dissertation", "data", "eval", "phase1")

CHECKPOINTS = {
    1: "tinystories_b64k4R6_s1_div0.0_crd0.0_diverse_currFromIter2_warm_model.pt",
    2: "tinystories_b64k4R6_s2_div0.0_crd0.0_diverse_currFromIter2_warm_model.pt",
    3: "tinystories_b64k4R6_s3_div0.0_crd0.0_diverse_currFromIter2_warm_model.pt",
}


def run(device: str = "cuda", vocab: int = 8000, max_docs: int = 20000,
        seeds: list[int] | None = None, n_top: int = 5, max_batches: int = 10):

    target_seeds = seeds if seeds is not None else [1, 2, 3]

    print("[ABL] Lade TinyStories-Daten ...")
    texts, tok = load_with_text(vocab=vocab, max_docs=max_docs)
    cat_wins = categorized_windows(texts, tok, seq_len=128, stride=64, min_per_cat=100)
    ids = flat_ids(texts, tok)
    data = FlatData(ids, tok.get_vocab_size())

    cats_sorted = sorted(cat_wins.keys())
    print(f"[ABL] Kategorien: {cats_sorted}")
    print(f"[ABL] Seeds: {target_seeds}  n_top={n_top}  device={device}")

    os.makedirs(DATA_OUT, exist_ok=True)

    for seed in target_seeds:
        ckpt_name = CHECKPOINTS.get(seed)
        if ckpt_name is None:
            print(f"[ABL] Kein Checkpoint fuer Seed {seed} definiert — uebersprungen.")
            continue

        ckpt_path = os.path.join(RESULTS, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"[ABL] Checkpoint nicht gefunden: {ckpt_path} — uebersprungen.")
            continue

        print(f"\n=== Seed {seed}: {ckpt_name} ===")
        model, cfg = make_model(data.vocab_size, device=device)
        _load_checkpoint(model, ckpt_path, device)

        print(f"[ABL] Sammle Routing-Daten ...")
        routing = collect_routing(model, cat_wins, bs=32, max_windows=1000, device=device)
        compute_lift(routing)

        print(f"[ABL] Ablationstest (n_top={n_top}, max_batches={max_batches}) ...")
        delta, cats, normal_loss = ablation_test(
            model, cat_wins, routing,
            n_top=n_top, bs=32, max_batches=max_batches, device=device,
        )

        # Ergebnis
        print(f"\n  Ablations-Diagonale (n_top={n_top}):")
        for i, cat in enumerate(cats):
            offdiag = (delta[i].sum() - delta[i, i]) / max(len(cats) - 1, 1)
            marker = "SPEZIALIST" if delta[i, i] > offdiag else "generalist"
            print(f"    {cat:14s}  selbst={delta[i, i]:+.4f}  andere={offdiag:+.4f}  {marker}")

        result = {
            "tag": f"competence_ablation_s{seed}",
            "seed": seed,
            "checkpoint": ckpt_name,
            "n_top_ablated": n_top,
            "ablation_cats": cats,
            "ablation_delta": delta.tolist(),
            "normal_loss": {c: float(v) for c, v in normal_loss.items()},
        }

        out_path = os.path.join(DATA_OUT, f"competence_ablation_s{seed}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"  -> Gespeichert: {out_path}")

    print("\n[ABL] Fertig.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--n_top", type=int, default=5)
    p.add_argument("--max_batches", type=int, default=10)
    args = p.parse_args()
    run(device=args.device, seeds=args.seeds, n_top=args.n_top, max_batches=args.max_batches)
