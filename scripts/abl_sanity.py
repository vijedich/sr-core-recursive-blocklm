"""Sanity-Check: Validiert Ablations-Losses fuer Curriculum C.

Testet:
1. Theoretisches Maximum fuer CE-Loss (NICHT log(vocab_size) - kann beliebig gross sein)
2. Baseline Loss auf cat_wins (wie in der Analyse)
3. Ablations-Loss auf den echten Top-5 Lift-Bloecken
4. Ablations-Loss auf random Bloecke (Kontrolle)
5. Per-Iteration: wo kommt der Schaden her?
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import numpy as np
import math
import json

from experiments.competence_centers_exp import (
    make_model, _load_checkpoint, RESULTS,
    load_with_text, categorized_windows, flat_ids, FlatData,
    collect_routing, compute_lift
)
from rblm.models import iteration_losses

device = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "checkpoints/tinystories_curriculum_fromIter2/seed_0/step_3000"
VOCAB = 8000

print(f"Lade Checkpoint: {CKPT}", flush=True)
model, cfg = make_model(vocab_size=VOCAB, device=device)
_load_checkpoint(model, CKPT, device=device)
model.eval()
nb = model.cfg.n_blocks
R = model.n_iters

print(f"  n_blocks={nb}  vocab={VOCAB}  R={R}  device={device}", flush=True)
print(f"  Wichtig: CE-Loss ist NICHT auf log(vocab) begrenzt!", flush=True)
print(f"  log({VOCAB}) = {math.log(VOCAB):.4f} Nats nur fuer UNIFORM predictor", flush=True)
print(f"  Trainiertes Modell kann bei falschen Vorhersagen Loss -> oo zeigen", flush=True)

# --- Daten laden (gleiche wie in Analyse) ---
print("\nLade TinyStories (20k Dokumente wie in Analyse) ...", flush=True)
texts, tok = load_with_text(vocab=VOCAB, max_docs=20000)
cat_wins = categorized_windows(texts, tok, seq_len=128, stride=64, min_per_cat=100)
cats = sorted(cat_wins.keys())
print(f"  Kategorien: {cats}", flush=True)

# --- Collect routing (fuer Top-Lift) ---
print("\nSammle Routing-Daten ...", flush=True)
ids = flat_ids(texts, tok)
data = FlatData(ids, tok.get_vocab_size())
routing = collect_routing(model, cat_wins, device=device)
compute_lift(routing)

# Top-5 Lift-Bloecke fuer jede Kategorie
def top_lift(cat, n):
    mean_lift = routing[cat]["lift"].mean(axis=0)  # (n_blocks,)
    return np.argsort(-mean_lift)[:n].tolist()

print("\n=== TOP-5 LIFT-BLOECKE PRO KATEGORIE ===", flush=True)
for cat in cats:
    t5 = top_lift(cat, 5)
    ml = routing[cat]["lift"].mean(axis=0)[t5]
    print(f"  {cat:16s}: {t5}  (lift: {[f'{x:.2f}' for x in ml]})", flush=True)

# --- Baseline Loss (auf cat_wins, identisch zur Analyse) ---
def eval_cat(seqs, n_batches=5, bs=32, ablate=None):
    """Mittlerer finaler Loss auf cat_wins-Sequenzen."""
    tot = 0.0; n = 0
    for start in range(0, min(n_batches * bs, len(seqs)), bs):
        batch = seqs[start:start + bs]
        toks = torch.stack([s[:-1] for s in batch]).to(device)
        tgt  = torch.stack([s[1:]  for s in batch]).to(device)
        mask = torch.ones_like(toks, dtype=torch.bool)
        logits, _ = model(toks, **({'ablate_mask': ablate} if ablate is not None else {}))
        per = iteration_losses(logits, tgt, mask)
        tot += per[-1].item(); n += 1
    return tot / max(n, 1)

print("\n=== BASELINE LOSS (cat_wins, identisch zur Analyse) ===", flush=True)
normal_loss = {}
for cat in cats:
    l = eval_cat(cat_wins[cat], n_batches=8)
    normal_loss[cat] = l
    print(f"  {cat:16s}: {l:.4f} Nats", flush=True)

# --- Ablations-Loss mit echten Top-5 Lift-Bloecken ---
print("\n=== ABLATIONS-LOSS (Top-5 Lift, ALLE Iters) ===", flush=True)
print("  Ablierend: Kausalitaet Bloecke, Evaluierend: alle Kategorien", flush=True)
target_cat = "causality"
t5_causal = top_lift(target_cat, 5)
print(f"  Ablierte Bloecke: {t5_causal}", flush=True)

abl_mask = torch.zeros(nb, dtype=torch.bool, device=device)
abl_mask[torch.tensor(t5_causal, device=device)] = True

print(f"  {'Eval-Kat':16s} | {'Normal':8s} | {'Ablated':8s} | {'Delta':8s} | {'%':6s}", flush=True)
print(f"  {'-'*60}", flush=True)
for cat in cats:
    with torch.no_grad():
        abl_loss = eval_cat(cat_wins[cat], n_batches=8, ablate=abl_mask)
    delta = abl_loss - normal_loss[cat]
    pct = delta / normal_loss[cat] * 100
    marker = "<< DIAGONAL" if cat == target_cat else ""
    print(f"  {cat:16s} | {normal_loss[cat]:8.4f} | {abl_loss:8.4f} | {delta:+8.4f} | {pct:+6.1f}%  {marker}", flush=True)

# --- Kontrollgruppe: Random Bloecke ---
print("\n=== KONTROLL-ABLATION (5 zufaellige Bloecke) ===", flush=True)
rng = np.random.default_rng(42)
rand_blocks = rng.choice(nb, size=5, replace=False).tolist()
print(f"  Ablierte Bloecke: {rand_blocks}", flush=True)
rand_mask = torch.zeros(nb, dtype=torch.bool, device=device)
rand_mask[torch.tensor(rand_blocks, device=device)] = True

with torch.no_grad():
    rand_loss = eval_cat(cat_wins[target_cat], n_batches=8, ablate=rand_mask)
print(f"  Kausalitaet: Normal={normal_loss[target_cat]:.4f}  Random-Ablation={rand_loss:.4f}  Delta={rand_loss-normal_loss[target_cat]:+.4f}", flush=True)

# --- Per-Iterations-Analyse: welche Iteration verursacht den Schaden? ---
print("\n=== PER-ITERATIONS-ANALYSE (Kausalitaet-Bloecke, nur eine Iter ablieren) ===", flush=True)
from experiments.competence_centers_exp import _per_iter_loss as per_iter_fn

seqs = cat_wins[target_cat]
with torch.no_grad():
    # Baseline
    tot = 0.0; n = 0
    for start in range(0, min(8*32, len(seqs)), 32):
        batch = seqs[start:start+32]
        toks = torch.stack([s[:-1] for s in batch]).to(device)
        tgt  = torch.stack([s[1:]  for s in batch]).to(device)
        tot += per_iter_fn(model, toks, tgt, [None]*R, device); n += 1
    base = tot / max(n, 1)
    print(f"  Baseline (kein Ablation): {base:.4f} Nats", flush=True)

    for ablate_r in range(R):
        masks = [None]*R
        masks[ablate_r] = abl_mask
        tot = 0.0; n = 0
        for start in range(0, min(8*32, len(seqs)), 32):
            batch = seqs[start:start+32]
            toks = torch.stack([s[:-1] for s in batch]).to(device)
            tgt  = torch.stack([s[1:]  for s in batch]).to(device)
            tot += per_iter_fn(model, toks, tgt, masks, device); n += 1
        l = tot / max(n, 1)
        print(f"  Nur r={ablate_r+1} abliert: {l:.4f} Nats  (delta={l-base:+.4f})", flush=True)

    # Alle abliert
    masks_all = [abl_mask]*R
    tot = 0.0; n = 0
    for start in range(0, min(8*32, len(seqs)), 32):
        batch = seqs[start:start+32]
        toks = torch.stack([s[:-1] for s in batch]).to(device)
        tgt  = torch.stack([s[1:]  for s in batch]).to(device)
        tot += per_iter_fn(model, toks, tgt, masks_all, device); n += 1
    l_all = tot / max(n, 1)
    print(f"  ALLE Iters abliert:  {l_all:.4f} Nats  (delta={l_all-base:+.4f})", flush=True)

print("\nSanity-Check abgeschlossen.", flush=True)
