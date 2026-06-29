"""Dense-Baseline (ModelA) auf TinyStories — gleiches Rezept/Backbone wie ModelC.

Zweck: ehrliche Dense-Referenz fuer den CPU-Benchmark (dense vs. sparse). ModelA = Stapel
aus `dense_depth` distinkten Bloecken, jeder GENAU EINMAL angewandt (keine Routing-/Sparse-
Logik). Deeply supervised (Readout nach jedem Block) → loss_per_iter gibt die Qualitaet pro
Tiefe.

Gleiche Hyperparameter wie die ModelC-Laeufe: 3000 Schritte, bs=32, seq=128, lr=2e-3,
AdamW wd=0.01, Cosine-Warmup=200, End-Gewichtung der Deep-Supervision.

Nutzung:
  python -m experiments.train_dense --dense_depth 24 --device cuda
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
import torch

from rblm.config import ModelConfig
from rblm.models import build_model, iteration_losses, weighting
from experiments.tinystories_exp import load_data, _lr, RESULTS


def make_dense(vocab_size, dense_depth, d_model=256, block_hidden=512, device="cpu"):
    cfg = ModelConfig(
        vocab_size=vocab_size, d_model=d_model, block_hidden=block_hidden,
        n_heads=4, context_layers=1, max_len=256,
        variant="A", dense_depth=dense_depth,
    )
    return build_model(cfg).to(device), cfg


@torch.no_grad()
def _eval(model, data, bs, seq_len, device, w, n_batches=8):
    model.eval()
    D = model.n_iters
    loss_sum = torch.zeros(D, device=device)
    tok = 0
    for _ in range(n_batches):
        toks, tgt, mask, _ = data.batch(bs, seq_len, device=device)
        logits, _ = model(toks)
        per = iteration_losses(logits, tgt, mask)
        m = int(mask.sum())
        loss_sum += per.detach() * m
        tok += m
    losses = (loss_sum / max(1, tok)).cpu().tolist()
    return losses


def train(dense_depth=24, steps=3000, bs=32, seq_len=128, lr=2e-3, device="cuda",
          seed=0, eval_every=500):
    data, _ = load_data()
    model, cfg = make_dense(data.vocab_size, dense_depth, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Dense] ModelA depth={dense_depth}: {n_params/1e6:.1f}M Parameter, "
          f"vocab={data.vocab_size}, device={device}", flush=True)

    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr(s, 200, steps))
    D = model.n_iters
    w = weighting(D, "end").to(device)

    exp_name = f"dense_d{dense_depth}"

    t0 = time.time(); skipped = 0
    model.train()
    for step in range(steps):
        toks, tgt, mask, _ = data.batch(bs, seq_len, device=device)
        logits, _ = model(toks)
        per = iteration_losses(logits, tgt, mask)
        loss = (w * per).sum()
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True); skipped += 1
            print(f"  [FINITE] nicht-finiter Loss step {step+1} uebersprungen", flush=True)
            continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if (step + 1) % eval_every == 0 or step == 0:
            lpi = _eval(model, data, bs, seq_len, device, w)
            print(f"  step {step+1:5d}  L1={lpi[0]:.3f}  Lfin={lpi[-1]:.3f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
            model.train()
    if skipped:
        print(f"  [FINITE] {skipped} Steps uebersprungen", flush=True)

    lpi = _eval(model, data, bs, seq_len, device, w, n_batches=12)
    print(f"[Dense] FERTIG depth={dense_depth}: loss_per_iter={[round(x,3) for x in lpi]}  "
          f"Lfin={lpi[-1]:.4f}", flush=True)

    # Sofort-Sicherung + versionierter Checkpoint
    os.makedirs(RESULTS, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(RESULTS, f"{exp_name}_s{seed}_model.pt"))
    try:
        import rblm.checkpoint as _ckpt
        _ckpt.save(
            model=model, experiment=exp_name,
            config={"model": {"vocab_size": data.vocab_size, "d_model": cfg.d_model,
                              "block_hidden": cfg.block_hidden, "n_heads": cfg.n_heads,
                              "context_layers": cfg.context_layers, "max_len": cfg.max_len,
                              "variant": "A", "dense_depth": dense_depth, "tie_head": cfg.tie_head},
                    "training": {"steps": steps, "bs": bs, "seq_len": seq_len, "lr": lr,
                                 "weight_decay": 0.01, "warmup": 200, "loss_weighting": "end"},
                    "data": {"dataset": "tinystories", "max_docs": 20000, "vocab_size": 8000,
                             "seq_len": seq_len}},
            metrics={"Lfin": lpi[-1], "L1": lpi[0], "loss_per_iter": lpi,
                     "anytime_delta": round(max(lpi) - min(lpi), 4),
                     "n_params": n_params, "training_steps": steps},
            seed=seed, step=steps, val_loss=lpi[-1])
    except Exception as e:
        print(f"[CKPT] Warnung: {e}", flush=True)

    return lpi


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense_depth", type=int, default=24)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    train(dense_depth=a.dense_depth, steps=a.steps, bs=a.bs, seq_len=a.seq_len,
          seed=a.seed, device=a.device)
