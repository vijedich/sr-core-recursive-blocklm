"""HeteroMini-Training (eine Config pro Aufruf) + voller Eval + Checkpoint.

Vereinheitlicht Dense (ModelA) und Sparse (ModelC). Training auf document-contiguous
Batches, End-Gewichtung der Deep-Supervision. Danach: fuer Sparse die volle HeteroMini-
Metrik (Working Set, Domaenen-Routing, Cache, Reuse, Bytes/Token), fuer Dense nur Loss.

Beispiele:
  python -m experiments.heteromini_train --variant dense  --depth 8  --steps 800
  python -m experiments.heteromini_train --variant sparse --n_blocks 32 --R 2 --steps 800
  python -m experiments.heteromini_train --variant sparse --n_blocks 32 --R 6 \
         --core_mode per_token --steps 800
"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np
import torch

from rblm.models import iteration_losses, weighting
from rblm.heteromini import HeteroMiniData
from experiments.tinystories_exp import make_model, _lr, RESULTS
from experiments.train_dense import make_dense
import rblm.checkpoint as ckptmod


def build(variant, depth, n_blocks, k, R, core_mode, vocab, device):
    if variant == "dense":
        m, cfg = make_dense(vocab, depth, device=device)
        exp = f"hm_dense_d{depth}"
    else:
        m, cfg = make_model(vocab, n_blocks=n_blocks, k=k, R=R, device=device,
                            core_mode=core_mode)
        tag = {None: "naked", "per_token": "srcore", "core_satellite": "srsat"}[core_mode]
        exp = f"hm_{tag}_b{n_blocks}_R{R}"
    return m, cfg, exp


def train(model, data, steps, bs, seq_len, lr, device, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr(s, 200, steps))
    R = model.n_iters
    w = weighting(R, "end").to(device)
    is_sparse = hasattr(model, "bank")
    t0 = time.time(); skipped = 0
    model.train()
    for step in range(steps):
        toks, tgt, mask, _ = data.batch(bs, seq_len, device=device, mode="contiguous")
        logits, aux = model(toks)
        per = iteration_losses(logits, tgt, mask)
        loss = (w * per).sum()
        if is_sparse and aux.get("iters"):
            loss = loss + 0.01 * sum(a["lb_loss"] for a in aux["iters"]) / R
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True); skipped += 1
            continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % max(1, steps // 5) == 0 or step == 0:
            print(f"  step {step+1:5d}  Lfin={per[-1].item():.3f}  {time.time()-t0:.0f}s",
                  flush=True)
    if skipped:
        print(f"  [FINITE] {skipped} Steps uebersprungen", flush=True)


@torch.no_grad()
def _loss_per_iter(model, data, bs, seq_len, device, n_batches=8):
    model.eval()
    R = model.n_iters
    ls = torch.zeros(R, device=device); tok = 0
    for _ in range(n_batches):
        toks, tgt, mask, _ = data.batch(bs, seq_len, device=device, mode="contiguous")
        logits, _ = model(toks)
        per = iteration_losses(logits, tgt, mask)
        m = int(mask.sum()); ls += per * m; tok += m
    return (ls / max(1, tok)).cpu().tolist()


def run(variant="sparse", depth=8, n_blocks=32, k=4, R=2, core_mode=None,
        steps=800, bs=16, seq_len=128, lr=2e-3, seed=0, device="cuda",
        eval_batches=8):
    data = HeteroMiniData()
    model, cfg, exp = build(variant, depth, n_blocks, k, R, core_mode,
                            data.vocab_size, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[HM] {exp}: {n_params/1e6:.1f}M Params, vocab={data.vocab_size}, "
          f"domains={data.domains}, steps={steps}", flush=True)

    t_train = time.time()
    train(model, data, steps, bs, seq_len, lr, device, seed)
    train_s = time.time() - t_train

    metrics = {"experiment": exp, "variant": variant, "seed": seed,
               "n_params": int(n_params), "steps": steps, "train_s": round(train_s, 1)}
    if variant == "dense":
        lpi = _loss_per_iter(model, data, bs, seq_len, device, eval_batches)
        metrics["loss_per_iter"] = [round(x, 4) for x in lpi]
        metrics["Lfin"] = round(lpi[-1], 4)
        metrics["anytime"] = round(max(lpi) - min(lpi), 4)
    else:
        from experiments.heteromini_eval import evaluate
        ev = evaluate(model, data, n_batches=eval_batches, bs=bs, seq_len=seq_len,
                      device=device)
        metrics.update(ev)
        metrics["Lfin"] = ev["loss_per_iter"][-1]

    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, f"heteromini_{exp}_s{seed}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    torch.save(model.state_dict(), os.path.join(RESULTS, f"heteromini_{exp}_s{seed}_model.pt"))
    try:
        ckptmod.save(model=model, experiment=exp,
                     config={"model": {"variant": variant, "n_blocks": n_blocks, "k": k,
                                       "R": R, "dense_depth": depth, "core_mode": core_mode,
                                       "vocab_size": data.vocab_size},
                             "data": {"dataset": "heteromini_v1", "domains": data.domains}},
                     metrics={kk: metrics[kk] for kk in ("Lfin", "n_params", "steps")
                              if kk in metrics},
                     seed=seed, step=steps, val_loss=metrics["Lfin"])
    except Exception as e:
        print(f"[CKPT] Warnung: {e}", flush=True)
    print(f"[HM] fertig {exp}: Lfin={metrics['Lfin']}  -> {out_path}", flush=True)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["dense", "sparse"], default="sparse")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--n_blocks", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--R", type=int, default=2)
    ap.add_argument("--core_mode", default=None,
                    choices=[None, "per_token", "core_satellite"])
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval_batches", type=int, default=8)
    a = ap.parse_args()
    run(variant=a.variant, depth=a.depth, n_blocks=a.n_blocks, k=a.k, R=a.R,
        core_mode=a.core_mode, steps=a.steps, bs=a.bs, seq_len=a.seq_len,
        seed=a.seed, device=a.device, eval_batches=a.eval_batches)
