"""Training + evaluation loop (Phase 1)."""
from __future__ import annotations
import math
import time
import torch

from .models import build_model, iteration_losses, weighting
from .synthetic import SyntheticData
from .metrics import RoutingAccumulator


def set_seed(s):
    torch.manual_seed(s)
    import numpy as np
    np.random.seed(s)


def lr_lambda(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(model, data, tcfg, dcfg):
    model.eval()
    n = model.n_iters
    loss_sum = torch.zeros(n)
    tok = 0
    is_C = model.cfg.variant == "C"
    acc = RoutingAccumulator(model.cfg.n_blocks, n, dcfg.n_regimes) if is_C else None
    for _ in range(tcfg.eval_batches):
        toks, tgt, mask, reg = data.batch(tcfg.batch_size, tcfg.seq_len, tcfg.device)
        logits, aux = model(toks)
        per = iteration_losses(logits, tgt, mask)         # (n,)
        m = int(mask.sum())
        loss_sum += per.detach() * m
        tok += m
        if is_C:
            acc.update(aux, mask, reg)
    model.train()
    per_iter = (loss_sum / max(1, tok)).tolist()
    out = {
        "loss_per_iter": per_iter,
        "ppl_per_iter": [float(math.exp(min(20, x))) for x in per_iter],
        "final_loss": per_iter[-1],
    }
    if is_C:
        out["routing"] = acc.finalize()
    return out


def run_training(cfg, log_fn=print):
    set_seed(cfg.train.seed)
    data = SyntheticData(cfg.data.n_regimes, cfg.data.base, seed=12345)  # fixed data stream
    cfg.model.vocab_size = data.vocab_size
    model = build_model(cfg.model).to(cfg.train.device)
    n_params = sum(p.numel() for p in model.parameters())
    core_params = _core_param_count(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, cfg.train.warmup, cfg.train.steps))
    w = weighting(model.n_iters, cfg.train.iter_loss_weighting).to(cfg.train.device)

    history = {"step": [], "train_loss": [], "eval": []}
    t0 = time.time()
    for step in range(cfg.train.steps):
        toks, tgt, mask, reg = data.batch(
            cfg.train.batch_size, cfg.train.seq_len, cfg.train.device)
        logits, aux = model(toks)
        per = iteration_losses(logits, tgt, mask)
        loss = (w * per).sum()
        if cfg.model.variant == "C":
            lb = sum(a["lb_loss"] for a in aux["iters"]) / len(aux["iters"])
            loss = loss + cfg.train.lb_loss_weight * lb
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        sched.step()

        if (step + 1) % cfg.train.eval_every == 0 or step == 0:
            ev = evaluate(model, data, cfg.train, cfg.data)
            history["step"].append(step + 1)
            history["train_loss"].append(float(loss.detach()))
            history["eval"].append(ev)
            extra = ""
            if "routing" in ev:
                r = ev["routing"]
                extra = (f" | Hroute={r['router_entropy_norm'][0]:.2f}"
                         f" dead={r['dead_blocks']} MI={r['regime_block_MI_norm']:.2f}")
            log_fn(f"[{cfg.name}] step {step+1:4d}  "
                   f"L1={ev['loss_per_iter'][0]:.3f}  "
                   f"L{model.n_iters}={ev['loss_per_iter'][-1]:.3f}{extra}")

    final = evaluate(model, data, cfg.train, cfg.data)
    return {
        "name": cfg.name,
        "variant": cfg.model.variant,
        "n_params": n_params,
        "core_params": core_params,
        "n_iters": model.n_iters,
        "block_apps": [model.block_apps_at_iter(r) for r in range(1, model.n_iters + 1)],
        "history": history,
        "final": final,
        "wall_s": time.time() - t0,
    }


def _core_param_count(model):
    shared = set()
    for name, p in model.named_parameters():
        if name.startswith(("emb", "pos", "context", "norm_out", "head")):
            shared.add(id(p))
    return sum(p.numel() for p in model.parameters() if id(p) not in shared)
