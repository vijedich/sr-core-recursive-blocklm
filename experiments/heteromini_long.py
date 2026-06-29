"""HeteroMini-Training EINER Config in 5k-Segmenten mit fortsetzbaren Snapshots.

Idee (Viktor): erst 5k Steps, Snapshot, auswerten — wenn mehr gebraucht wird, von genau
diesem Snapshot weitere 5k trainieren. Kein 50k-Marathon auf der 2060.

- max_steps ist das ABSOLUTE Zielsteps. Existiert ein Continuation-Snapshot
  (results/hm_cont_{exp}_s0.pt), wird automatisch von dort fortgesetzt (Modell+Optimizer+
  Scheduler+RNG). Ist sein step schon >= max_steps -> nichts zu tun.
- Milestone-Auswertung (inline, kein Reload): WS, reuse_p90, K8 bytes/token, anytime, dead,
  Lfin -> Verlaufstabelle results/hm_traj_{exp}_s0.json (wird angehaengt).
- lr: warmup(200) + Cosine ueber lr_horizon (Default 10000), damit ein zweites 5k-Segment
  glatt weiter-decayed.

Nutzung:
  python -m experiments.heteromini_long --variant sparse --n_blocks 32 --R 6 \
         --core_mode per_token --max_steps 5000
  # spaeter weitere 5k:
  python -m experiments.heteromini_long ... --max_steps 10000
"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np
import torch

from rblm.models import iteration_losses, weighting
from rblm.heteromini import HeteroMiniData
from rblm.reg_losses import (core_locality_loss, soft_full_jaccard_loss,
                              soft_sharp_jaccard_loss, mean_core_overlap,
                              router_entropy, router_entropy_loss,
                              router_entropy_loss_targeted)
from experiments.tinystories_exp import make_model, _lr, RESULTS
from experiments.train_dense import make_dense
from experiments.heteromini_eval import _collect, _streams, _reuse_distance
from experiments.offload_sim import sim_lru, _flatten, dense_streams


def build(variant, depth, n_blocks, k, R, core_mode, vocab, device,
          d_model=256, block_hidden=512):
    if variant == "dense":
        m, cfg = make_dense(vocab, depth, device=device)
        return m, cfg, f"hm_dense_d{depth}"
    m, cfg = make_model(vocab, n_blocks=n_blocks, k=k, R=R, device=device, core_mode=core_mode,
                        d_model=d_model, block_hidden=block_hidden)
    tag = {None: "naked", "per_token": "srcore", "core_satellite": "srsat"}[core_mode]
    k_tag = f"_k{k}" if k != 4 else ""
    # Blockgroesse (d_model/block_hidden) in den Namen, wenn nicht-Default — sonst Kollision
    # bei gleichem n/k/R. Default (256/512) => kein Suffix => alle Alt-Checkpoints namensgleich.
    w_tag = f"_d{d_model}h{block_hidden}" if (d_model, block_hidden) != (256, 512) else ""
    return m, cfg, f"hm_{tag}_b{n_blocks}{k_tag}_R{R}{w_tag}"


@torch.no_grad()
def milestone_row(model, variant, data, step, n_batches, bs, seq_len, device, K=8):
    model.eval()
    block_params = sum(p.numel() for p in
                       (model.blocks[0] if variant == "dense" else model.bank.blocks[0]).parameters())
    if variant == "dense":
        D = model.cfg.dense_depth
        R = model.n_iters
        ls = torch.zeros(R, device=device); tok = 0
        for _ in range(n_batches):
            toks, tgt, mask, _ = data.batch(bs, seq_len, device=device, mode="contiguous")
            lg, _ = model(toks); per = iteration_losses(lg, tgt, mask)
            m = int(mask.sum()); ls += per * m; tok += m
        lpi = (ls / max(1, tok)).cpu().tolist()
        streams = dense_streams(D, n_batches * bs, seq_len)
        n_tokens = n_batches * bs * seq_len
        ws, reuse_p90, dead = float(D), float(D), 0
    else:
        traces, _, losses, R, k = _collect(model, data, "contiguous", n_batches, bs, seq_len, device)
        lpi = losses
        streams = _streams(traces)
        n_tokens = sum(tr.shape[1] * tr.shape[2] for tr in traces)
        wsl, used = [], set()
        for tr in traces:
            Rr, B, T, kk = tr.shape
            a = tr.transpose(1, 2, 0, 3).reshape(B * T, Rr * kk)
            wsl.extend(len(np.unique(r)) for r in a)
            used.update(np.unique(tr).tolist())
        ws = float(np.mean(wsl))
        reuse_p90 = _reuse_distance(streams)["p90"]
        dead = int(model.cfg.n_blocks - len(used))
    flat = _flatten(streams)
    k8 = round(sim_lru(flat, K) / n_tokens * block_params * 2 / 1024, 1)
    model.train()
    return {"step": step, "Lfin": round(lpi[-1], 4), "anytime": round(max(lpi) - min(lpi), 4),
            "WS": round(ws, 2), "reuse_p90": round(reuse_p90, 1),
            "k8_kb_per_token": k8, "dead_blocks": dead}


def run(variant="sparse", depth=24, n_blocks=32, k=4, R=6, core_mode="per_token",
        max_steps=5000, lr_horizon=10000, bs=16, seq_len=128, lr=2e-3, seed=0,
        device="cuda", eval_batches=6, lambda_core=0.0, core_reg_mode="soft_full",
        core_reg_alpha=2.0, exp_tag="", cont_src="",
        noise_std=None, lambda_entropy=0.0, target_entropy=None, lambda_target=0.0,
        d_model=256, block_hidden=512):
    data = HeteroMiniData()
    model, cfg, exp_base = build(variant, depth, n_blocks, k, R, core_mode, data.vocab_size, device,
                                 d_model=d_model, block_hidden=block_hidden)
    exp = exp_base + exp_tag
    is_sparse = hasattr(model, "bank")
    n_params = sum(p.numel() for p in model.parameters())
    cont_path = os.path.join(RESULTS, f"hm_cont_{exp}_s{seed}.pt")
    traj_path = os.path.join(RESULTS, f"hm_traj_{exp}_s{seed}.json")
    # cont_src: wenn gesetzt, wird DARAUS geladen (statt cont_path)
    load_path  = cont_src if cont_src and os.path.exists(cont_src) else cont_path

    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr(s, 200, lr_horizon))

    start = 0
    rows = []
    if os.path.exists(load_path):
        ck = torch.load(load_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"]); torch.set_rng_state(ck["rng_torch"].cpu())
        if ck.get("rng_cuda") is not None and device != "cpu":
            torch.cuda.set_rng_state(ck["rng_cuda"].cpu())
        if ck.get("rng_numpy") is not None: np.random.set_state(ck["rng_numpy"])
        start = int(ck["step"])
        if load_path == cont_path and os.path.exists(traj_path):
            rows = json.load(open(traj_path, encoding="utf-8")).get("rows", [])
        src_label = "cont_src" if load_path == cont_src else "cont"
        print(f"[HM-long] {exp}: fortgesetzt ab step {start} ({src_label})", flush=True)
    # noise_std-Override: nach dem Laden, damit Checkpoint-Architektur unberuehrt bleibt
    if noise_std is not None and is_sparse:
        model.bank.noise_std = noise_std
        print(f"[HM-long] {exp}: noise_std -> {noise_std}", flush=True)
    if start >= max_steps:
        print(f"[HM-long] {exp}: bereits bei step {start} >= {max_steps} — nichts zu tun.", flush=True)
        return rows

    # Milestones: Vielfache von 2500 + immer max_steps am Ende
    grid = ([1000] if start < 1000 else []) + list(range(2500, max_steps + 1, 2500)) + [max_steps]
    milestones = sorted(s for s in set(grid) if start < s <= max_steps)
    reg_label = ""
    if lambda_core > 0:
        reg_label += (f"  lambda_core={lambda_core} mode={core_reg_mode}"
                      + (f" alpha={core_reg_alpha}" if "sharp" in core_reg_mode else ""))
    if lambda_entropy > 0:
        reg_label += f"  lambda_entropy={lambda_entropy}"
    if target_entropy is not None and lambda_target > 0:
        reg_label += f"  target_entropy={target_entropy} lambda_target={lambda_target}"
    if noise_std is not None:
        reg_label += f"  noise_std={noise_std}"
    print(f"[HM-long] {exp}: {n_params/1e6:.1f}M Params, train {start}->{max_steps}, "
          f"lr_horizon={lr_horizon}, milestones={milestones}{reg_label}", flush=True)

    def _save_snapshot(cur_step):
        """Fortsetzbaren Snapshot schreiben — an JEDEM Milestone (Crash-Recovery), nicht nur
        am Ende. Ein Reboot/Kill mid-run resumiert damit vom letzten Milestone statt von 0."""
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "rng_torch": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state() if device != "cpu" else None,
                    "rng_numpy": np.random.get_state(), "step": cur_step,
                    "config": {"variant": variant, "n_blocks": n_blocks, "k": k, "R": R,
                               "depth": depth, "core_mode": core_mode,
                               "vocab_size": data.vocab_size,
                               "d_model": d_model, "block_hidden": block_hidden,
                               "noise_std": noise_std,
                               "target_entropy": target_entropy,
                               "lambda_target": lambda_target}}, cont_path)

    Rn = model.n_iters
    w = weighting(Rn, "end").to(device)
    t0 = time.time(); skipped = 0
    # Laufende Logging-Akkumulatoren (zwischen Milestones)
    soft_ov_acc = hard_ov_acc = ent_acc = log_n = 0.0
    model.train()
    for step in range(start, max_steps):
        toks, tgt, mask, _ = data.batch(bs, seq_len, device=device, mode="contiguous")
        logits, aux = model(toks)
        per = iteration_losses(logits, tgt, mask)
        loss = (w * per).sum()
        if is_sparse and aux.get("iters"):
            loss = loss + 0.01 * sum(a["lb_loss"] for a in aux["iters"]) / Rn
            a0 = aux["iters"][0]
            rp = a0.get("router_probs")           # (B, T, n_blocks), in-graph or None
            if lambda_core > 0.0 and rp is not None:
                if core_reg_mode == "soft_full":
                    closs = soft_full_jaccard_loss(rp)
                elif core_reg_mode == "soft_sharp":
                    closs = soft_sharp_jaccard_loss(rp, alpha=core_reg_alpha)
                else:  # legacy gate-overlap mode
                    B, T = toks.shape
                    closs = core_locality_loss(
                        a0["route_idx"], a0["route_gates"],
                        model.bank.n_blocks, B, T)
                loss = loss + lambda_core * closs
                soft_ov_acc += float(1.0 - closs.item())
            if lambda_entropy > 0.0 and rp is not None:
                loss = loss + lambda_entropy * router_entropy_loss(rp)
            if target_entropy is not None and lambda_target > 0.0 and rp is not None:
                loss = loss + lambda_target * router_entropy_loss_targeted(rp, target_entropy)
            # Logging-Metriken (kein Grad)
            with torch.no_grad():
                if rp is not None:
                    hard_ov_acc += mean_core_overlap(a0["topk_idx"])
                    ent_acc     += router_entropy(rp)
                    log_n       += 1
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True); skipped += 1; continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) in milestones:
            row = milestone_row(model, variant, data, step + 1, eval_batches, bs, seq_len, device)
            n = max(1, log_n)
            row["soft_overlap_train"] = round(soft_ov_acc / n, 4)
            row["hard_overlap_train"] = round(hard_ov_acc / n, 4)
            row["router_entropy"]     = round(ent_acc     / n, 4)
            soft_ov_acc = hard_ov_acc = ent_acc = log_n = 0.0
            rows.append(row)
            with open(traj_path, "w", encoding="utf-8") as f:
                json.dump({"experiment": exp, "variant": variant, "n_params": int(n_params),
                           "lambda_core": lambda_core, "core_reg_mode": core_reg_mode,
                           "rows": rows}, f, indent=2)
            print(f"  [{exp}] step {step+1}: Lfin={row['Lfin']} WS={row['WS']} "
                  f"reuseP90={row['reuse_p90']} K8={row['k8_kb_per_token']}KB "
                  f"anytime={row['anytime']} dead={row['dead_blocks']} "
                  f"soft_ov={row['soft_overlap_train']} hard_ov={row['hard_overlap_train']} "
                  f"ent={row['router_entropy']}  ({time.time()-t0:.0f}s)",
                  flush=True)
            model.train()
            _save_snapshot(step + 1)   # Crash-fester Milestone-Snapshot

    # Finaler Snapshot (redundant zum letzten Milestone, aber sicher falls max_steps kein Milestone)
    _save_snapshot(max_steps)
    if skipped:
        print(f"  [{exp}] {skipped} nicht-finite Steps uebersprungen", flush=True)
    print(f"[HM-long] fertig {exp} @ step {max_steps} -> Snapshot {cont_path}", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["dense", "sparse"], default="sparse")
    ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--n_blocks", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=256,
                    help="Residual-Breite (sparse). Nicht-Default => Namens-Suffix _d{d}h{h}")
    ap.add_argument("--block_hidden", type=int, default=512,
                    help="Block-MLP-Hidden (sparse). Steuert mit d_model die Blockgroesse")
    ap.add_argument("--R", type=int, default=6)
    ap.add_argument("--core_mode", default=None,
                    choices=["per_token", "core_satellite"],
                    help="weglassen = Naked Sparse (core_mode=None)")
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--lr_horizon", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lambda_core", type=float, default=0.0,
                    help="Core-Locality-Regularizer Gewicht (0 = aus)")
    ap.add_argument("--core_reg_mode", default="soft_full",
                    choices=["soft_full", "soft_sharp", "gate_overlap"],
                    help="Regularizer-Variante")
    ap.add_argument("--core_reg_alpha", type=float, default=2.0,
                    help="Schaerfe-Exponent fuer soft_sharp (default 2.0)")
    ap.add_argument("--exp_tag", default="",
                    help="Suffix fuer exp-Name und Cont-Snapshot")
    ap.add_argument("--cont_src", default="",
                    help="Optionaler Quell-Cont-Snapshot (statt auto-abgeleiteten Pfad)")
    ap.add_argument("--noise_std", type=float, default=None,
                    help="Router-Noise-Override nach dem Laden (None = Modell-Default 0.3)")
    ap.add_argument("--lambda_entropy", type=float, default=0.0,
                    help="Entropie-Minimierungs-Gewicht auf R1-router_probs (0 = aus)")
    ap.add_argument("--target_entropy", type=float, default=None,
                    help="Ziel-Entropie fuer bounded consolidation (None = aus)")
    ap.add_argument("--lambda_target", type=float, default=0.0,
                    help="Gewicht fuer target-entropy-Loss (0 = aus)")
    a = ap.parse_args()
    run(variant=a.variant, depth=a.depth, n_blocks=a.n_blocks, k=a.k, R=a.R,
        core_mode=a.core_mode, max_steps=a.max_steps, lr_horizon=a.lr_horizon,
        bs=a.bs, seq_len=a.seq_len, seed=a.seed, device=a.device,
        lambda_core=a.lambda_core, core_reg_mode=a.core_reg_mode,
        core_reg_alpha=a.core_reg_alpha, exp_tag=a.exp_tag, cont_src=a.cont_src,
        noise_std=a.noise_std, lambda_entropy=a.lambda_entropy,
        target_entropy=a.target_entropy, lambda_target=a.lambda_target,
        d_model=a.d_model, block_hidden=a.block_hidden)
