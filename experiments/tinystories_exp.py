"""Experiment 3 — Tunnel-Validierung auf natuerlicher Sprache (TinyStories)

Kernfragen:
  F1  Lernt das Modell echte Sprache?         Verlust << ln(vocab) nach Training
  F2  Tunnel-Lokalitaet auf nat. Sprache?     Jaccard(r->r+1) vs. Zufalls-Routing
  F3  Hub-Struktur?                           Gini-Koeffizient der Blocknutzung
  F4  Einzigartige Bloecke/Token?             Transfer-Schaetzung vs. Layer-Offloading
  F5  Miss-Rate unter Cache-Budget?           Gelernt vs. Zufalls-Routing-Kontrolle

Voraussetzungen:
  pip install datasets tokenizers
  Netzwerkzugang zu huggingface.co

Nutzung:
  python -m experiments.tinystories_exp                    # Vollauf (GPU, 10k Schritte)
  python -m experiments.tinystories_exp --smoke            # Schnelltest (200 Schritte)
  python -m experiments.tinystories_exp --steps 5000       # Kuerzer laufen lassen
"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rblm.config import ModelConfig, DataConfig
from rblm.models import build_model, iteration_losses, weighting
from rblm.metrics import RoutingAccumulator
from rblm.tinystories import try_build_tinystories

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


# ---------- data ----------

def load_data(ts_vocab=8000, ts_max_docs=20000):
    dcfg = DataConfig(kind="tinystories", ts_vocab=ts_vocab, ts_max_docs=ts_max_docs)
    print(f"[Exp3] Lade TinyStories (vocab={ts_vocab}, max_docs={ts_max_docs}) ...", flush=True)
    data, msg = try_build_tinystories(dcfg)
    if data is None:
        raise RuntimeError(
            f"TinyStories konnte nicht geladen werden: {msg}\n"
            "Loesung: pip install datasets tokenizers  und  Netzwerkzugang sicherstellen."
        )
    n_tok = data.ids.numel()
    print(f"[Exp3] {n_tok:,} Tokens geladen, Vokabular: {data.vocab_size}", flush=True)
    return data, dcfg


# ---------- model ----------

def make_model(vocab_size, n_blocks=64, k=4, R=6,
               d_model=256, block_hidden=512, device="cpu", core_mode=None):
    cfg = ModelConfig(
        vocab_size=vocab_size, d_model=d_model, block_hidden=block_hidden,
        n_heads=4, context_layers=1, max_len=256,
        variant="C", n_blocks=n_blocks, k_active=k, routed_iters=R,
        key_dim=64, router_noise_std=0.3, core_mode=core_mode,
    )
    return build_model(cfg).to(device), cfg


# ---------- training ----------

def _lr(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * p))


def train(model, data, steps=10000, bs=32, seq_len=128, lr=2e-3,
          device="cpu", eval_every=500, seed=0,
          div_w=0.0, coord_w=0.0, diverse=False,
          diverse_until: int = 0, diverse_from_iter: int = 0,
          check_finite: bool = True, strict_finite: bool = False):
    """
    diverse_until: Schritte mit aktiver Diversity; danach wird diverse abgeschaltet.
      0 (default) = diverse-Flag gilt fuer den gesamten Lauf (bisheriges Verhalten).
    diverse_from_iter: Erst ab dieser 0-basierten Iterations-Nummer wird Diversity
      angewendet. 0 = alle Iterationen, 2 = nur r3-r6 (Curriculum-Variante C).
    check_finite: Prueft Loss und Gradienten auf NaN/Inf vor jedem Optimizer-Schritt.
      Standard AN. Bei nicht-finitem Loss/Gradienten wird der Optimizer-Schritt
      UEBERSPRUNGEN (zero_grad + continue) statt den Lauf abzubrechen — so ueberlebt
      ein einzelner schlechter Batch (Seed-2-Stabilitaetsfall). Step-Nummer wird
      geloggt. Bei >50 Skips in Folge: echte Divergenz, Abbruch.
    strict_finite: Statt Ueberspringen sofort RuntimeError werfen (harte Diagnose).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _lr(s, warmup=200, total=steps))
    R = model.n_iters
    w = weighting(R, "end").to(device)
    history = []
    t0 = time.time()
    last_grad_norm = 0.0
    skipped_total = 0
    skipped_streak = 0
    model.train()
    for step in range(steps):
        toks, tgt, mask, _ = data.batch(bs, seq_len, device=device)
        # Curriculum: Diversity nur fuer die ersten diverse_until Schritte
        diverse_now = diverse if diverse_until == 0 else (step < diverse_until)
        logits, aux = model(toks, diverse_train=diverse_now,
                            diverse_from_iter=diverse_from_iter)
        per = iteration_losses(logits, tgt, mask)
        lb  = sum(a["lb_loss"] for a in aux["iters"]) / R
        div = aux["diversity_loss"]    # soft overlap metric (monitoring)
        crd = aux["coord_loss"]        # block coordinate repulsion, Phase-3
        # div_w multiplies soft overlap loss; hard diversity uses diverse_train flag
        loss = (w * per).sum() + 0.01 * lb + div_w * div + coord_w * crd

        if check_finite and not torch.isfinite(loss):
            msg = (f"  [FINITE] Nicht-finiter Loss bei step {step+1}: {loss.item()}"
                   f"  per={[f'{p:.4f}' for p in per.tolist()]}")
            print(msg, flush=True)
            if strict_finite:
                raise RuntimeError(f"Non-finite loss at step {step+1}")
            # Robust: schlechten Batch verwerfen, Gewichte unveraendert, weiter.
            opt.zero_grad(set_to_none=True)
            skipped_total += 1
            skipped_streak += 1
            if skipped_streak > 50:
                raise RuntimeError(
                    f"Divergenz: >50 nicht-finite Steps in Folge ab step {step+1}")
            continue

        opt.zero_grad()
        loss.backward()

        if check_finite:
            bad = [(n, p.grad.abs().max().item())
                   for n, p in model.named_parameters()
                   if p.grad is not None and not torch.isfinite(p.grad).all()]
            if bad:
                print(f"  [FINITE] Nicht-finite Gradienten bei step {step+1}: "
                      f"{[(n, v) for n, v in bad[:5]]}", flush=True)
                if strict_finite:
                    raise RuntimeError(
                        f"Non-finite gradients at step {step+1}: {[n for n,_ in bad]}")
                # Robust: Gradienten verwerfen, Gewichte unveraendert, weiter.
                opt.zero_grad(set_to_none=True)
                skipped_total += 1
                skipped_streak += 1
                if skipped_streak > 50:
                    raise RuntimeError(
                        f"Divergenz: >50 nicht-finite Steps in Folge ab step {step+1}")
                continue
        skipped_streak = 0

        last_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        sched.step()

        if (step + 1) % eval_every == 0 or step == 0:
            ev = _quick_eval(model, data, bs, seq_len, device, R, w)
            elapsed = time.time() - t0
            history.append({"step": step + 1, **ev, "grad_norm": round(last_grad_norm, 4)})
            div_str = f"  Div={ev['div']:.3f}" if (div_w > 0 or diverse) else ""
            print(f"  step {step+1:5d}  L1={ev['L1']:.3f}  Lfin={ev['Lfin']:.3f}"
                  f"  Jacc={ev['jacc']:.3f}{div_str}"
                  f"  |grad|={last_grad_norm:.3f}  {elapsed:.0f}s", flush=True)
            model.train()
    if skipped_total:
        print(f"  [FINITE] {skipped_total} nicht-finite Steps uebersprungen "
              f"(Lauf fortgesetzt statt abgebrochen).", flush=True)
    return history


@torch.no_grad()
def _quick_eval(model, data, bs, seq_len, device, R, w, n_batches=4):
    model.eval()
    loss_sum = torch.zeros(R, device=device)
    tok = 0
    jacc_sum = 0.0
    jacc_cnt = 0
    div_sum = 0.0
    for _ in range(n_batches):
        toks, tgt, mask, _ = data.batch(bs, seq_len, device=device)
        logits, aux = model(toks)
        per = iteration_losses(logits, tgt, mask)
        m = int(mask.sum())
        loss_sum += per.detach() * m
        tok += m
        # Jaccard between last two iterations
        if R > 1:
            idx_a = aux["iters"][-2]["topk_idx"].reshape(-1, model.cfg.k_active)
            idx_b = aux["iters"][-1]["topk_idx"].reshape(-1, model.cfg.k_active)
            N, k = idx_a.shape
            nb = model.cfg.n_blocks
            mem_a = torch.zeros(N, nb, dtype=torch.bool, device=device)
            mem_b = torch.zeros(N, nb, dtype=torch.bool, device=device)
            mem_a.scatter_(1, idx_a, True)
            mem_b.scatter_(1, idx_b, True)
            inter = (mem_a & mem_b).sum(dim=1).float()
            union = (mem_a | mem_b).sum(dim=1).float()
            jacc_sum += (inter / union.clamp(min=1)).sum().item()
            jacc_cnt += N
        # Soft-overlap diversity metric (Phase 3)
        div_sum += aux["diversity_loss"].item()
    losses = (loss_sum / max(1, tok)).cpu().tolist()
    return {
        "L1": losses[0], "Lfin": losses[-1],
        "loss_per_iter": losses,
        "jacc": jacc_sum / max(1, jacc_cnt),
        "div": div_sum / n_batches,
    }


# ---------- full evaluation ----------

@torch.no_grad()
def full_eval(model, data, n_blocks, n_batches=12, bs=32, seq_len=128, device="cpu"):
    """Tunnel metrics + trace extraction for cache simulation."""
    model.eval()
    R = model.n_iters
    k = model.cfg.k_active
    acc = RoutingAccumulator(n_blocks, R, n_regimes=1)
    traces = []       # (n_batches, R, B, T, k) for cache sim
    unique_counts = []

    for _ in range(n_batches):
        toks, tgt, mask, _ = data.batch(bs, seq_len, device=device)
        reg = torch.zeros(bs, dtype=torch.long, device=device)
        logits, aux = model(toks)
        # RoutingAccumulator keeps state on CPU; move aux tensors before update
        aux_cpu = [{"topk_idx": a["topk_idx"].cpu(),
                    "full_probs": a["full_probs"].cpu(),
                    "usage": a["usage"].cpu(),
                    "lb_loss": a["lb_loss"]} for a in aux["iters"]]
        acc.update({"iters": aux_cpu}, mask.cpu(), reg.cpu())
        # Extract traces
        batch_trace = np.stack(
            [a["topk_idx"].numpy() for a in aux_cpu])  # (R, B, T, k)
        traces.append(batch_trace)
        # Unique blocks per token (work on CPU)
        all_idx = torch.stack(
            [a["topk_idx"].reshape(-1, k) for a in aux_cpu], dim=0)  # (R, N, k)
        flat = all_idx.permute(1, 0, 2).reshape(-1, R * k)           # (N, R*k)
        flat_mask = mask.reshape(-1).cpu()
        flat_masked = flat[flat_mask].cpu()
        n_tok = flat_masked.shape[0]
        mem = torch.zeros(n_tok, n_blocks, dtype=torch.bool)
        for col in range(R * k):
            mem.scatter_(1, flat_masked[:, col:col+1], True)
        unique_counts.append(mem.sum(dim=1).float().numpy())

    routing = acc.finalize()
    traces_arr = np.stack(traces)   # (n_batches, R, B, T, k)
    counts = np.concatenate(unique_counts)

    unique_stats = {
        "mean": float(counts.mean()),
        "p50": float(np.median(counts)),
        "p90": float(np.percentile(counts, 90)),
        "max_possible": R * k,
        "reduction_64": round(64.0 / max(1.0, float(counts.mean())), 2),
        "reduction_256": round(256.0 / max(1.0, float(counts.mean())), 2),
        "reduction_1000": round(1000.0 / max(1.0, float(counts.mean())), 2),
    }
    return routing, unique_stats, traces_arr


# ---------- cache simulation ----------

def _simulate_lru(steps, capacity, n_blocks):
    """Single-stream LRU cache simulation. steps: list of int arrays."""
    cache = []
    total = misses = 0
    for req in steps:
        for b in req:
            total += 1
            if b in cache:
                cache.remove(b); cache.append(b)
            else:
                misses += 1; cache.append(b)
                while len(cache) > capacity:
                    cache.pop(0)
    return misses / max(1, total)


def cache_sim(traces_arr, n_blocks, caps, rng):
    """Miss-rate vs cache budget, learned vs random routing."""
    nb, R, B, T, k = traces_arr.shape
    # Build per-sequence step streams
    learned_steps = []
    random_steps = []
    for bi in range(nb):
        for s in range(B):
            seq_l, seq_r = [], []
            for t in range(T):
                for r in range(R):
                    seq_l.append(np.unique(traces_arr[bi, r, s, t]))
                    seq_r.append(rng.choice(n_blocks, size=k, replace=False))
            learned_steps.append(seq_l)
            random_steps.append(seq_r)

    results = {"caps": caps, "learned": [], "random": []}
    for cap in caps:
        mr_l = np.mean([_simulate_lru(s, cap, n_blocks) for s in learned_steps])
        mr_r = np.mean([_simulate_lru(s, cap, n_blocks) for s in random_steps])
        results["learned"].append(round(float(mr_l), 4))
        results["random"].append(round(float(mr_r), 4))
    return results


# ---------- hub Gini ----------

def gini(freqs):
    freqs = np.sort(np.array(freqs, dtype=float))
    n = len(freqs)
    if freqs.sum() == 0:
        return 0.0
    return float((2 * (np.arange(1, n + 1) * freqs).sum() / (n * freqs.sum())) - (n + 1) / n)


# ---------- per-iteration ablation ----------

@torch.no_grad()
def iteration_diagnostics(model, data, n_batches=6, bs=32, seq_len=128, device="cpu",
                          run_forced=True):
    """Vier Messungen ohne Retraining:
      A) Relative Zustandsaenderung ||h_r - h_{r-1}|| / ||h_{r-1}||
      B) Jaccard-Matrix aller Iterationspaare (nicht nur benachbarte)
      C) Erzwungene Diversitaet: Loss wenn Bloecke aus r-1 in r verboten sind
         (run_forced=False ueberspringt C — verhindert CUDA-Haenger bei Diversity-Training)
      D) Output-KL: KL(p(r+1) || p(r)) zwischen aufeinanderfolgenden Ausgaben
    """
    model.eval()
    R = model.n_iters
    k = model.cfg.k_active
    nb = model.cfg.n_blocks
    bank = model.bank

    sc_acc  = np.zeros(R)
    kl_acc  = np.zeros(R - 1)
    jm_acc  = np.zeros((R, R))
    ln_acc  = np.zeros(R)
    lf_acc  = np.zeros(R)
    cnt = 0

    for _ in range(n_batches):
        toks, tgt, mask, _ = data.batch(bs, seq_len, device=device)
        B, T = toks.shape
        N = B * T
        fm = mask.reshape(-1)
        M = int(fm.sum())
        tgt_flat = tgt.reshape(-1)[fm]

        # Vorwaertslauf: Zustaende + Logits an jedem Iteration-Ende
        h0 = model.encode(toks)
        h = h0
        states = [h0]
        topk_cpu = []
        logits_all = []
        for _ in range(R):
            h_new, aux = bank(h, training=False)
            states.append(h_new)
            topk_cpu.append(aux["topk_idx"].cpu())   # (B, T, k)
            logits_all.append(model.readout(h_new))
            h = h_new

        # A: relative Zustandsaenderung pro Iteration
        for r in range(R):
            hp = states[r][mask].reshape(M, -1)
            hc = states[r + 1][mask].reshape(M, -1)
            sc_acc[r] += ((hc - hp).norm(dim=-1) /
                          hp.norm(dim=-1).clamp(min=1e-6)).mean().item()

        # B: vollstaendige Jaccard-Matrix
        fm_cpu = fm.cpu()
        topk_flat = [t.reshape(N, k) for t in topk_cpu]
        mems = []
        for r in range(R):
            mem = torch.zeros(M, nb, dtype=torch.bool)
            mem.scatter_(1, topk_flat[r][fm_cpu], True)
            mems.append(mem)
        for i in range(R):
            for j in range(R):
                inter = (mems[i] & mems[j]).sum(1).float()
                union = (mems[i] | mems[j]).sum(1).float()
                jm_acc[i, j] += (inter / union.clamp(min=1)).mean().item()

        # D: KL-Divergenz zwischen aufeinanderfolgenden Ausgaben
        for r in range(R - 1):
            p   = logits_all[r].reshape(N, -1)[fm].softmax(-1)
            q_n = logits_all[r + 1].reshape(N, -1)[fm].softmax(-1)
            kl_acc[r] += (q_n * (q_n.log() - p.log())).sum(-1).clamp(min=0).mean().item()

        # C: erzwungene Diversitaet (jede Iteration meidet Bloecke der vorherigen)
        if run_forced:
            h_f = h0.clone()
            prev_ti = None
            for r in range(R):
                if prev_ti is not None:
                    abl = torch.zeros(nb, dtype=torch.bool, device=device)
                    abl[prev_ti.reshape(-1).unique()] = True
                    h_f_new, aux_f = bank(h_f, training=False, ablate_mask=abl)
                else:
                    h_f_new, aux_f = bank(h_f, training=False)
                prev_ti = aux_f["topk_idx"].reshape(-1)
                h_f = h_f_new
                lg_n = logits_all[r].reshape(N, -1)[fm]
                lg_f = model.readout(h_f).reshape(N, -1)[fm]
                ln_acc[r] += F.cross_entropy(lg_n, tgt_flat).item()
                lf_acc[r] += F.cross_entropy(lg_f, tgt_flat).item()
        else:
            ln_acc += np.array([F.cross_entropy(
                logits_all[r].reshape(N, -1)[fm], tgt_flat).item()
                for r in range(R)])
            lf_acc[:] = float('nan')
        cnt += 1

    return {
        "state_rel_change":       (sc_acc  / cnt).tolist(),
        "kl_div_consecutive":     (kl_acc  / cnt).tolist(),
        "jaccard_full_matrix":    (jm_acc  / cnt).tolist(),
        "loss_normal":            (ln_acc  / cnt).tolist(),
        "loss_forced":            (lf_acc  / cnt).tolist(),
        "forced_minus_normal":    ((lf_acc - ln_acc) / cnt).tolist(),
    }


# ---------- main run ----------

def run(steps=10000, n_blocks=64, k=4, R=6, device="cuda",
        seed=0, ts_vocab=8000, ts_max_docs=20000,
        bs=32, seq_len=128, smoke=False,
        div_w=0.0, coord_w=0.0, diverse=False,
        diverse_until: int = 0, diverse_from_iter: int = 0,
        pretrained_ckpt: str | None = None,
        exp_name: str | None = None,
        check_finite: bool = True, strict_finite: bool = False,
        core_mode: str | None = None):

    data, dcfg = load_data(ts_vocab, ts_max_docs)
    model, cfg = make_model(data.vocab_size, n_blocks=n_blocks, k=k, R=R,
                            device=device, core_mode=core_mode)

    # Warm Start: Gewichte aus bestehendem Checkpoint laden
    if pretrained_ckpt:
        import rblm.checkpoint as _ckpt_mod
        _ckpt_mod.load(pretrained_ckpt, model, device=device, verify=True)
        print(f"[Exp3] Warm Start von: {pretrained_ckpt}", flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    phase3 = div_w > 0 or coord_w > 0 or diverse
    curriculum = diverse_until > 0 or diverse_from_iter > 0
    # Experiment-Name fuer den versionierten Checkpoint.
    if exp_name:
        _exp_name = exp_name
    elif curriculum:
        if diverse_until > 0:
            _exp_name = f"tinystories_curriculum_{int(round(100*diverse_until/(200 if smoke else steps)))}pct"
        else:
            _exp_name = f"tinystories_curriculum_fromIter{diverse_from_iter}"
    elif phase3:
        _exp_name = "tinystories_phase3"
    else:
        _exp_name = "tinystories_phase2"
    phase_str = ""
    if phase3:
        phase_str = (f"  [Phase3: div_w={div_w} coord_w={coord_w}"
                     f"{' diverse_train=ON' if diverse else ''}]")
    if curriculum:
        phase_str += (f"  [Curriculum: until={diverse_until}"
                      f" from_iter={diverse_from_iter}]")
    if pretrained_ckpt:
        phase_str += "  [WarmStart]"
    print(f"[Exp3] Modell C: {n_params/1e6:.1f}M Parameter, "
          f"n_blocks={n_blocks} k={k} R={R} vocab={data.vocab_size}{phase_str}", flush=True)

    actual_steps = 200 if smoke else steps
    eval_every = 50 if smoke else 500
    print(f"[Exp3] Training: {actual_steps} Schritte, bs={bs}, seq={seq_len}, device={device}",
          flush=True)
    t_train = time.time()
    history = train(model, data, steps=actual_steps, bs=bs, seq_len=seq_len,
                    device=device, eval_every=eval_every, seed=seed,
                    div_w=div_w, coord_w=coord_w, diverse=diverse,
                    diverse_until=diverse_until, diverse_from_iter=diverse_from_iter,
                    check_finite=check_finite, strict_finite=strict_finite)
    train_s = time.time() - t_train

    # Tag (frueh berechnet, damit die Sofort-Sicherung ihn nutzen kann)
    phase_tag = ("_div{}_crd{}{}"
                 .format(div_w, coord_w, "_diverse" if diverse else "")
                 if phase3 else "")
    curriculum_tag = ""
    if diverse_until > 0:
        pct = int(round(100 * diverse_until / actual_steps))
        curriculum_tag = f"_curr{pct}pct"
    elif diverse_from_iter > 0:
        curriculum_tag = f"_currFromIter{diverse_from_iter}"
    warm_tag = "_warm" if pretrained_ckpt else ""
    core_tag = f"_{core_mode}" if core_mode else ""
    tag = f"tinystories_b{n_blocks}k{k}R{R}_s{seed}{phase_tag}{curriculum_tag}{warm_tag}{core_tag}"

    # === SOFORT-SICHERUNG der Gewichte VOR der teuren Auswertungsphase ===
    # full_eval/cache_sim/iteration_diagnostics koennen haengen oder crashen
    # (Seed-3-Fall: trainiertes Modell ging verloren, weil der Checkpoint erst
    # NACH der Auswertung geschrieben wurde). Wir sichern Gewichte + einen
    # versionierten Checkpoint mit History-Metriken sofort. routing_stats sind
    # spaeter via scripts/eval_suite.py nachruestbar.
    os.makedirs(RESULTS, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(RESULTS, f"{tag}_model.pt"))
    print(f"[Exp3] Gewichte gesichert VOR Auswertung: results/{tag}_model.pt", flush=True)
    _save_versioned_checkpoint(
        model, cfg, _exp_name, seed, actual_steps, history,
        n_blocks=n_blocks, k=k, R=R, bs=bs, seq_len=seq_len,
        div_w=div_w, coord_w=coord_w, diverse=diverse,
        diverse_until=diverse_until, diverse_from_iter=diverse_from_iter,
        pretrained_ckpt=pretrained_ckpt, ts_vocab=ts_vocab, ts_max_docs=ts_max_docs,
        vocab_size=data.vocab_size, train_s=train_s)

    # === Auswertungsphase (best-effort) ===
    # Gewichte + versionierter Checkpoint sind oben bereits gesichert. Ein Haenger
    # oder Crash hier (full_eval/cache_sim/iteration_diagnostics) verliert daher
    # NICHT mehr das trainierte Modell — er ueberspringt nur die Analyse-Artefakte,
    # die via scripts/eval_suite.py nachgeruestet werden koennen.
    chance = math.log(data.vocab_size)
    try:
        print("[Exp3] Auswertung ...", flush=True)
        n_eval_batches = 4 if smoke else 12
        routing, unique_stats, traces_arr = full_eval(
            model, data, n_blocks, n_batches=n_eval_batches,
            bs=bs, seq_len=seq_len, device=device)

        usage = np.array(routing["usage_per_block"])
        hub_gini = gini(usage)
        top5_share = float(np.sort(usage)[-5:].sum() / max(usage.sum(), 1))

        caps = [max(k, n_blocks // 8), n_blocks // 4, n_blocks // 2]
        rng = np.random.default_rng(42)
        cache = cache_sim(traces_arr, n_blocks, caps, rng)

        final_jacc = routing["jaccard_consecutive"]

        print("[Exp3] Per-Iteration-Ablation ...", flush=True)
        abl_batches = 4 if smoke else 8
        abl = iteration_diagnostics(model, data, n_batches=abl_batches,
                                    bs=bs, seq_len=seq_len, device=device)

        # Effizienzmetrik: Qualitaetsgewinn pro zusaetzlich geladenem Block
        lpi_final = history[-1]["loss_per_iter"]
        extra_blocks = max(unique_stats["mean"] - k, 0.001)
        efficiency_gain = round((lpi_final[0] - lpi_final[-1]) / extra_blocks, 5)

        out = {
            "experiment": "Exp3_TinyStories",
            "config": {"steps": actual_steps, "n_blocks": n_blocks, "k": k, "R": R,
                       "div_w": div_w, "coord_w": coord_w, "diverse": diverse,
                       "diverse_until": diverse_until, "diverse_from_iter": diverse_from_iter,
                       "pretrained_ckpt": pretrained_ckpt,
                       "bs": bs, "seq_len": seq_len, "vocab": data.vocab_size,
                       "ts_max_docs": ts_max_docs, "device": device},
            "train_s": round(train_s, 1),
            "history": history,
            "chance_loss": round(chance, 3),
            "final_loss_per_iter": lpi_final,
            "jaccard_consecutive": final_jacc,
            "dead_blocks": routing["dead_blocks"],
            "router_entropy_norm": routing["router_entropy_norm"],
            "hub_gini": round(hub_gini, 4),
            "top5_block_share": round(top5_share, 4),
            "unique_blocks_per_token": unique_stats,
            "cache_sim": cache,
            "ablation": abl,
            "efficiency_gain": efficiency_gain,
            "efficiency_gain_note": "(L1-Lfin) / (unique_blocks_mean - k)",
        }
        with open(os.path.join(RESULTS, f"{tag}.json"), "w") as f:
            json.dump(out, f, indent=2)

        # routing_stats nachtraeglich in den (bereits gesicherten) Checkpoint legen
        _augment_checkpoint_routing_stats(
            _exp_name, seed, actual_steps,
            {
                "jaccard_consecutive":  final_jacc,
                "dead_blocks":          routing["dead_blocks"],
                "router_entropy_norm":  routing["router_entropy_norm"],
                "hub_gini":             round(hub_gini, 4),
                "top5_block_share":     round(top5_share, 4),
                "unique_blocks_per_token": unique_stats,
                "cache_sim":            cache,
                "ablation_state_rel_change":    abl["state_rel_change"],
                "ablation_forced_minus_normal": abl["forced_minus_normal"],
            })

        _figure(out, tag)
        _figure_ablation(out, tag)
        _print(out)
        _print_ablation(out["ablation"])
        return out
    except Exception as _e:
        import traceback
        print(f"[Exp3] WARNUNG: Auswertungsphase fehlgeschlagen ({_e}). "
              f"Trainiertes Modell ist gesichert (results/{tag}_model.pt + Checkpoint). "
              f"Analyse via scripts/eval_suite.py nachholbar.", flush=True)
        traceback.print_exc()
        return {
            "experiment": "Exp3_TinyStories",
            "config": {"steps": actual_steps, "n_blocks": n_blocks, "k": k, "R": R,
                       "seed": seed, "bs": bs, "seq_len": seq_len,
                       "vocab": data.vocab_size, "device": device},
            "train_s": round(train_s, 1),
            "history": history,
            "chance_loss": round(chance, 3),
            "eval_failed": str(_e),
        }


def _save_versioned_checkpoint(model, cfg, exp_name, seed, steps, history, *,
                               n_blocks, k, R, bs, seq_len, div_w, coord_w,
                               diverse, diverse_until, diverse_from_iter,
                               pretrained_ckpt, ts_vocab, ts_max_docs,
                               vocab_size, train_s):
    """Versionierten Checkpoint sofort nach dem Training speichern (mit
    History-Metriken). routing_stats werden — falls die Auswertung durchlaeuft —
    spaeter via _augment_checkpoint_routing_stats ergaenzt."""
    try:
        import rblm.checkpoint as _ckpt
        lpi_final = history[-1]["loss_per_iter"]
        _ckpt.save(
            model       = model,
            experiment  = exp_name,
            config      = {
                "model": {
                    "vocab_size":       vocab_size,
                    "d_model":          cfg.d_model,
                    "block_hidden":     cfg.block_hidden,
                    "n_heads":          cfg.n_heads,
                    "context_layers":   cfg.context_layers,
                    "max_len":          cfg.max_len,
                    "n_blocks":         n_blocks,
                    "k_active":         k,
                    "routed_iters":     R,
                    "key_dim":          cfg.key_dim,
                    "router_noise_std": cfg.router_noise_std,
                    "coord_dim":        getattr(cfg, "coord_dim", 3),
                    "core_mode":        getattr(cfg, "core_mode", None),
                },
                "training": {
                    "steps":             steps,
                    "bs":                bs,
                    "seq_len":           seq_len,
                    "lr":                0.002,
                    "weight_decay":      0.01,
                    "warmup":            200,
                    "div_w":             div_w,
                    "coord_w":           coord_w,
                    "diverse":           diverse,
                    "diverse_until":     diverse_until,
                    "diverse_from_iter": diverse_from_iter,
                    "pretrained_ckpt":   pretrained_ckpt,
                    "loss_weighting":    "end",
                    "lb_loss_weight":    0.01,
                },
                "data": {
                    "dataset":    "tinystories",
                    "max_docs":   ts_max_docs,
                    "vocab_size": ts_vocab,
                    "seq_len":    seq_len,
                },
            },
            metrics     = {
                "final_loss":    history[-1]["Lfin"],
                "L1":            history[-1]["L1"],
                "Lfin":          history[-1]["Lfin"],
                "loss_per_iter": lpi_final,
                "anytime_delta": round(max(lpi_final) - min(lpi_final), 4),
                "training_steps": steps,
                "train_s":        round(train_s, 1),
            },
            routing_stats = None,
            seed      = seed,
            step      = steps,
            val_loss  = history[-1]["Lfin"],
        )
    except Exception as _e:
        print(f"[CKPT] Warnung: Checkpoint-Speicherung fehlgeschlagen: {_e}", flush=True)


def _augment_checkpoint_routing_stats(exp_name, seed, step, routing_stats):
    """Schreibt routing_stats.json in ein bereits gespeichertes Checkpoint-Verzeichnis."""
    try:
        import rblm.checkpoint as _ckpt
        ckpt_dir = os.path.join(_ckpt.CKPT_ROOT, exp_name,
                                f"seed_{seed}", f"step_{step}")
        if os.path.isdir(ckpt_dir):
            with open(os.path.join(ckpt_dir, "routing_stats.json"),
                      "w", encoding="utf-8") as f:
                json.dump(routing_stats, f, indent=2, ensure_ascii=False)
    except Exception as _e:
        print(f"[CKPT] Warnung: routing_stats nicht ergaenzt: {_e}", flush=True)


# ---------- figures ----------

def _figure(out, tag):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    # 1) Training curves
    ax = axes[0, 0]
    steps_h = [h["step"] for h in out["history"]]
    ax.plot(steps_h, [h["L1"] for h in out["history"]], label="L(iter 1)", color="#2563eb")
    ax.plot(steps_h, [h["Lfin"] for h in out["history"]], label="L(iter final)", color="#b91c1c")
    ax.axhline(out["chance_loss"], color="gray", ls=":", label=f"Zufall ln({out['config']['vocab']})")
    ax.set_xlabel("Trainingsschritte"); ax.set_ylabel("Verlust")
    ax.set_title("F1: Lernt das Modell Sprache?"); ax.legend(fontsize=8); ax.grid(alpha=.3)

    # 2) Per-iteration loss at final eval
    ax = axes[0, 1]
    lpi = out["final_loss_per_iter"]
    R = len(lpi)
    ax.plot(range(1, R + 1), lpi, "o-", color="#2563eb")
    ax.axhline(out["chance_loss"], color="gray", ls=":")
    ax.set_xlabel("Modell-Iteration r"); ax.set_ylabel("Verlust")
    ax.set_title("Anytime-Qualitaetskurve (finales Eval)"); ax.grid(alpha=.3)

    # 3) Block usage (hub structure)
    ax = axes[1, 0]
    # We reconstruct usage from the stored ratio (approximate; use top-n from unique stats)
    # Actually we don't have usage_per_block in out; store Gini and note
    g = out["hub_gini"]
    t5 = out["top5_block_share"]
    n_blocks = out["config"]["n_blocks"]
    ax.bar(["Top-5 Bloecke", f"Rest ({n_blocks-5})"],
           [t5, 1 - t5], color=["#dc2626", "#93c5fd"])
    ax.set_ylabel("Anteil aller Aktivierungen")
    ax.set_title(f"F3: Hub-Struktur (Gini={g:.3f})\nTop-5 tragen {t5*100:.0f}% aller Aktivierungen")
    ax.grid(alpha=.3)

    # 4) Cache miss-rate
    ax = axes[1, 1]
    caps = out["cache_sim"]["caps"]
    ax.plot(caps, out["cache_sim"]["learned"], "o-", color="#2563eb", label="gelernt")
    ax.plot(caps, out["cache_sim"]["random"], "o--", color="#fca5a5", label="zufaellig")
    ax.set_xlabel(f"Cache-Kapazitaet (Bloecke von {n_blocks})")
    ax.set_ylabel("Miss-Rate")
    ax.set_title("F5: Gelernt vs. zufaelliges Routing\nunter Cache-Budget")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    jacc_str = ", ".join(f"{j:.2f}" for j in out["jaccard_consecutive"])
    ub = out["unique_blocks_per_token"]
    fig.suptitle(
        f"Exp3 TinyStories: n_blocks={n_blocks}, k={out['config']['k']}, R={out['config']['R']}, "
        f"{out['config']['steps']} Schritte\n"
        f"Jaccard(r→r+1): [{jacc_str}]   "
        f"Einzigartige Bloecke/Token: {ub['mean']:.1f} (→{ub['reduction_64']}× vs. n_b=64, "
        f"{ub['reduction_1000']}× vs. n_b=1000)",
        fontsize=9
    )
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, f"fig_exp3_{tag}.png"), dpi=130)
    plt.close(fig)


# ---------- ablation figures ----------

def _figure_ablation(out, tag):
    abl = out["ablation"]
    R = out["config"]["R"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # 1) Relative Zustandsaenderung
    ax = axes[0]
    sc = abl["state_rel_change"]
    ax.bar(range(1, R + 1), sc, color="#2563eb")
    ax.set_xlabel("Iteration r")
    ax.set_ylabel("||h_r - h_{r-1}|| / ||h_{r-1}||")
    ax.set_title("A) Relative Zustandsaenderung\npro Iteration")
    ax.grid(alpha=.3)

    # 2) Jaccard-Matrix Heatmap
    ax = axes[1]
    jm = np.array(abl["jaccard_full_matrix"])
    im = ax.imshow(jm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(R)); ax.set_xticklabels([f"r{i+1}" for i in range(R)])
    ax.set_yticks(range(R)); ax.set_yticklabels([f"r{i+1}" for i in range(R)])
    ax.set_title("B) Jaccard-Matrix\n(alle Iterationspaare)")
    for i in range(R):
        for j in range(R):
            ax.text(j, i, f"{jm[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if jm[i,j] > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # 3) Normal- vs. Forced-Diversity-Loss
    ax = axes[2]
    xs = range(1, R + 1)
    ax.plot(xs, abl["loss_normal"], "o-", color="#2563eb", label="normal")
    ax.plot(xs, abl["loss_forced"], "o--", color="#dc2626", label="forced diversity")
    ax.set_xlabel("Iteration r (kumulativ)")
    ax.set_ylabel("Verlust (CE)")
    ax.set_title("C) Erzwungene Diversitaet\n(jede Iter. meidet Vorgaenger-Bloecke)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    fig.suptitle(
        f"Per-Iteration-Ablation — {out['config']['steps']} Schritte, "
        f"n_blocks={out['config']['n_blocks']} k={out['config']['k']} R=R",
        fontsize=9
    )
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, f"fig_exp3_ablation_{tag}.png"), dpi=130)
    plt.close(fig)


# ---------- print ----------

def _print(out):
    cfg = out["config"]
    ub = out["unique_blocks_per_token"]
    chance = out["chance_loss"]
    print("\n" + "=" * 68)
    print(f"EXP3 TINYSTORIES  n_blocks={cfg['n_blocks']} k={cfg['k']} R={cfg['R']}")
    print(f"  vocab={cfg['vocab']}  steps={cfg['steps']}  {out['train_s']:.0f}s")
    print("=" * 68)
    lpi = out["final_loss_per_iter"]
    print(f"Verlust pro Iteration:  {[round(x,3) for x in lpi]}")
    print(f"Zufall:                 {chance:.3f}  -> unter Zufall: {lpi[-1] < chance}")
    jacc = out["jaccard_consecutive"]
    print(f"Jaccard (r->r+1):       {[round(j,3) for j in jacc]}")
    print(f"Hub-Gini:               {out['hub_gini']:.4f}  (0=flach, 1=ein Block alles)")
    print(f"Top-5-Block-Anteil:     {out['top5_block_share']*100:.1f}%")
    print(f"Tote Bloecke:           {out['dead_blocks']} / {cfg['n_blocks']}")
    print(f"\nEinzigartige Bloecke/Token: mean={ub['mean']:.1f}  p50={ub['p50']:.0f}  "
          f"p90={ub['p90']:.0f}  (max={ub['max_possible']})")
    print(f"Transfer-Reduktion vs. Layer-Offloading:")
    print(f"  n_blocks= 64: {ub['reduction_64']}x weniger Transfer")
    print(f"  n_blocks=256: {ub['reduction_256']}x weniger Transfer")
    print(f"  n_blocks=1000: {ub['reduction_1000']}x weniger Transfer")
    print(f"\nCache-Miss-Rate:")
    for cap, ml, mr in zip(out["cache_sim"]["caps"],
                            out["cache_sim"]["learned"], out["cache_sim"]["random"]):
        ratio = mr / max(ml, 0.001)
        print(f"  cap={cap:3d}: gelernt={ml:.3f}  zufaellig={mr:.3f}  ({ratio:.1f}x)")
    eg = out.get("efficiency_gain")
    if eg is not None:
        lpi = out["final_loss_per_iter"]
        ub_mean = out["unique_blocks_per_token"]["mean"]
        print(f"\nEffizienzmetrik (L1-Lfin) / (unique_blocks - k):")
        print(f"  L1={lpi[0]:.3f}  Lfin={lpi[-1]:.3f}  "
              f"unique_blocks={ub_mean:.1f}  k={cfg['k']}")
        print(f"  Effizienzgewinn = {eg:.5f} Nats/Block")


def _print_ablation(abl):
    R = len(abl["state_rel_change"])
    print("\n--- Per-Iteration-Ablation ---")
    print("  A) Relative Zustandsaenderung ||h_r - h_{r-1}|| / ||h_{r-1}||:")
    vals = "  ".join(f"r{r+1}:{v:.4f}" for r, v in enumerate(abl["state_rel_change"]))
    print(f"    {vals}")
    print("  D) KL-Divergenz p(r) -> p(r+1) [Output-Aenderung]:")
    vals = "  ".join(f"r{r+1}->{r+2}:{v:.4f}" for r, v in enumerate(abl["kl_div_consecutive"]))
    print(f"    {vals}")
    print("  B) Jaccard-Matrix aller Iterationspaare:")
    jm = abl["jaccard_full_matrix"]
    header = "       " + "  ".join(f"r{j+1}" for j in range(R))
    print(f"  {header}")
    for i, row in enumerate(jm):
        cells = "  ".join(f"{v:.2f}" for v in row)
        print(f"    r{i+1}:  {cells}")
    print("  C) Erzwungene Diversitaet (Loss-Delta = Forced - Normal, kumulativ):")
    for r in range(R):
        n = abl["loss_normal"][r]
        ff = abl["loss_forced"][r]
        d = abl["forced_minus_normal"][r]
        print(f"    r={r+1}: normal={n:.3f}  forced={ff:.3f}  delta={d:+.3f}")


# ---------- entry point ----------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--n_blocks", type=int, default=64)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--R", type=int, default=6)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--ts_vocab", type=int, default=8000)
    ap.add_argument("--ts_max_docs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true", help="Schnelltest (200 Schritte)")
    ap.add_argument("--div_w",   type=float, default=0.0,
                    help="Phase-3: Soft-Jaccard-Diversitaetsloss-Gewicht (0=aus)")
    ap.add_argument("--coord_w", type=float, default=0.0,
                    help="Phase-3: Koordinaten-Repulsionsloss-Gewicht (0=aus)")
    ap.add_argument("--diverse", action="store_true",
                    help="Phase-3: Hartes Diversity-Training (Bloecke aus r-1 in r verboten)")
    ap.add_argument("--diverse_until", type=int, default=0,
                    help="Curriculum: Schritte mit aktiver Diversity; danach normal (0=immer an)")
    ap.add_argument("--diverse_from_iter", type=int, default=0,
                    help="Curriculum C: Diversity erst ab dieser 0-basierten Iteration (0=alle)")
    ap.add_argument("--pretrained_ckpt", type=str, default=None,
                    help="Warm Start: Checkpoint-Verzeichnis oder .pt-Datei laden vor Training")
    ap.add_argument("--exp_name", type=str, default=None,
                    help="Checkpoint-Experiment-Name (ueberschreibt automatische Ableitung)")
    ap.add_argument("--check_finite", dest="check_finite", action="store_true", default=True,
                    help="NaN/Inf-Pruefung pro Schritt; schlechte Steps werden uebersprungen (Standard AN)")
    ap.add_argument("--no_check_finite", dest="check_finite", action="store_false",
                    help="NaN/Inf-Pruefung abschalten")
    ap.add_argument("--strict_finite", action="store_true",
                    help="Bei nicht-finitem Loss/Gradient sofort abbrechen statt ueberspringen (harte Diagnose)")
    ap.add_argument("--core_mode", type=str, default=None,
                    choices=[None, "per_token", "core_satellite"],
                    help="SR-Fixed-Core: per_token (Kern-Reuse ab r2) oder core_satellite")
    a = ap.parse_args()
    run(steps=a.steps, n_blocks=a.n_blocks, k=a.k, R=a.R,
        device=a.device, seed=a.seed, ts_vocab=a.ts_vocab,
        ts_max_docs=a.ts_max_docs, bs=a.bs, seq_len=a.seq_len, smoke=a.smoke,
        div_w=a.div_w, coord_w=a.coord_w, diverse=a.diverse,
        diverse_until=a.diverse_until, diverse_from_iter=a.diverse_from_iter,
        pretrained_ckpt=a.pretrained_ckpt, exp_name=a.exp_name,
        check_finite=a.check_finite, strict_finite=a.strict_finite,
        core_mode=a.core_mode)
