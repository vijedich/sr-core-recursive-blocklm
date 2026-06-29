"""Benchmark: gesehene vs. unbekannte Prompts (Memorisierung vs. Generalisierung).

Faehrt jedes trainierte Modell auf zwei Sets an und prueft die Vorhersage:
  SEEN    : Sequenzen aus dem Trainingskorpus (data/heteromini_v1) — exakt mittrainiert.
  UNKNOWN : frische, nie gesehene Dokumente derselben Domaenen (Held-out, gleicher Tokenizer).

Metriken je Set (und pro Domaene): mittlerer Loss (CE), Perplexitaet, Top-1-Next-Token-
Accuracy, Loss pro Iteration (Rekursionsverhalten). Der Abstand UNKNOWN-SEEN ist die
Generalisierungsluecke (hoch = Memorisierung, klein = generalisiert).

Wiederverwendbar: --glob/--checkpoints, Architektur aus dem Checkpoint (rblm.model_io).

Nutzung:
  python -m experiments.seen_vs_unknown --glob "results/hm_cont_*.pt"
"""
from __future__ import annotations
import argparse, json, math, os
import numpy as np
import torch
import torch.nn.functional as F

from rblm.heteromini import HeteroMiniData, build_heldout, DATA_ROOT
from rblm import model_io
from experiments.tinystories_exp import RESULTS

HELDOUT_DIR = DATA_ROOT + "_heldout"


@torch.no_grad()
def evaluate(model, data, n_batches=40, bs=16, seq_len=128, device="cpu"):
    model.eval()
    R = model.n_iters
    nd = data.n_domains
    ce_iter = torch.zeros(R, device=device)      # Loss pro Iteration (gesamt)
    tot_tok = 0
    correct = 0
    dom_ce = np.zeros(nd); dom_cor = np.zeros(nd); dom_tok = np.zeros(nd)
    for _ in range(n_batches):
        toks, tgt, mask, dom = data.batch(bs, seq_len, device=device, mode="contiguous")
        logits, _ = model(toks)
        V = logits[-1].shape[-1]
        fm = mask.reshape(-1)
        tgt_f = tgt.reshape(-1)
        m = int(fm.sum()); tot_tok += m
        for r in range(R):
            ce_iter[r] += F.cross_entropy(logits[r].reshape(-1, V)[fm], tgt_f[fm],
                                          reduction="sum")
        pred = logits[-1].reshape(-1, V).argmax(-1)
        correct += int((pred[fm] == tgt_f[fm]).sum())
        # pro Domaene (Domaene gilt je Sequenz)
        ce_tok = F.cross_entropy(logits[-1].reshape(-1, V), tgt_f, reduction="none").reshape(bs, seq_len)
        cor_tok = (logits[-1].argmax(-1) == tgt).float()
        doms = dom.cpu().numpy()
        ce_b = ce_tok.sum(1).cpu().numpy(); cor_b = cor_tok.sum(1).cpu().numpy()
        for b in range(bs):
            d = int(doms[b])
            if d < 0:
                continue
            dom_ce[d] += ce_b[b]; dom_cor[d] += cor_b[b]; dom_tok[d] += seq_len
    lpi = (ce_iter / max(1, tot_tok)).cpu().tolist()
    loss = lpi[-1]
    res = {"loss": round(loss, 4), "ppl": round(math.exp(loss), 2),
           "top1": round(correct / max(1, tot_tok), 4),
           "loss_per_iter": [round(x, 4) for x in lpi],
           "anytime": round(max(lpi) - min(lpi), 4), "n_tokens": tot_tok,
           "per_domain": {}}
    for d in range(nd):
        if dom_tok[d] > 0:
            dl = dom_ce[d] / dom_tok[d]
            res["per_domain"][data.domains[d]] = {
                "loss": round(float(dl), 4), "ppl": round(math.exp(dl), 2),
                "top1": round(float(dom_cor[d] / dom_tok[d]), 4)}
    return res


def run(checkpoints, n_batches=40, bs=16, seq_len=128, device="cpu"):
    seen = HeteroMiniData()
    if not os.path.exists(os.path.join(HELDOUT_DIR, "ids.npy")):
        print("[seen_vs_unknown] Held-out fehlt — baue es (frische Docs) ...", flush=True)
        build_heldout()
    unknown = HeteroMiniData(HELDOUT_DIR)
    print(f"[seen_vs_unknown] SEEN {seen.meta['n_tokens']:,} Tok / UNKNOWN "
          f"{unknown.meta['n_tokens']:,} Tok, Domaenen {seen.domains}\n", flush=True)

    out_rows = []
    print(f'{"model":22} {"set":8} {"loss":7} {"ppl":8} {"top1":7} {"Lr1":7} {"Lrfin":7} {"anyt":6}')
    print("-" * 76)
    for path in checkpoints:
        model, arch, step = model_io.load_checkpoint(path, seen.vocab_size, device)
        name = model_io.label(arch, step)
        rec = {"model": name}
        for setname, dat in (("seen", seen), ("unknown", unknown)):
            ev = evaluate(model, dat, n_batches, bs, seq_len, device)
            rec[setname] = ev
            print(f'{name:22} {setname:8} {ev["loss"]:<7.3f} {ev["ppl"]:<8.2f} '
                  f'{ev["top1"]:<7.3f} {ev["loss_per_iter"][0]:<7.3f} {ev["loss"]:<7.3f} '
                  f'{ev["anytime"]:<6.3f}')
        gap = round(rec["unknown"]["loss"] - rec["seen"]["loss"], 4)
        rec["gen_gap_loss"] = gap
        rec["ppl_ratio"] = round(rec["unknown"]["ppl"] / max(rec["seen"]["ppl"], 1e-9), 3)
        print(f'{name:22} {"GAP":8} dloss={gap:+.3f}  ppl x{rec["ppl_ratio"]}  '
              f'(unknown-seen; klein=generalisiert)\n')
        out_rows.append(rec)

    with open(os.path.join(RESULTS, "seen_vs_unknown.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": out_rows}, f, indent=2, ensure_ascii=False)
    # Per-Domaene-Generalisierungsluecke
    print("\n=== Generalisierungsluecke pro Domaene (UNKNOWN-SEEN Loss) ===")
    print(f'{"model":22} ' + " ".join(f"{d:>8}" for d in seen.domains))
    for rec in out_rows:
        deltas = []
        for d in seen.domains:
            s = rec["seen"]["per_domain"].get(d, {}).get("loss")
            u = rec["unknown"]["per_domain"].get(d, {}).get("loss")
            deltas.append(f"{u - s:+8.3f}" if (s is not None and u is not None) else f'{"-":>8}')
        print(f'{rec["model"]:22} ' + " ".join(deltas))
    print("Gespeichert: results/seen_vs_unknown.json")
    return out_rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/hm_cont_*.pt")
    ap.add_argument("--checkpoints", nargs="*", default=None)
    ap.add_argument("--n_batches", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    cks = a.checkpoints or model_io.discover(a.glob)
    if not cks:
        raise SystemExit(f"Keine Checkpoints gefunden (glob={a.glob!r}).")
    run(cks, n_batches=a.n_batches, bs=a.bs, seq_len=a.seq_len, device=a.device)
