"""Rekursionsgewinn-Analyse: gain = loss_r1 - loss_rR pro Beispiel, gruppiert nach Domaene.

Forschungsfrage (Viktor): Welche Aufgaben/Domaenen profitieren tatsaechlich von Rekursion?
Nutzt die Deep-Supervision-Readouts (Loss nach jeder Iteration). Pro contiguous-Eval-Sequenz
(Domaene eindeutig):
  ce_r[seq] = mittlere CE der Sequenz nach Iteration r
  gain[seq] = ce_r1 - ce_rR     (>0 = Rekursion hilft dieser Sequenz)

Aggregiert: mean_gain pro Domaene (+ std, n) und Top-/Bottom-Beispiele (mit Text-Snippet).

Laedt einen Continuation-Snapshot (results/hm_cont_{exp}_s0.pt) und baut das Modell daraus.

Nutzung:
  python -m experiments.gain_analysis --exp hm_srcore_b32_R6 --n_batches 40
"""
from __future__ import annotations
import argparse, json, os, re as _re

def _seed_from_path(p: str) -> int:
    m = _re.search(r"_s(\d+)\.pt$", str(p))
    return int(m.group(1)) if m else 0
import numpy as np
import torch
import torch.nn.functional as F

from rblm.heteromini import HeteroMiniData, DATA_ROOT
from rblm import model_io
from experiments.tinystories_exp import RESULTS

HELDOUT_DIR = DATA_ROOT + "_heldout"


def _tokenizer():
    try:
        from tokenizers import ByteLevelBPETokenizer
        return ByteLevelBPETokenizer(os.path.join(DATA_ROOT, "vocab.json"),
                                     os.path.join(DATA_ROOT, "merges.txt"))
    except Exception:
        return None


@torch.no_grad()
def gain_analysis(model, data, n_batches=40, bs=16, seq_len=128, device="cpu", top_k=6):
    model.eval()
    tok = _tokenizer()
    per_dom = {d: [] for d in range(data.n_domains)}
    examples = []   # (gain, dom_id, token_ids)
    for _ in range(n_batches):
        toks, tgt, mask, dom = data.batch(bs, seq_len, device=device, mode="contiguous")
        logits, _ = model(toks)
        B, T = toks.shape
        V = logits[0].shape[-1]
        # CE pro Token fuer erste und letzte Iteration
        ce_first = F.cross_entropy(logits[0].reshape(-1, V), tgt.reshape(-1),
                                   reduction="none").reshape(B, T)
        ce_last = F.cross_entropy(logits[-1].reshape(-1, V), tgt.reshape(-1),
                                  reduction="none").reshape(B, T)
        gain_tok = (ce_first - ce_last)                  # (B,T)
        seq_gain = gain_tok.mean(dim=1).cpu().numpy()    # (B,)
        doms = dom.cpu().numpy()
        toks_cpu = toks.cpu().numpy()
        for b in range(B):
            d = int(doms[b])
            if d < 0:
                continue
            per_dom[d].append(float(seq_gain[b]))
            examples.append((float(seq_gain[b]), d, toks_cpu[b]))

    dom_stats = {}
    for d, vals in per_dom.items():
        if vals:
            a = np.array(vals)
            dom_stats[data.domains[d]] = {"mean_gain": round(float(a.mean()), 4),
                                          "std": round(float(a.std()), 4), "n": len(vals)}
    examples.sort(key=lambda x: x[0], reverse=True)

    def snippet(ids):
        if tok is None:
            return None
        try:
            return tok.decode([int(i) for i in ids[:48]]).replace("\n", " ")[:120]
        except Exception:
            return None

    top = [{"gain": round(g, 4), "domain": data.domains[d], "snippet": snippet(t)}
           for g, d, t in examples[:top_k]]
    bottom = [{"gain": round(g, 4), "domain": data.domains[d], "snippet": snippet(t)}
              for g, d, t in examples[-top_k:]]
    return {"domain_gain": dom_stats, "top_examples": top, "bottom_examples": bottom,
            "n_examples": len(examples)}


def run(path, n_batches=40, bs=16, seq_len=128, device="cpu"):
    data = HeteroMiniData()
    model, arch, step = model_io.load_checkpoint(path, data.vocab_size, device)
    name = model_io.label(arch)
    res = gain_analysis(model, data, n_batches, bs, seq_len, device)
    res["experiment"] = name; res["step"] = step
    out = os.path.join(RESULTS, f"gain_{name}_s{_seed_from_path(path)}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n=== Rekursionsgewinn {name} @ step {step} (gain = loss_r1 - loss_rR) ===")
    print(f"{'Domaene':10} {'mean_gain':>10} {'std':>8} {'n':>5}")
    for dom, s in sorted(res["domain_gain"].items(), key=lambda kv: -kv[1]["mean_gain"]):
        print(f"{dom:10} {s['mean_gain']:>10.4f} {s['std']:>8.4f} {s['n']:>5}")
    print("\nTop-Beispiele (groesster Rekursionsgewinn):")
    for e in res["top_examples"]:
        print(f"  +{e['gain']:.3f} [{e['domain']:5}] {e['snippet']}")
    print(f"\nGespeichert: {out}")
    return res


def run_seen_unknown(path, n_batches=40, bs=16, seq_len=128, device="cpu"):
    """Vergleicht Rekursionsgewinn auf GESEHENEN vs. UNBEKANNTEN Dokumenten pro Domaene.

    code_gain_seen vs code_gain_unknown: unterscheidet ob Rekursion Struktur ERKENNT
    (unknown hoch) oder nur bekannte Muster FITTET (unknown tief).
    """
    data_seen = HeteroMiniData()
    if not os.path.isdir(HELDOUT_DIR):
        print(f"[GainSU] Kein Heldout-Verzeichnis: {HELDOUT_DIR} — uebersprungen.")
        return None
    data_unknown = HeteroMiniData(HELDOUT_DIR)
    model, arch, step = model_io.load_checkpoint(path, data_seen.vocab_size, device)
    name = model_io.label(arch, step)

    print(f"[GainSU] {name}: evaluiere Seen ...", flush=True)
    seen_res = gain_analysis(model, data_seen, n_batches, bs, seq_len, device)
    print(f"[GainSU] {name}: evaluiere Unknown ...", flush=True)
    unk_res = gain_analysis(model, data_unknown, n_batches, bs, seq_len, device)

    comparison = {}
    for dom in data_seen.domains:
        s = seen_res["domain_gain"].get(dom)
        u = unk_res["domain_gain"].get(dom)
        if s and u:
            comparison[dom] = {
                "gain_seen":    round(s["mean_gain"], 4),
                "gain_unknown": round(u["mean_gain"], 4),
                "delta":        round(u["mean_gain"] - s["mean_gain"], 4),
                "ratio_u_s":    round(u["mean_gain"] / max(0.0001, s["mean_gain"]), 3),
                "n_seen":       s["n"],
                "n_unknown":    u["n"],
            }

    result = {"experiment": name, "step": step,
              "seen": seen_res, "unknown": unk_res, "comparison": comparison}
    out = os.path.join(RESULTS, f"gain_seen_unknown_{name}_s{_seed_from_path(path)}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n=== Gain Seen vs. Unknown  {name} ===")
    print(f"{'Domaene':8} {'gain_seen':>10} {'gain_unk':>10} {'delta':>8} {'ratio':>7}")
    for dom, v in sorted(comparison.items(), key=lambda kv: -kv[1]["gain_seen"]):
        print(f"{dom:8} {v['gain_seen']:>10.4f} {v['gain_unknown']:>10.4f} "
              f"{v['delta']:>+8.4f} {v['ratio_u_s']:>7.3f}x")
    print(f"Gespeichert: {out}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/hm_cont_*.pt",
                    help="Glob fuer Checkpoints (Default: alle Continuation-Snapshots)")
    ap.add_argument("--checkpoints", nargs="*", default=None,
                    help="explizite Checkpoint-Pfade (ueberschreibt --glob)")
    ap.add_argument("--n_batches", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seen_unknown", action="store_true",
                    help="Vergleich gain_seen vs gain_unknown (braucht heteromini_v1_heldout)")
    a = ap.parse_args()
    cks = a.checkpoints or model_io.discover(a.glob)
    if not cks:
        raise SystemExit(f"Keine Checkpoints gefunden (glob={a.glob!r}).")
    fn = run_seen_unknown if a.seen_unknown else run
    for p in cks:
        fn(p, n_batches=a.n_batches, bs=a.bs, seq_len=a.seq_len, device=a.device)
