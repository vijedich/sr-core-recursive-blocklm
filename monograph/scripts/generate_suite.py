"""Qualitative Generierungs-Suite — 4 Modelle x 7 Prompts.

Greedy + temperature 0.7/top_k 40.
Fuer srcore_b32_k8_R6: Ausgabe nach Iteration 1, 2, 4, 6 (logits[r-1]).

Nutzung:
  python scripts/generate_suite.py --device cuda
  python scripts/generate_suite.py --device cuda --max_new 80
"""
from __future__ import annotations
import argparse, json, os, sys, time
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rblm.heteromini import DATA_ROOT
from rblm import model_io

DATA_DIR = DATA_ROOT                        # data/heteromini_v1
RESULTS  = os.path.join(ROOT, "results")

PROMPTS = [
    ("code_hello",    "# hello world"),
    ("code_add",      "# function to add two numbers"),
    ("code_fib",      "def fibonacci"),
    ("code_import",   "import numpy as np"),
    ("wiki_capital",  "The capital of France is"),
    ("lit_story",     "Once upon a time"),
    ("wiki_history",  "== History of"),
]

MODELS = [
    ("dense_d24@10k",    "hm_cont_hm_dense_d24_s0.pt"),
    ("naked_b32_R6@10k", "hm_cont_hm_naked_b32_R6_s0.pt"),
    ("k4_R6@10k",        "hm_cont_hm_srcore_b32_R6_s0.pt"),
    ("k8_R6@15k",        "hm_cont_hm_srcore_b32_k8_R6_s0.pt"),
]

# Fuer k8_R6: zeige Ausgabe auch nach diesen Iterationen (0-basiert → r1=idx0, etc.)
ITER_SHOW = [0, 1, 3, 5]    # = r1, r2, r4, r6 (0-basierte Indizes)


def load_tokenizer():
    from tokenizers import ByteLevelBPETokenizer
    return ByteLevelBPETokenizer(
        os.path.join(DATA_DIR, "vocab.json"),
        os.path.join(DATA_DIR, "merges.txt"),
    )


def encode(tok, text):
    return tok.encode("<bos>" + text).ids


def decode(tok, ids):
    try:
        return tok.decode(ids, skip_special_tokens=False)
    except Exception:
        return "[decode error]"


@torch.no_grad()
def generate(model, input_ids, max_new, device, temperature=0.0, top_k=0,
             iter_idx=-1):
    """
    Autoregressiv generieren. iter_idx: welchen Iterations-Logit verwenden?
      -1 = letzter (Standard)
      0..R-1 = explizite Iteration

    Gibt (generated_ids, ms_per_token) zurueck.
    """
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    t0 = time.perf_counter()
    for _ in range(max_new):
        logits_all, _ = model(ids)
        logits = logits_all[iter_idx]          # (1, T, V)
        next_logit = logits[0, -1, :]          # (V,)
        if temperature <= 0 or top_k == 0:
            next_id = next_logit.argmax().item()
        else:
            next_logit = next_logit / temperature
            if top_k > 0:
                top_vals, _ = torch.topk(next_logit, top_k)
                next_logit[next_logit < top_vals[-1]] = float("-inf")
            probs = F.softmax(next_logit, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    ms_per_tok = elapsed_ms / max_new
    generated = ids[0, len(input_ids):].tolist()
    return generated, ms_per_tok


def run_model(model_name, ck_path, tok, prompts, max_new, device, is_k8=False):
    """Generiert fuer alle Prompts, gibt results-Dict zurueck."""
    model, arch, step = model_io.load_checkpoint(ck_path, 8000, device)
    model.eval()
    R = arch.get("R", 1)

    results = {"model": model_name, "arch": model_io.label(arch, step), "R": R, "prompts": {}}

    for pid, prompt_text in prompts:
        input_ids = encode(tok, prompt_text)
        entry = {"prompt": prompt_text, "input_ids": input_ids[:10]}

        # Greedy mit letzter Iteration
        gen_g, ms_g = generate(model, input_ids, max_new, device,
                                temperature=0.0, iter_idx=-1)
        entry["greedy"] = decode(tok, gen_g)
        entry["greedy_ms_per_tok"] = round(ms_g, 2)

        # Sampling temperature 0.7 / top_k 40
        torch.manual_seed(42)
        gen_s, ms_s = generate(model, input_ids, max_new, device,
                                temperature=0.7, top_k=40, iter_idx=-1)
        entry["sampled"] = decode(tok, gen_s)
        entry["sampled_ms_per_tok"] = round(ms_s, 2)

        # Per-Iteration nur fuer k8_R6 (und nur wenn Modell >= 4 Iterationen hat)
        if is_k8 and R >= 4:
            iters = {}
            for idx in ITER_SHOW:
                if idx < R:
                    gen_i, _ = generate(model, input_ids, max_new, device,
                                        temperature=0.0, iter_idx=idx)
                    iters[f"r{idx+1}"] = decode(tok, gen_i)
            entry["per_iter_greedy"] = iters

        results["prompts"][pid] = entry
        print(f"    [{pid}] greedy: {entry['greedy'][:60].replace(chr(10),' ')!r}")

    return results


def make_markdown(all_results, prompts):
    lines = ["# Qualitative Generierungs-Suite\n",
             f"*HeteroMini-Tokenizer (vocab=8000). max_new=varies. "
             f"Greedy + Temp 0.7/Top-K 40.*\n"]

    for pid, prompt_text in prompts:
        lines.append(f"\n---\n\n## Prompt: `{prompt_text}`\n")

        for res in all_results:
            mname = res["model"]
            p = res["prompts"].get(pid, {})
            if not p:
                continue

            lines.append(f"### {mname} ({res['arch']})\n")
            lines.append(f"**Greedy** ({p.get('greedy_ms_per_tok','?')} ms/tok):\n")
            lines.append(f"```\n{prompt_text}{p.get('greedy','')}\n```\n")
            lines.append(f"**Sampling** (T=0.7, k=40, {p.get('sampled_ms_per_tok','?')} ms/tok):\n")
            lines.append(f"```\n{prompt_text}{p.get('sampled','')}\n```\n")

            if "per_iter_greedy" in p:
                lines.append("**Per-Iteration (Greedy, srcore_k8_R6):**\n")
                for rname, text in sorted(p["per_iter_greedy"].items()):
                    lines.append(f"*{rname}:* `{prompt_text}{text[:80].replace(chr(10),' ')}`\n\n")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_new", type=int, default=100)
    a = ap.parse_args()

    print("Lade Tokenizer ...", flush=True)
    tok = load_tokenizer()

    all_results = []
    for mname, fname in MODELS:
        ck_path = os.path.join(RESULTS, fname)
        if not os.path.exists(ck_path):
            print(f"  [SKIP] {fname} nicht gefunden")
            continue
        print(f"\n=== {mname} ===", flush=True)
        is_k8 = "k8" in mname
        res = run_model(mname, ck_path, tok, PROMPTS, a.max_new, a.device, is_k8)
        all_results.append(res)

    # Speichern
    out_json = os.path.join(RESULTS, "generation_suite.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    md = make_markdown(all_results, PROMPTS)
    out_md = os.path.join(ROOT, "GENERATION_SUITE.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n=== FERTIG ===")
    print(f"  JSON:     {out_json}")
    print(f"  Markdown: {out_md}")


if __name__ == "__main__":
    main()
