"""Qualitative Generation Suite: b32@15k vs b64@10k vs b64@15k.

Metriken pro Modell x Prompt x Sampling-Modus x R:
  - repetition_rate  (Wiederholungsschleifen)
  - distinct_1 / distinct_2
  - domain_token_hits (domänenspezifische Tokens)
  - avg_len_before_loop

Nutzung (nach b64@15k):
  python scripts/qual_gen.py --device cuda
  python scripts/qual_gen.py --device cuda --models b32 b64_10k b64_15k
"""
from __future__ import annotations
import argparse, json, os, re, sys
from collections import Counter
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rblm import model_io

RESULTS = os.path.join(ROOT, "results")

# Verfügbare Checkpoints
CKPT_REGISTRY = {
    "b32_15k_s0":      os.path.join(RESULTS, "BASELINE_srcore_b32_k8_R6_15k.pt"),
    "b32_15k_s1":      os.path.join(RESULTS, "hm_cont_hm_srcore_b32_k8_R6_s1.pt"),
    "b64_10k":         os.path.join(RESULTS, "hm_srcore_b64_k8_R6_s0_10k.pt"),
    "b64_15k":         os.path.join(RESULTS, "hm_cont_hm_srcore_b64_k8_R6_s0.pt"),
    "b64_ctrl_17k":    os.path.join(RESULTS, "hm_cont_hm_srcore_b64_k8_R6_ctrl_17k_s0.pt"),
    "b64_lam003_17k":  os.path.join(RESULTS, "hm_cont_hm_srcore_b64_k8_R6_entmin_r1_lam003_s0.pt"),
    "b64_lam005_17k":  os.path.join(RESULTS, "hm_cont_hm_srcore_b64_k8_R6_entmin_r1_lam005_s0.pt"),
    "b64_lam007_17k":  os.path.join(RESULTS, "hm_cont_hm_srcore_b64_k8_R6_entmin_lam007_s0.pt"),
}

# Domain-Token-Dictionaries (Mindest-Indikatoren)
DOMAIN_TOKENS = {
    "code": ["def", "return", "import", "class", "self", "for", "if",
             "range", "print", "int", "str", "True", "False", "None",
             "elif", "else", "while", "yield", "lambda", "raise"],
    "wiki": ["the", "is", "was", "known", "history", "century", "located",
             "refers", "born", "died", "founded", "population", "region",
             "called", "city", "country", "language", "period"],
    "lit":  ["the", "he", "she", "had", "said", "his", "her", "it",
             "was", "looked", "felt", "door", "room", "eyes", "hand",
             "old", "night", "away", "came", "stood"],
    "web":  ["the", "you", "how", "best", "way", "first", "make",
             "learn", "use", "great", "good", "here", "step", "tips",
             "guide", "easy", "need", "your", "simple"],
}

PROMPTS = [
    # Code
    {"domain": "code",  "text": "# hello world\n",                       "label": "code_helloworld"},
    {"domain": "code",  "text": "def fibonacci(",                         "label": "code_fib"},
    {"domain": "code",  "text": "import numpy as np\n",                   "label": "code_import"},
    {"domain": "code",  "text": "class User:\n    def __init__(self,",    "label": "code_class"},
    {"domain": "code",  "text": "try:\n    import",                       "label": "code_try"},
    # Wiki
    {"domain": "wiki",  "text": "The capital of France is",               "label": "wiki_capital"},
    {"domain": "wiki",  "text": "== History of",                          "label": "wiki_history"},
    {"domain": "wiki",  "text": "The Roman Empire was",                   "label": "wiki_roman"},
    {"domain": "wiki",  "text": "In mathematics, a function is",          "label": "wiki_math"},
    # Literatur
    {"domain": "lit",   "text": "Once upon a time",                       "label": "lit_once"},
    {"domain": "lit",   "text": "The old house stood",                    "label": "lit_house"},
    {"domain": "lit",   "text": "She opened the letter and",              "label": "lit_letter"},
    # Web / Instruction
    {"domain": "web",   "text": "How to make a cup of tea:",              "label": "web_tea"},
    {"domain": "web",   "text": "The best way to learn Python is",        "label": "web_python"},
    {"domain": "web",   "text": "Here are three reasons why",             "label": "web_reasons"},
]

SAMPLING_MODES = [
    {"label": "greedy",   "temperature": None, "top_k": None},
    {"label": "t07k40",   "temperature": 0.7,  "top_k": 40},
    {"label": "t09k80",   "temperature": 0.9,  "top_k": 80},
]

GEN_LEN      = 80
R_READOUTS   = [1, 3, 6]


# ─── Tokeniser ────────────────────────────────────────────────────────────────

_TOKENIZER = None

def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from tokenizers import ByteLevelBPETokenizer
        from rblm.heteromini import DATA_ROOT
        _TOKENIZER = ByteLevelBPETokenizer(
            os.path.join(DATA_ROOT, "vocab.json"),
            os.path.join(DATA_ROOT, "merges.txt"),
        )
    return _TOKENIZER


def encode(text: str) -> list[int]:
    return get_tokenizer().encode(text).ids


def decode(ids: list[int]) -> str:
    return get_tokenizer().decode(ids)


def decode_tokens(ids: list[int]) -> list[str]:
    tok = get_tokenizer()
    return [tok.id_to_token(i) or "" for i in ids]


# ─── Metriken ─────────────────────────────────────────────────────────────────

def repetition_rate(tokens: list[str], window: int = 20) -> float:
    if len(tokens) < 2: return 0.0
    repeats = sum(1 for i in range(window, len(tokens))
                  if tokens[i] in tokens[max(0, i - window):i])
    return round(repeats / max(1, len(tokens) - window), 3)


def distinct_n(tokens: list[str], n: int) -> float:
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams: return 0.0
    return round(len(set(ngrams)) / len(ngrams), 3)


def domain_hit_rate(tokens: list[str], domain: str) -> float:
    vocab = set(DOMAIN_TOKENS.get(domain, []))
    hits  = sum(1 for t in tokens
                if t.lower().strip("Ġ▁") in vocab or t.strip("Ġ▁") in vocab)
    return round(hits / max(1, len(tokens)), 3)


def avg_len_before_loop(tokens: list[str], window: int = 8) -> float:
    for i in range(window, len(tokens)):
        span = tokens[max(0, i - window):i]
        if tokens[i] in span:
            return float(i)
    return float(len(tokens))


# ─── Generierung ──────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, prompt_ids: list[int], gen_len: int,
             R_readouts: list[int], temperature=None, top_k=None,
             device="cuda") -> dict[int, list[int]]:
    """Autoregressive Generation; gibt Tokens pro R-Readout zurück."""
    arch = getattr(model, "cfg", None)
    R_max = model.n_iters

    ctx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated: dict[int, list[int]] = {r: [] for r in R_readouts}

    for _ in range(gen_len):
        logits_all, _ = model(ctx)
        for r in R_readouts:
            r_idx = min(r - 1, R_max - 1)
            lg    = logits_all[r_idx][0, -1, :].float()
            if temperature is not None and temperature > 0:
                lg = lg / temperature
            if top_k is not None and top_k > 0:
                topk_vals, _ = torch.topk(lg, min(top_k, lg.size(-1)))
                lg[lg < topk_vals[-1]] = float("-inf")
            if temperature is None:
                nxt = int(lg.argmax().item())
            else:
                probs = F.softmax(lg, dim=-1)
                nxt   = int(torch.multinomial(probs, 1).item())
            generated[r].append(nxt)
        # Weiterführen mit dem R=R_max-Token (bestes Readout)
        main_lg = logits_all[-1][0, -1, :].float()
        if temperature is not None and temperature > 0:
            main_lg = main_lg / temperature
        if top_k is not None and top_k > 0:
            topk_vals, _ = torch.topk(main_lg, min(top_k, main_lg.size(-1)))
            main_lg[main_lg < topk_vals[-1]] = float("-inf")
        if temperature is None:
            nxt_main = int(main_lg.argmax().item())
        else:
            nxt_main = int(torch.multinomial(F.softmax(main_lg, -1), 1).item())
        ctx = torch.cat([ctx, torch.tensor([[nxt_main]], device=device)], dim=1)

    return generated


# ─── Haupt-Loop ───────────────────────────────────────────────────────────────

def _load_any(ck_path: str, device: str):
    raw = torch.load(ck_path, map_location=device, weights_only=False)
    if "arch" in raw:
        return model_io.load_checkpoint(ck_path, device=device)
    cfg  = raw["config"]
    step = int(raw.get("step", 0))
    from experiments.tinystories_exp import make_model
    from rblm.heteromini import HeteroMiniData
    vocab = HeteroMiniData().vocab_size
    model, _ = make_model(vocab, n_blocks=cfg["n_blocks"], k=cfg["k"],
                          R=cfg["R"], device=device, core_mode=cfg["core_mode"])
    model.load_state_dict(raw["model"])
    arch = {"n_blocks": cfg["n_blocks"], "k": cfg["k"], "R": cfg["R"],
            "core_mode": cfg["core_mode"]}
    return model, arch, step


def run_model(model_key: str, ck_path: str, device: str) -> dict:
    print(f"\n{'='*60}", flush=True)
    print(f"  Modell: {model_key}  ({ck_path})", flush=True)

    if not os.path.exists(ck_path):
        print(f"  FEHLER: Checkpoint nicht gefunden: {ck_path}", flush=True)
        return {}

    model, arch, step = _load_any(ck_path, device=device)
    model.eval()
    label = f"{model_key}@{step}"
    print(f"  {label}", flush=True)

    results = {}
    for prompt in PROMPTS:
        p_label = prompt["label"]
        domain  = prompt["domain"]
        p_ids   = encode(prompt["text"])
        results[p_label] = {"domain": domain, "prompt": prompt["text"], "modes": {}}

        for mode in SAMPLING_MODES:
            m_label = mode["label"]
            temp    = mode["temperature"]
            topk    = mode["top_k"]
            print(f"  [{p_label}] {m_label} ...", flush=True, end="")

            gen = generate(model, p_ids, GEN_LEN, R_READOUTS,
                           temperature=temp, top_k=topk, device=device)

            mode_res = {}
            for r in R_READOUTS:
                tok_ids  = gen[r]
                tok_strs = decode_tokens(tok_ids)
                text_out = prompt["text"] + decode(tok_ids)
                mode_res[f"R{r}"] = {
                    "text":           text_out,
                    "repetition":     repetition_rate(tok_strs),
                    "distinct_1":     distinct_n(tok_strs, 1),
                    "distinct_2":     distinct_n(tok_strs, 2),
                    "domain_hits":    domain_hit_rate(tok_strs, domain),
                    "len_before_loop": avg_len_before_loop(tok_strs),
                }
            results[p_label]["modes"][m_label] = mode_res
            print(" OK", flush=True)

    return {"model": label, "arch": arch, "results": results}


# ─── Ausgabe / Vergleich ──────────────────────────────────────────────────────

def print_comparison(all_results: dict, model_keys: list[str]):
    metrics = ["repetition", "distinct_2", "domain_hits", "len_before_loop"]
    mode    = "greedy"
    R       = "R6"
    print("\n" + "="*70)
    print(f"  Vergleich: {' / '.join(model_keys)}  (mode={mode}, R={R})")
    print("="*70)
    for p in PROMPTS:
        p_label = p["label"]
        domain  = p["domain"]
        print(f"\n  [{domain}] {p['text']!r:.50s}")
        print(f"  {'Metrik':<20}" + "".join(f"  {k:<14}" for k in model_keys))
        print("  " + "-" * (20 + 16 * len(model_keys)))
        for m in metrics:
            row = f"  {m:<20}"
            for mk in model_keys:
                try:
                    v = all_results[mk]["results"][p_label]["modes"][mode][R][m]
                    row += f"  {v:<14}"
                except (KeyError, TypeError):
                    row += f"  {'N/A':<14}"
            print(row)
        # Greedy-Text Vergleich bei R1 vs R6
        print(f"\n  {'':4} {'':8}", end="")
        for mk in model_keys:
            print(f"  [{mk}]", end="")
        print()
        for ri in ["R1", "R6"]:
            print(f"  {ri}:", flush=True)
            for mk in model_keys:
                try:
                    txt = all_results[mk]["results"][p_label]["modes"][mode][ri]["text"]
                    short = txt.replace("\n", " \\n ")[:80]
                    print(f"    {mk:<12}: {short}")
                except (KeyError, TypeError):
                    print(f"    {mk:<12}: N/A")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", nargs="+",
                    default=["b32_15k_s0", "b64_10k", "b64_15k"],
                    choices=list(CKPT_REGISTRY.keys()))
    ap.add_argument("--out", default=os.path.join(RESULTS, "qual_gen_comparison.json"))
    a = ap.parse_args()

    all_results = {}
    for mk in a.models:
        ck = CKPT_REGISTRY.get(mk, "")
        res = run_model(mk, ck, a.device)
        if res:
            all_results[mk] = res

    print_comparison(all_results, a.models)

    def _serial(obj):
        if isinstance(obj, dict):   return {str(k): _serial(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [_serial(i) for i in obj]
        return obj

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(_serial(all_results), f, indent=2, ensure_ascii=False)
    print(f"\nGespeichert: {a.out}")


if __name__ == "__main__":
    main()
