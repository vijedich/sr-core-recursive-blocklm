"""HeteroMini-v1 — kleines heterogenes Pretraining-Testfeld (Streaming-Build).

Domaenen (HF, ungated, klein gestreamt):
  web  : HuggingFaceFW/fineweb-edu (sample-10BT)   field 'text'
  wiki : wikimedia/wikipedia (20231101.en)         field 'text'
  code : codeparrot/codeparrot-clean-valid         field 'content'
  lit  : sedthh/gutenberg_english                  field 'TEXT'

Build:
  - streamt pro Domaene kleine Samples (keine riesigen Downloads),
  - trainiert EINEN ByteLevelBPE-Tokenizer (vocab=8000, wie bisher) auf der Mischung,
  - tokenisiert jedes Dokument mit <bos>-Praefix, haelt Dokumente als zusammenhaengende
    Chunks (doc_start/doc_len/doc_domain),
  - speichert ids (uint16), Dokument-Index und Tokenizer nach data/heteromini_v1/.

HeteroMiniData liefert Batches in zwei Modi:
  contiguous : Fenster INNERHALB eines Dokuments (Lokalitaet erhalten, Domaene eindeutig)
  shuffled   : voellig zufaellige Token-Positionen (Lokalitaet zerstoert; domain=-1)
"""
from __future__ import annotations
import itertools, json, os
import numpy as np


DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "heteromini_v1")

# (domain, load_dataset-args, kwargs, text-field)
DOMAINS = [
    ("web",  ["HuggingFaceFW/fineweb-edu"],        {"name": "sample-10BT"}, "text"),
    ("wiki", ["wikimedia/wikipedia"],              {"name": "20231101.en"}, "text"),
    ("code", ["codeparrot/codeparrot-clean-valid"], {},                     "content"),
    ("lit",  ["sedthh/gutenberg_english"],         {},                      "TEXT"),
]


def _collect_texts(spec, max_docs, max_chars, skip=0):
    name, args, kw, field = spec
    from datasets import load_dataset
    ds = load_dataset(*args, split="train", streaming=True, **kw)
    it = iter(ds)
    if skip:                       # erste `skip` Rohzeilen ueberspringen (Held-out disjunkt)
        next(itertools.islice(it, skip, skip), None)
    out = []
    for row in itertools.islice(it, max_docs * 3):  # Reserve fuer Kurz-Doku-Filter
        t = row.get(field) or ""
        if len(t) < 200:           # zu kurze Dokumente ueberspringen
            continue
        out.append(t[:max_chars])  # pro Dokument deckeln (bounded download/RAM)
        if len(out) >= max_docs:
            break
    return out


def build_heldout(n_docs_per_domain=300, skip=3500, max_chars=8000,
                  src_dir=DATA_ROOT, out_dir=None, domains=None, verbose=True):
    """Held-out-Set: FRISCHE Dokumente derselben Domaenen (gestreamt mit Offset `skip`, daher
    disjunkt vom Trainingssample), tokenisiert mit dem BEREITS trainierten Tokenizer aus
    src_dir. Gleiches Speicherformat wie build() -> via HeteroMiniData(out_dir) ladbar."""
    from tokenizers import ByteLevelBPETokenizer
    out_dir = out_dir or (DATA_ROOT + "_heldout")
    domains = domains or DOMAINS
    os.makedirs(out_dir, exist_ok=True)
    tok = ByteLevelBPETokenizer(os.path.join(src_dir, "vocab.json"),
                                os.path.join(src_dir, "merges.txt"))
    dom_names, ids_chunks, doc_start, doc_len, doc_domain = [], [], [], [], []
    pos = 0
    for spec in domains:
        name = spec[0]
        if verbose:
            print(f"[Held-out] streame '{name}' (skip {skip}) ...", flush=True)
        try:
            texts = _collect_texts(spec, n_docs_per_domain, max_chars, skip=skip)
        except Exception as e:
            print(f"[Held-out]   FEHLER '{name}': {type(e).__name__}: {str(e)[:100]}", flush=True)
            continue
        if not texts:
            continue
        di = len(dom_names); dom_names.append(name)
        for t in texts:
            enc = tok.encode("<bos>" + t).ids
            if len(enc) < 16:
                continue
            ids_chunks.append(np.asarray(enc, dtype=np.uint16))
            doc_start.append(pos); doc_len.append(len(enc)); doc_domain.append(di)
            pos += len(enc)
    ids = np.concatenate(ids_chunks)
    np.save(os.path.join(out_dir, "ids.npy"), ids)
    np.save(os.path.join(out_dir, "doc_start.npy"), np.asarray(doc_start, dtype=np.int64))
    np.save(os.path.join(out_dir, "doc_len.npy"), np.asarray(doc_len, dtype=np.int32))
    np.save(os.path.join(out_dir, "doc_domain.npy"), np.asarray(doc_domain, dtype=np.uint8))
    # vocab/merges fuer Decoding kopieren
    import shutil
    for fn in ("vocab.json", "merges.txt"):
        shutil.copy2(os.path.join(src_dir, fn), os.path.join(out_dir, fn))
    meta = {"domains": dom_names, "vocab_size": tok.get_vocab_size(),
            "n_tokens": int(ids.size), "n_docs": int(len(doc_start)),
            "heldout": True, "skip": skip}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    if verbose:
        print(f"[Held-out] fertig: {meta['n_tokens']:,} Tokens, {meta['n_docs']} Docs, "
              f"Domaenen {dom_names} -> {out_dir}", flush=True)
    return meta


def build(max_docs_per_domain=2000, max_chars=8000, vocab=8000,
          domains=None, out_dir=DATA_ROOT, verbose=True):
    from tokenizers import ByteLevelBPETokenizer
    domains = domains or DOMAINS
    os.makedirs(out_dir, exist_ok=True)

    dom_texts = {}
    for spec in domains:
        name = spec[0]
        if verbose:
            print(f"[HeteroMini] streame Domaene '{name}' ...", flush=True)
        try:
            dom_texts[name] = _collect_texts(spec, max_docs_per_domain, max_chars)
            if verbose:
                print(f"[HeteroMini]   {len(dom_texts[name])} Dokumente", flush=True)
        except Exception as e:
            print(f"[HeteroMini]   FEHLER bei '{name}': {type(e).__name__}: {str(e)[:120]}"
                  f" — uebersprungen", flush=True)
    dom_names = [n for n in (s[0] for s in domains) if dom_texts.get(n)]
    if len(dom_names) < 2:
        raise RuntimeError(f"Zu wenige Domaenen erfolgreich: {dom_names}")

    # Tokenizer auf der Mischung trainieren (gleicher Typ/vocab wie bisher)
    if verbose:
        print(f"[HeteroMini] trainiere ByteLevelBPE (vocab={vocab}) ...", flush=True)
    all_texts = [t for n in dom_names for t in dom_texts[n]]
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(all_texts, vocab_size=vocab, special_tokens=["<pad>", "<bos>"])
    tok.save_model(out_dir)

    # Tokenisieren, Dokumentgrenzen + Domaene erhalten
    ids_chunks, doc_start, doc_len, doc_domain = [], [], [], []
    pos = 0
    for di, name in enumerate(dom_names):
        for t in dom_texts[name]:
            enc = tok.encode("<bos>" + t).ids
            if len(enc) < 16:
                continue
            ids_chunks.append(np.asarray(enc, dtype=np.uint16))
            doc_start.append(pos)
            doc_len.append(len(enc))
            doc_domain.append(di)
            pos += len(enc)
    ids = np.concatenate(ids_chunks)
    doc_start = np.asarray(doc_start, dtype=np.int64)
    doc_len = np.asarray(doc_len, dtype=np.int32)
    doc_domain = np.asarray(doc_domain, dtype=np.uint8)

    np.save(os.path.join(out_dir, "ids.npy"), ids)
    np.save(os.path.join(out_dir, "doc_start.npy"), doc_start)
    np.save(os.path.join(out_dir, "doc_len.npy"), doc_len)
    np.save(os.path.join(out_dir, "doc_domain.npy"), doc_domain)
    meta = {
        "domains": dom_names,
        "vocab_size": tok.get_vocab_size(),
        "n_tokens": int(ids.size),
        "n_docs": int(doc_start.size),
        "tokens_per_domain": {dom_names[d]: int(doc_len[doc_domain == d].sum())
                              for d in range(len(dom_names))},
        "docs_per_domain": {dom_names[d]: int((doc_domain == d).sum())
                            for d in range(len(dom_names))},
        "max_chars": max_chars,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    if verbose:
        print(f"[HeteroMini] fertig: {meta['n_tokens']:,} Tokens, {meta['n_docs']} Docs, "
              f"Domaenen {dom_names}", flush=True)
        print(f"[HeteroMini] tokens/domain: {meta['tokens_per_domain']}", flush=True)
    return meta


class HeteroMiniData:
    """Batches mit domain_label; zwei Modi (contiguous / shuffled)."""

    def __init__(self, out_dir=DATA_ROOT):
        import torch
        self.torch = torch
        self.ids = np.load(os.path.join(out_dir, "ids.npy"))
        self.doc_start = np.load(os.path.join(out_dir, "doc_start.npy"))
        self.doc_len = np.load(os.path.join(out_dir, "doc_len.npy"))
        self.doc_domain = np.load(os.path.join(out_dir, "doc_domain.npy"))
        with open(os.path.join(out_dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.domains = self.meta["domains"]
        self.vocab_size = self.meta["vocab_size"]
        self.n_domains = len(self.domains)

    def batch(self, batch_size, seq_len, device="cpu", mode="contiguous", regimes=None):
        import torch
        need = seq_len + 1
        if mode == "shuffled":
            N = self.ids.size - 1
            starts = np.random.randint(0, N - 1, size=batch_size * seq_len)
            flat = self.ids[starts].astype(np.int64)
            flatp1 = self.ids[starts + 1].astype(np.int64)
            toks = torch.from_numpy(flat.reshape(batch_size, seq_len)).to(device)
            tgt = torch.from_numpy(flatp1.reshape(batch_size, seq_len)).to(device)
            dom = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        else:  # contiguous: Fenster innerhalb EINES Dokuments
            ok = np.where(self.doc_len >= need)[0]
            sel = ok[np.random.randint(0, ok.size, size=batch_size)]
            toks_np = np.empty((batch_size, seq_len), dtype=np.int64)
            tgt_np = np.empty((batch_size, seq_len), dtype=np.int64)
            dom_np = np.empty(batch_size, dtype=np.int64)
            for i, d in enumerate(sel):
                s = self.doc_start[d]
                off = np.random.randint(0, self.doc_len[d] - need + 1)
                w = self.ids[s + off: s + off + need].astype(np.int64)
                toks_np[i] = w[:-1]
                tgt_np[i] = w[1:]
                dom_np[i] = self.doc_domain[d]
            toks = torch.from_numpy(toks_np).to(device)
            tgt = torch.from_numpy(tgt_np).to(device)
            dom = torch.from_numpy(dom_np).to(device)
        mask = torch.ones_like(toks, dtype=torch.bool)
        return toks, tgt, mask, dom


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_docs_per_domain", type=int, default=2000)
    ap.add_argument("--max_chars", type=int, default=8000)
    ap.add_argument("--vocab", type=int, default=8000)
    ap.add_argument("--out_dir", default=DATA_ROOT)
    a = ap.parse_args()
    build(max_docs_per_domain=a.max_docs_per_domain, max_chars=a.max_chars,
          vocab=a.vocab, out_dir=a.out_dir)
