"""TinyStories loader.

NOTE: this requires network access to huggingface.co, which is NOT available in
the build sandbox (only pypi/github/etc are allowlisted). It is provided so the
prototype runs unchanged once you enable network or run locally:

    pip install datasets tokenizers
    python -m rblm.tinystories  # downloads + caches

If the download fails, run_demo.py automatically stays on synthetic data.
"""
from __future__ import annotations
import torch


def try_build_tinystories(dcfg):
    try:
        from datasets import load_dataset
        from tokenizers import ByteLevelBPETokenizer
    except Exception as e:
        return None, f"datasets/tokenizers not installed ({e})"
    try:
        ds = load_dataset("roneneldan/TinyStories", split="train",
                          streaming=False)
    except Exception as e:
        return None, f"download failed (likely no network to huggingface): {e}"
    texts = [ds[i]["text"] for i in range(min(dcfg.ts_max_docs, len(ds)))]
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(texts, vocab_size=dcfg.ts_vocab,
                            special_tokens=["<pad>", "<bos>"])
    ids = []
    for t in texts:
        ids.extend(tok.encode("<bos>" + t).ids)
    data = torch.tensor(ids, dtype=torch.long)
    return TinyStoriesData(data, tok.get_vocab_size()), "ok"


class TinyStoriesData:
    """Plain causal-LM batches; no regime tags (specialisation is unsupervised)."""
    def __init__(self, ids, vocab_size):
        self.ids = ids
        self.vocab_size = vocab_size

    def batch(self, batch_size, seq_len, device="cpu", regimes=None):
        import torch
        N = self.ids.numel() - 1
        starts = torch.randint(0, N - seq_len, (batch_size,))
        toks = torch.stack([self.ids[s:s + seq_len] for s in starts]).to(device)
        tgt = torch.stack([self.ids[s + 1:s + 1 + seq_len] for s in starts]).to(device)
        mask = torch.ones_like(toks, dtype=torch.bool)
        reg = torch.zeros(batch_size, dtype=torch.long, device=device)
        return toks, tgt, mask, reg


if __name__ == "__main__":
    from .config import DataConfig
    d, msg = try_build_tinystories(DataConfig())
    print(msg, None if d is None else d.vocab_size)
