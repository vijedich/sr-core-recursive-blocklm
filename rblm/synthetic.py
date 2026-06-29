"""Synthetic, regime-tagged sequences.

Each sequence carries a *regime tag* at position 1. The tag selects one of
several mechanistically distinct rules that determine the following symbols:

  REPEAT(L)   s[j] = s[j-L]                 (copy / identity, needs context fetch)
  INCREMENT   s[j] = (s[j-1] + 1) mod base  (predecessor + small nonlinearity)
  FIB         s[j] = (s[j-1]+s[j-2]) mod b  (two predecessors + add+mod -> "hard")
  ALTERNATE   s[j] = pattern[j mod P]       (purely positional cycle)

Why this design (maps to the four Phase-1 questions):
  * distinct mechanisms  -> blocks have a *reason* to specialise (Q3)
  * FIB is harder        -> extra recursive iterations should help most (Q2)
  * a clean loss mask on rule-determined positions -> sharp learning signal,
    so L_{r+1} < L_r is measurable rather than buried in unpredictable noise.
"""
from __future__ import annotations
import numpy as np
import torch

REGIME_NAMES = ["REPEAT", "INCREMENT", "FIB", "ALTERNATE"]
_REPEAT_LAG = 3
_ALT_PATTERN = None  # set per base


class SyntheticData:
    def __init__(self, n_regimes=4, base=8, seed=0):
        assert 1 <= n_regimes <= 4
        self.n_regimes = n_regimes
        self.base = base
        self.rng = np.random.default_rng(seed)
        # special ids
        self.PAD, self.BOS = 0, 1
        self.reg0 = 2
        self.content0 = 2 + n_regimes
        self.vocab_size = 2 + n_regimes + base
        # fixed global cycle pattern for ALTERNATE (length P)
        self.P = 4
        self.alt_pattern = (np.arange(self.P) * 2 + 1) % base

    # ---- one sequence of content symbols of length n (numpy) ----
    def _content(self, regime, n):
        b = self.base
        s = np.zeros(n, dtype=np.int64)
        if regime == 0:  # REPEAT
            L = _REPEAT_LAG
            s[:L] = self.rng.integers(0, b, size=min(L, n))
            for j in range(L, n):
                s[j] = s[j - L]
            determined_from = L
        elif regime == 1:  # INCREMENT
            s[0] = self.rng.integers(0, b)
            for j in range(1, n):
                s[j] = (s[j - 1] + 1) % b
            determined_from = 1
        elif regime == 2:  # FIB
            s[:2] = self.rng.integers(0, b, size=min(2, n))
            for j in range(2, n):
                s[j] = (s[j - 1] + s[j - 2]) % b
            determined_from = 2
        else:  # ALTERNATE
            for j in range(n):
                s[j] = self.alt_pattern[j % self.P]
            determined_from = 0
        return s, determined_from

    def batch(self, batch_size, seq_len, device="cpu", regimes=None):
        T = seq_len
        n_content = T - 2  # positions 0=BOS, 1=REG
        tokens = np.full((batch_size, T), self.PAD, dtype=np.int64)
        loss_mask = np.zeros((batch_size, T), dtype=bool)
        regime_ids = np.zeros(batch_size, dtype=np.int64)
        for i in range(batch_size):
            r = (regimes[i] if regimes is not None
                 else self.rng.integers(0, self.n_regimes))
            regime_ids[i] = r
            content, det_from = self._content(r, n_content)
            tokens[i, 0] = self.BOS
            tokens[i, 1] = self.reg0 + r
            tokens[i, 2:] = self.content0 + content
            # mask: predicting content symbol at content-index j (>=det_from)
            # next-token target at position t is tokens[t+1]; t+1>=2 -> j=t-1
            for t in range(1, T - 1):
                j = (t + 1) - 2
                if j >= det_from:
                    loss_mask[i, t] = True
        toks = torch.from_numpy(tokens).to(device)
        target = torch.full_like(toks, self.PAD)
        target[:, :-1] = toks[:, 1:]
        mask = torch.from_numpy(loss_mask).to(device)
        reg = torch.from_numpy(regime_ids).to(device)
        return toks, target, mask, reg
