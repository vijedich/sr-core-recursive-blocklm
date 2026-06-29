"""Depth-controlled task with a KNOWN true depth per example.

Modular permutation walk:  v_0 given; for i=1..d:  v_i = T[(v_{i-1}+a_i) mod base]
where T is a FIXED random permutation and a_i are per-step operands.

Why this is a good depth probe:
  * Non-collapsing: because operands a_i vary per step AND T is a non-linear
    permutation, the d-step result cannot be folded into one affine op, so a
    genuine sequential computation of d steps is required.
  * Known true depth d_q -> we can test corr(optimal_stop_depth, d_q).
  * Distractor tokens (DIST marker) forbid a pure positional heuristic.

Sequence layout (one supervised answer per sequence):
  BOS START <v0>  [ (DIST <rand>)* OP <a_i> ]*d   QUERY <answer>
The QUERY position is supervised to predict <answer> = v_d.
"""
from __future__ import annotations
import numpy as np
import torch

PAD, BOS, START, OP, DIST, QUERY = 0, 1, 2, 3, 4, 5
N_SPECIAL = 6


class DepthWalk:
    def __init__(self, base=8, d_max=6, max_distract=4, seed=0):
        self.base = base
        self.d_max = d_max
        self.max_distract = max_distract
        self.val0 = N_SPECIAL
        self.vocab_size = N_SPECIAL + base
        self.rng = np.random.default_rng(seed)
        self.T = np.random.default_rng(777).permutation(base)   # fixed global perm

    def _val_tok(self, v):
        return self.val0 + int(v) % self.base

    def _one(self, d, T_len):
        b = self.base
        v = self.rng.integers(0, b)
        toks = [BOS, START, self._val_tok(v)]
        for i in range(d):
            for _ in range(self.rng.integers(0, self.max_distract + 1)):
                toks += [DIST, self._val_tok(self.rng.integers(0, b))]
            a = self.rng.integers(0, b)
            toks += [OP, self._val_tok(a)]
            v = self.T[(v + a) % b]
        ans_pos = len(toks)             # QUERY goes here; answer predicted next
        toks += [QUERY, self._val_tok(v)]
        # pad / truncate
        if len(toks) > T_len:
            return None
        query_index = ans_pos           # position of QUERY token
        toks = toks + [PAD] * (T_len - len(toks))
        return np.array(toks, dtype=np.int64), query_index, int(v)

    def batch(self, batch_size, seq_len, device="cpu", depths=None):
        toks = np.full((batch_size, seq_len), PAD, dtype=np.int64)
        mask = np.zeros((batch_size, seq_len), dtype=bool)
        depth_ids = np.zeros(batch_size, dtype=np.int64)
        i = 0
        while i < batch_size:
            d = int(depths[i]) if depths is not None else int(self.rng.integers(1, self.d_max + 1))
            out = self._one(d, seq_len)
            if out is None:
                continue
            row, qidx, ans = out
            toks[i] = row
            mask[i, qidx] = True        # supervise the QUERY position
            depth_ids[i] = d
            i += 1
        t = torch.from_numpy(toks).to(device)
        target = torch.full_like(t, PAD)
        target[:, :-1] = t[:, 1:]
        return t, target, torch.from_numpy(mask).to(device), torch.from_numpy(depth_ids).to(device)
