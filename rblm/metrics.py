"""Routing instrumentation for Model C (computed over masked positions).

Everything the Phase-1 questions need:
  Q2 (depth helps)        -> per-iteration loss curve (in train.py)
  Q3 (specialisation)     -> regime<->block mutual information + per-regime usage
  Q4 (no collapse)        -> router entropy, dead-block count, max block share
  streaming scaffold      -> Jaccard overlap & delta_r (LOGGED ONLY in Phase 1)
"""
from __future__ import annotations
import math
import torch


def _membership(topk_idx, n_blocks):
    """(N,k) indices -> (N,n_blocks) {0,1} membership."""
    N = topk_idx.shape[0]
    mem = torch.zeros(N, n_blocks, dtype=torch.bool)
    mem.scatter_(1, topk_idx, True)
    return mem


class RoutingAccumulator:
    def __init__(self, n_blocks, n_iters, n_regimes):
        self.n_blocks = n_blocks
        self.n_iters = n_iters
        self.n_regimes = n_regimes
        self.usage = torch.zeros(n_iters, n_blocks)            # selection counts
        self.prob_sum = torch.zeros(n_iters, n_blocks)         # soft prob mass
        self.tok_count = torch.zeros(n_iters)
        self.regime_block = torch.zeros(n_regimes, n_blocks)   # co-occurrence
        # streaming scaffold (between consecutive iterations)
        self.jacc_sum = torch.zeros(n_iters - 1)
        self.delta_sum = torch.zeros(n_iters - 1)
        self.pair_count = torch.zeros(n_iters - 1)
        self.union_blocks = [set() for _ in range(n_iters)]

    @torch.no_grad()
    def update(self, aux, mask, regime):
        iters = aux["iters"]
        B, T = mask.shape
        flat_mask = mask.reshape(-1)
        # regime per token (broadcast sequence regime to its positions)
        reg_tok = regime.view(B, 1).expand(B, T).reshape(-1)[flat_mask]
        mems = []
        for r, a in enumerate(iters):
            idx = a["topk_idx"].reshape(-1, a["topk_idx"].shape[-1])[flat_mask]  # (M,k)
            mem = _membership(idx, self.n_blocks)               # (M,nb)
            mems.append(mem)
            self.usage[r] += mem.sum(dim=0).float()
            self.prob_sum[r] += a["full_probs"].reshape(B * T, -1)[flat_mask].sum(0)
            self.tok_count[r] += mem.shape[0]
            self.union_blocks[r] |= set(torch.unique(idx).tolist())
            if r == 0:  # regime<->block co-occurrence on iteration 1 (routing entry)
                for g in range(self.n_regimes):
                    sel = reg_tok == g
                    if sel.any():
                        self.regime_block[g] += mem[sel].sum(dim=0).float()
        for r in range(self.n_iters - 1):
            a, b = mems[r], mems[r + 1]
            inter = (a & b).sum(dim=1).float()
            union = (a | b).sum(dim=1).float()
            jacc = (inter / union.clamp(min=1))
            delta = (b & ~a).sum(dim=1).float()
            self.jacc_sum[r] += jacc.sum()
            self.delta_sum[r] += delta.sum()
            self.pair_count[r] += a.shape[0]

    def finalize(self):
        out = {}
        # --- entropy / collapse (mean soft routing distribution per iter) ---
        ent, max_share = [], []
        for r in range(self.n_iters):
            p = self.prob_sum[r] / self.prob_sum[r].sum().clamp(min=1)
            ent.append(float(-(p * (p + 1e-12).log()).sum() / math.log(self.n_blocks)))
            max_share.append(float(p.max()))
        out["router_entropy_norm"] = ent           # 1.0 = perfectly balanced
        out["max_block_share"] = max_share
        # --- dead blocks (never selected across all iters) ---
        total_use = self.usage.sum(dim=0)
        out["dead_blocks"] = int((total_use == 0).sum())
        out["usage_per_block"] = (total_use / total_use.sum().clamp(min=1)).tolist()
        # --- streaming scaffold ---
        out["jaccard_consecutive"] = (self.jacc_sum / self.pair_count.clamp(min=1)).tolist()
        out["delta_blocks_mean"] = (self.delta_sum / self.pair_count.clamp(min=1)).tolist()
        out["union_blocks_per_iter"] = [len(s) for s in self.union_blocks]
        # --- specialisation: regime <-> block mutual information ---
        C = self.regime_block
        tot = C.sum().clamp(min=1)
        Prb = C / tot
        Pr = Prb.sum(dim=1, keepdim=True)
        Pb = Prb.sum(dim=0, keepdim=True)
        denom = (Pr * Pb).clamp(min=1e-12)
        mi = (Prb * (Prb.clamp(min=1e-12) / denom).log2()).sum()
        Hr = -(Pr * (Pr + 1e-12).log2()).sum()
        out["regime_block_MI_bits"] = float(mi)
        out["regime_block_MI_norm"] = float(mi / Hr.clamp(min=1e-6))  # 0..1
        out["regime_block_matrix"] = (C / C.sum(dim=1, keepdim=True).clamp(min=1)).tolist()
        return out
