"""Routed block bank (Model C core).

One recursive iteration:
  1. router scores all blocks for every token   (q . k_b)
  2. noisy top-k selection (k active blocks / token)   <- true sparsity
  3. sparse dispatch: each token is computed ONLY through its k blocks
  4. h <- h + sum_b gate_b * F_b(h)

Phase-1 choices (per the user):
  * soft *gates* (renormalised softmax over the selected k) -> differentiable
  * hard top-k *selection* -> only k blocks are actually computed (faithful FLOPs)
  * fixed coordinates: stored, logged, NOT used for routing yet (Phase-3 stub)
  * candidate neighbourhood == all blocks for now (n_candidates stub)

Returns, per iteration, an aux dict with everything the metrics need:
topk_idx, gate mass, full soft prob (for entropy/importance), load-balance loss.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualMLPBlock


class RoutedBlockBank(nn.Module):
    def __init__(self, d, n_blocks, k_active, key_dim,
                 block_hidden, noise_std=0.3, coord_dim=3):
        super().__init__()
        self.d = d
        self.n_blocks = n_blocks
        self.k = k_active
        self.noise_std = noise_std
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(d, block_hidden) for _ in range(n_blocks)]
        )
        self.route_norm = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, key_dim)
        self.keys = nn.Parameter(torch.randn(n_blocks, key_dim) / math.sqrt(key_dim))
        # Phase-3: trainable spatial coordinates (were a fixed buffer in Phase 1)
        self.coords = nn.Parameter(torch.randn(n_blocks, coord_dim))

    def route(self, h_flat, training):
        """h_flat: (N, d) -> topk_idx (N,k), gates (N,k), full_probs (N,B)."""
        q = self.q_proj(self.route_norm(h_flat))
        logits = q @ self.keys.t() / math.sqrt(q.shape[-1])  # (N, B)
        if training and self.noise_std > 0:
            logits = logits + self.noise_std * torch.randn_like(logits)
        # Defensive: keep logits finite and bounded. A single NaN/Inf scoring row
        # otherwise propagates into topk_idx and the sort-based dispatch, where it
        # has caused silent CUDA crashes (Seed-2-Stabilitaetsfall). Clamp to a wide
        # band that never affects healthy training (softmax saturates well before).
        logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
        logits = logits.clamp(-30.0, 30.0)
        full_probs = logits.softmax(dim=-1)
        topk_val, topk_idx = logits.topk(self.k, dim=-1)
        gates = topk_val.softmax(dim=-1)
        return topk_idx, gates, full_probs

    def dispatch(self, h_flat, topk_idx, gates):
        """Sort-based sparse dispatch: O(1) CPU-GPU syncs instead of O(n_blocks).

        Tokens are sorted by their assigned block ID. One sync to read the
        block boundaries, then each block runs only its assigned tokens
        (sparse FLOPs preserved). Old per-block loop caused n_blocks * R
        syncs per forward pass (~384 at n_blocks=64, R=6).
        """
        N, d = h_flat.shape
        k = self.k
        # Each token i appears k times, paired with its k block assignments
        tok_idx = torch.arange(N, device=h_flat.device).repeat_interleave(k)  # (N*k,)
        block_ids = topk_idx.reshape(-1)    # (N*k,)
        gate_weights = gates.reshape(-1)    # (N*k,)
        # Sort by block ID — single GPU op, no sync
        order = block_ids.argsort(stable=True)
        sorted_blocks = block_ids[order]
        sorted_toks = tok_idx[order]
        sorted_gates = gate_weights[order]
        # Boundaries between consecutive block IDs — single sync here
        unique_b, counts = sorted_blocks.unique_consecutive(return_counts=True)
        out = torch.zeros_like(h_flat)
        usage = torch.zeros(self.n_blocks, device=h_flat.device)
        start = 0
        for b_id, cnt in zip(unique_b.tolist(), counts.tolist()):
            end = start + cnt
            toks = sorted_toks[start:end]               # tokens using this block
            h_in = h_flat[toks]                         # (cnt, d) — sparse FLOPs
            y = self.blocks[b_id](h_in)
            g = sorted_gates[start:end].unsqueeze(-1)   # (cnt, 1)
            out.index_add_(0, toks, g * y)
            usage[b_id] += cnt
            start = end
        return out, usage

    def load_balance_loss(self, full_probs, topk_idx):
        """Switch-style aux: n * sum(importance_b * load_b). Minimised at balance."""
        N = full_probs.shape[0]
        importance = full_probs.mean(dim=0)                 # (B,)
        onehot = torch.zeros(N, self.n_blocks, device=full_probs.device)
        onehot.scatter_(1, topk_idx, 1.0)
        load = onehot.mean(dim=0)                            # (B,)
        return self.n_blocks * (importance * load).sum()

    def coord_repulsion_loss(self):
        """Push block coordinates apart so they span 3-D space (Phase 3)."""
        c = self.coords                                 # (n_blocks, 3)
        diff = c.unsqueeze(0) - c.unsqueeze(1)         # (n, n, 3)
        dist2 = (diff ** 2).sum(-1)                    # (n, n)
        n = self.n_blocks
        mask = torch.triu(
            torch.ones(n, n, device=c.device, dtype=torch.bool), diagonal=1)
        return (1.0 / (dist2[mask] + 0.01)).mean()

    def forward(self, h, training, ablate_mask=None, random_route=False,
                prev_full_probs=None):
        """h: (B,T,d) -> new_h (B,T,d), aux dict for this iteration.

        prev_full_probs: detached (N, n_blocks) from previous iteration —
          when provided, computes a soft Jaccard diversity loss so the router
          learns to choose different blocks than the previous iteration (Phase 3).
        """
        B, T, d = h.shape
        h_flat = h.reshape(-1, d)
        topk_idx, gates, full_probs = self.route(h_flat, training)
        if random_route:  # shuffle control: ignore learned routing, pick k at random
            N = h_flat.shape[0]
            topk_idx = torch.stack(
                [torch.randperm(self.n_blocks, device=h.device)[:self.k]
                 for _ in range(N)], dim=0)
            gates = torch.full((N, self.k), 1.0 / self.k, device=h.device)
        if ablate_mask is not None:
            # force-disable some blocks: re-route avoiding them (analysis only)
            logits = (self.q_proj(self.route_norm(h_flat)) @ self.keys.t()
                      / math.sqrt(self.keys.shape[-1]))
            logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
            logits = logits.clamp(-30.0, 30.0)
            logits = logits.masked_fill(ablate_mask.view(1, -1), float("-inf"))
            full_probs = logits.softmax(dim=-1)
            topk_val, topk_idx = logits.topk(self.k, dim=-1)
            gates = topk_val.softmax(dim=-1)
        out, usage = self.dispatch(h_flat, topk_idx, gates)
        lb = self.load_balance_loss(full_probs, topk_idx)
        # Diversity loss: inner-product overlap with previous iteration's routing.
        # Gradient flows through full_probs (current), not through prev_full_probs
        # (which is already detached), so each router step learns to diverge from
        # the step before it without destabilising earlier steps.
        if prev_full_probs is not None:
            div_loss = (full_probs * prev_full_probs).sum(dim=-1).mean()
        else:
            div_loss = full_probs.new_zeros(())
        new_h = h + out.reshape(B, T, d)
        aux = {
            "topk_idx": topk_idx.detach().view(B, T, self.k),
            "full_probs": full_probs.detach(),     # (N,B) detached — logging/analysis
            "router_probs": full_probs.view(B, T, self.n_blocks),  # (B,T,n_blocks) in-graph
            "usage": usage.detach(),
            "lb_loss": lb,
            "div_loss": div_loss,                  # scalar, in graph for Phase-3 training
            # Rohe Route fuer SR-Core-Reuse (in-graph gates, detached idx):
            "route_idx": topk_idx,                 # (N,k)
            "route_gates": gates,                  # (N,k), in-graph
        }
        return new_h, aux

    def apply_route(self, h, topk_idx, gates, full_probs_ref):
        """SR-Core: eine GEGEBENE Route auf das aktuelle h anwenden (KEIN Router).
        Verwendet die in r1 gewaehlten Bloecke wieder, aber auf dem fortgeschriebenen h.
        full_probs_ref nur fuer Metrik-Kompatibilitaet (r1-full_probs durchreichen)."""
        B, T, d = h.shape
        h_flat = h.reshape(-1, d)
        out, usage = self.dispatch(h_flat, topk_idx, gates)
        new_h = h + out.reshape(B, T, d)
        aux = {
            "topk_idx": topk_idx.detach().view(B, T, self.k),
            "full_probs": full_probs_ref,          # r1-Referenz detached
            "router_probs": None,                  # kein Router bei SR-Core-Reuse
            "usage": usage.detach(),
            "lb_loss": h.new_zeros(()),
            "div_loss": h.new_zeros(()),
            "route_idx": topk_idx,
            "route_gates": gates,
        }
        return new_h, aux

    def apply_core_satellite(self, h, core_idx, training):
        """SR-Core+Satellit: (k-1) feste Kernbloecke (core_idx) + 1 frei geroutetes
        Satellitenblock (top-1 ausserhalb des Kerns) auf das aktuelle h. Router bleibt
        je Iteration aktiv (nur fuer den Satelliten)."""
        B, T, d = h.shape
        h_flat = h.reshape(-1, d)
        N = h_flat.shape[0]
        q = self.q_proj(self.route_norm(h_flat))
        logits = q @ self.keys.t() / math.sqrt(q.shape[-1])
        if training and self.noise_std > 0:
            logits = logits + self.noise_std * torch.randn_like(logits)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        # Kern maskieren, damit der Satellit ein NEUER Block ist
        mask = torch.zeros(N, self.n_blocks, dtype=torch.bool, device=h.device)
        mask.scatter_(1, core_idx, True)
        sat_idx = logits.masked_fill(mask, float("-inf")).argmax(dim=-1, keepdim=True)  # (N,1)
        full_idx = torch.cat([core_idx, sat_idx], dim=1)                                 # (N,k)
        sel_logits = torch.gather(logits, 1, full_idx)
        gates = sel_logits.softmax(dim=-1)
        out, usage = self.dispatch(h_flat, full_idx, gates)
        new_h = h + out.reshape(B, T, d)
        full_probs = logits.softmax(dim=-1)
        lb = self.load_balance_loss(full_probs, full_idx)
        aux = {
            "topk_idx": full_idx.detach().view(B, T, self.k),
            "full_probs": full_probs.detach(),
            "usage": usage.detach(),
            "lb_loss": lb,
            "div_loss": h.new_zeros(()),
            "route_idx": full_idx,
            "route_gates": gates,
        }
        return new_h, aux
