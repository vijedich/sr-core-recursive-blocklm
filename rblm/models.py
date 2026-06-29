"""The three comparison models, sharing one backbone.

Shared (same architecture, separately trained weights):
  token embedding (+ learned positions) -> minimal causal context encoder
  -> CORE (differs)  -> shared readout head applied AFTER EACH iteration.

Cores:
  A  dense       : `dense_depth` distinct blocks, each applied once.
  B  recurrent   : ONE shared block applied `recurrent_iters` times.
  C  routed      : bank of `n_blocks`, top-`k` selected per iteration,
                   `routed_iters` iterations.

Every model is *deeply supervised*: the readout head runs after each block /
iteration, giving a per-depth quality curve and making "L_{r+1} < L_r" directly
measurable. Compute is reported in BLOCK-APPLICATIONS so A/B/C can be compared
at matched FLOPs (block size is identical everywhere).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualMLPBlock, ContextEncoder, CausalAttention
from .router import RoutedBlockBank


class SeqModelBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.max_len, cfg.d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.context = ContextEncoder(cfg.d_model, cfg.n_heads, cfg.context_layers)
        self.norm_out = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_head:
            self.head.weight = self.emb.weight

    def encode(self, tokens):
        T = tokens.shape[1]
        h = self.emb(tokens) + self.pos[:, :T]
        return self.context(h)

    def readout(self, h):
        return self.head(self.norm_out(h))

    # implemented by subclasses: returns (states, aux)
    def core(self, h0, training, **kw):
        raise NotImplementedError

    @property
    def n_iters(self):
        raise NotImplementedError

    def forward(self, tokens, **kw):
        h0 = self.encode(tokens)
        states, aux = self.core(h0, self.training, **kw)
        logits = [self.readout(h) for h in states]   # one per iteration
        return logits, aux

    def block_apps_at_iter(self, r):
        """Block-applications used up to and including iteration r (1-indexed)."""
        raise NotImplementedError


class ModelA(SeqModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(cfg.d_model, cfg.block_hidden)
             for _ in range(cfg.dense_depth)]
        )

    @property
    def n_iters(self):
        return self.cfg.dense_depth

    def core(self, h0, training, **kw):
        h = h0
        states = []
        for blk in self.blocks:
            h = h + blk(h)
            states.append(h)
        return states, {}

    def block_apps_at_iter(self, r):
        return r            # one distinct block per step


class ModelB(SeqModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.block = ResidualMLPBlock(cfg.d_model, cfg.block_hidden)

    @property
    def n_iters(self):
        return self.cfg.recurrent_iters

    def core(self, h0, training, **kw):
        h = h0
        states = []
        for _ in range(self.cfg.recurrent_iters):
            h = h + self.block(h)
            states.append(h)
        return states, {}

    def block_apps_at_iter(self, r):
        return r            # one (shared) block per step


class ModelC(SeqModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.bank = RoutedBlockBank(
            d=cfg.d_model, n_blocks=cfg.n_blocks, k_active=cfg.k_active,
            key_dim=cfg.key_dim, block_hidden=cfg.block_hidden,
            noise_std=cfg.router_noise_std, coord_dim=cfg.coord_dim,
        )
        self.read = (CausalAttention(cfg.d_model, cfg.n_heads)
                     if getattr(cfg, "recurrent_read", False) else None)

    @property
    def n_iters(self):
        return self.cfg.routed_iters

    def core(self, h0, training, ablate_mask=None, state_reset=False,
             random_route=False, diverse_train=False, diverse_from_iter: int = 0,
             core_mode=None, **kw):
        """diverse_train (Phase 3): at each iteration r≥2, hard-ablate the k blocks
        most used in iteration r-1.  Forces the router to explore new blocks each step
        instead of collapsing to the same k-set (Jaccard≈0.98 defect, see ablation).

        diverse_from_iter: first 0-based iteration index at which diversity is applied.
          0 (default) = all iterations (existing behaviour).
          2           = Curriculum-Variante C: nur r3–r6 in 1-basierter Notation.

        core_mode (Sparse-Recursive-Fixed-Core, ueberschreibt cfg.core_mode wenn gesetzt):
          None             -> Naked Sparse: jede Iteration frei top-k routen.
          "per_token"      -> r1 routet den Kern, r2..rR verwenden DENSELBEN Kern wieder
                              (kein Router nach r1) — testet, ob Reuse Tiefe ersetzt.
          "core_satellite" -> r1 waehlt (k-1) Kernbloecke, je Iteration 1 frei geroutetes
                              Satellitenblock dazu.
        """
        mode = core_mode if core_mode is not None else getattr(self.cfg, "core_mode", None)
        k = self.cfg.k_active
        h = h0
        states, auxes = [], []
        prev_fp = None
        core_route = None         # (idx (N,k), gates (N,k)) aus r1
        core_full_probs = None    # r1-full_probs (Metrik-Referenz fuer Reuse-Iterationen)
        diversity_loss = h0.new_zeros(())
        for r in range(self.cfg.routed_iters):
            base = h0 if state_reset else h
            x = self.read(base) if self.read is not None else base
            if mode in ("per_token", "core_satellite") and r >= 1 and core_route is not None:
                if mode == "per_token":
                    h, aux = self.bank.apply_route(
                        x, core_route[0], core_route[1], core_full_probs)
                else:  # core_satellite: feste (k-1) Kernbloecke + 1 Satellit
                    h, aux = self.bank.apply_core_satellite(
                        x, core_route[0][:, :max(1, k - 1)], training)
            else:
                # Compute per-iteration ablation mask when diverse_train is active
                iter_ablate = ablate_mask
                if diverse_train and training and auxes and len(auxes) >= diverse_from_iter:
                    prev_topk = auxes[-1]["topk_idx"]          # (B,T,k), detached
                    counts = torch.bincount(
                        prev_topk.reshape(-1),
                        minlength=self.cfg.n_blocks,
                    )
                    top_blocks = counts.topk(self.cfg.k_active).indices
                    iter_ablate = torch.zeros(
                        self.cfg.n_blocks, dtype=torch.bool, device=h0.device)
                    iter_ablate[top_blocks] = True
                    # Safety: guarantee ≥ k_active unmasked candidates.
                    n_valid = int((~iter_ablate).sum())
                    if n_valid < self.cfg.k_active:
                        release_n = self.cfg.k_active - n_valid
                        least_used = counts[top_blocks].argsort()[:release_n]
                        iter_ablate[top_blocks[least_used]] = False
                h, aux = self.bank(x, training, ablate_mask=iter_ablate,
                                   random_route=random_route, prev_full_probs=prev_fp)
                if r == 0 and mode in ("per_token", "core_satellite"):
                    core_route = (aux["route_idx"], aux["route_gates"])
                    core_full_probs = aux["full_probs"]
            diversity_loss = diversity_loss + aux["div_loss"]
            prev_fp = aux["full_probs"]  # already detached
            states.append(h)
            auxes.append(aux)
        coord_loss = self.bank.coord_repulsion_loss()
        return states, {"iters": auxes,
                        "diversity_loss": diversity_loss,
                        "coord_loss": coord_loss}

    def block_apps_at_iter(self, r):
        return r * self.cfg.k_active     # k blocks per iteration


def build_model(cfg):
    return {"A": ModelA, "B": ModelB, "C": ModelC}[cfg.variant](cfg)


# ---------- loss ----------
def iteration_losses(logits, target, mask):
    """Per-iteration masked cross-entropy. Returns tensor (n_iters,)."""
    losses = []
    flat_mask = mask.reshape(-1)
    tgt = target.reshape(-1)
    for lg in logits:
        l = F.cross_entropy(
            lg.reshape(-1, lg.shape[-1])[flat_mask], tgt[flat_mask],
            reduction="mean",
        )
        losses.append(l)
    return torch.stack(losses)


def weighting(n, kind):
    if kind == "linear":
        w = torch.arange(1, n + 1, dtype=torch.float32)
    elif kind == "end":
        w = torch.full((n,), 0.3 / max(1, n - 1), dtype=torch.float32)
        w[-1] = 0.7 if n > 1 else 1.0
        return w  # already normalised to sum 1
    else:
        w = torch.ones(n, dtype=torch.float32)
    return w / w.sum()
