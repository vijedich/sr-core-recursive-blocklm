"""Routing regularisation losses for cache-aware training.

core_locality_loss       -- OLD: gradient via route_gates only (probe v1, ineffective)
soft_full_jaccard_loss   -- NEW: gradient via full pre-topk router_probs (Modus 1)
soft_sharp_jaccard_loss  -- NEW: sharpened full probs, emphasises topk candidates (Modus 2)
mean_core_overlap        -- non-differentiable hard metric for logging
router_entropy           -- mean entropy of full routing distribution
"""
from __future__ import annotations
import torch


def core_locality_loss(route_idx: torch.Tensor, route_gates: torch.Tensor,
                       n_blocks: int, B: int, T: int) -> torch.Tensor:
    """Soft consecutive-token Jaccard overlap penalty (R=1 only).

    Penalises routing paths where adjacent tokens choose completely different
    blocks. Gradient flows through route_gates (soft gate weights over the
    k selected blocks), not through the hard block-selection indices.

    Loss = 1 - mean_soft_jaccard(t, t-1)  in [0, 1].
    0 = perfect overlap every step, 1 = no overlap at all.

    Args:
        route_idx  : (B*T, k) hard block indices — in-graph (not detached)
        route_gates: (B*T, k) soft gate weights  — in-graph
        n_blocks   : total number of blocks in bank
        B, T       : batch size, sequence length
    """
    k = route_idx.shape[-1]
    idx  = route_idx.view(B, T, k)
    gate = route_gates.view(B, T, k)

    # Sparse soft mask (B, T, n_blocks): gate mass at selected block positions.
    # scatter_ is differentiable w.r.t. source (gate), not w.r.t. idx.
    soft = gate.new_zeros(B, T, n_blocks)
    soft.scatter_(2, idx, gate)

    p_t    = soft[:, 1:, :]   # (B, T-1, n_blocks)
    p_prev = soft[:, :-1, :]  # (B, T-1, n_blocks)

    intersection = torch.minimum(p_t, p_prev).sum(dim=-1)          # (B, T-1)
    union        = torch.maximum(p_t, p_prev).sum(dim=-1).clamp_min(1e-8)

    return 1.0 - (intersection / union).mean()


def soft_full_jaccard_loss(router_probs_bt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Soft Jaccard on full pre-topk router probabilities (Modus 1).

    Gradient flows through router_probs_bt all the way to routing logits.
    Encourages consecutive tokens to have similar routing distributions,
    which should push the hard top-k selection toward overlapping blocks.

    Args:
        router_probs_bt: (B, T, n_blocks) — in-graph, from aux["iters"][0]["router_probs"]
    """
    p_t    = router_probs_bt[:, 1:, :]   # (B, T-1, n_blocks)
    p_prev = router_probs_bt[:, :-1, :]  # (B, T-1, n_blocks)
    inter  = torch.minimum(p_t, p_prev).sum(dim=-1)
    union  = torch.maximum(p_t, p_prev).sum(dim=-1).clamp_min(eps)
    return 1.0 - (inter / union).mean()


def soft_sharp_jaccard_loss(router_probs_bt: torch.Tensor, alpha: float = 2.0,
                            eps: float = 1e-8) -> torch.Tensor:
    """Sharpened Soft Jaccard on router probabilities (Modus 2).

    Raises probabilities to the power alpha before normalising, concentrating
    mass on high-probability blocks. At alpha=1 this equals soft_full_jaccard_loss.
    At alpha=2/4 the loss focuses on candidates likely to end up in top-k.

    Args:
        router_probs_bt: (B, T, n_blocks) — in-graph
        alpha           : sharpening exponent (default 2.0)
    """
    q = router_probs_bt.pow(alpha)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
    return soft_full_jaccard_loss(q, eps=eps)


@torch.no_grad()
def router_entropy(router_probs_bt: torch.Tensor, eps: float = 1e-8) -> float:
    """Mean Shannon entropy of the routing distribution — for logging."""
    p = router_probs_bt.clamp_min(eps)
    return float(-(p * p.log()).sum(dim=-1).mean().item())


def router_entropy_loss(router_probs_bt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Differentiable mean Shannon entropy H(p) — for entropy minimization.

    Add positively to the total loss with a small lambda to drive the router
    toward sharper, more decisive distributions:
        loss += lambda_entropy * router_entropy_loss(router_probs)

    Returns H(p) >= 0.  Gradient pushes probability mass toward dominant blocks.
    """
    p = router_probs_bt.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1).mean()


def router_entropy_loss_targeted(router_probs_bt: torch.Tensor,
                                 target_entropy: float,
                                 eps: float = 1e-8) -> torch.Tensor:
    """Bounded entropy consolidation — only penalises routers that are too diffuse.

    Unlike router_entropy_loss (which minimises H unconditionally), this loss
    applies pressure only when H(p) > target_entropy and is zero once the router
    has already reached the target sharpness:

        loss += lambda_target * router_entropy_loss_targeted(router_probs, H_target)

    This separates consolidation (H > H_target → push down) from over-sharpening
    (H < H_target → no gradient), which is the key difference vs. pure entropy min.

    Typical target values derived from the entmin sweep:
        3.832 = ctrl@17k (no pressure)
        3.777 = lam003   (generalization-balanced)
        3.732 = lam005   (systems sweet spot)
        3.647 = lam007   (breakpoint)
    """
    p = router_probs_bt.clamp_min(eps)
    H = -(p * p.log()).sum(dim=-1).mean()
    return torch.relu(H - target_entropy)


@torch.no_grad()
def mean_core_overlap(topk_idx_bt: torch.Tensor) -> float:
    """Hard Jaccard overlap between consecutive tokens — for logging only.

    Args:
        topk_idx_bt: (B, T, k) int64 tensor of selected block indices (detached)

    Returns mean Jaccard over all consecutive token pairs and batch.
    """
    B, T, k = topk_idx_bt.shape
    if T < 2:
        return float("nan")
    n_blocks = int(topk_idx_bt.max().item()) + 1

    oh = torch.zeros(B, T, n_blocks, dtype=torch.bool, device=topk_idx_bt.device)
    oh.scatter_(2, topk_idx_bt, True)

    inter = (oh[:, 1:, :] & oh[:, :-1, :]).sum(dim=-1).float()
    union = (oh[:, 1:, :] | oh[:, :-1, :]).sum(dim=-1).clamp_min(1).float()

    return round(float((inter / union).mean().item()), 4)
