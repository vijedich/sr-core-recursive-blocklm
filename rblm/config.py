"""Configuration objects for the Phase-1 prototype.

The *defaults* here follow the user's Phase-1 specification (d=256, 64 blocks,
k=4, up to 6 iterations). The in-sandbox demo (run_demo.py) overrides these with
a smaller config so it runs on 1 CPU; the science is identical, only the scale
differs.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int = 16000
    d_model: int = 256          # semantic state dimension  (h_r in the plan)
    block_hidden: int = 512     # MLP block inner width
    n_heads: int = 4            # heads of the shared context encoder
    context_layers: int = 1     # deliberately minimal: leaves work for the core
    max_len: int = 256
    tie_head: bool = True       # tie output head to embedding -> param-fair A/B/C

    # ---- core-specific ----
    variant: str = "C"          # "A" dense | "B" recurrent | "C" routed
    # A: number of distinct stacked blocks (== max depth)
    dense_depth: int = 24
    # B: number of recurrent applications of the single shared block
    recurrent_iters: int = 24
    # C: routed block bank
    n_blocks: int = 64
    k_active: int = 4
    n_candidates: int = 16      # Phase-3 stub: neighbourhood size (unused when ==n_blocks)
    routed_iters: int = 6
    key_dim: int = 64
    router_noise_std: float = 0.3   # noisy-top-k exploration (train only)
    coord_dim: int = 3              # fixed spatial coordinates (logged, not used in Phase 1)
    recurrent_read: bool = False    # Exp1: shared in-loop attention so each iteration can re-read
    # Sparse-Recursive-Fixed-Core (testet, ob ein kleiner wiederverwendeter Kern Tiefe ersetzt):
    #   None             -> Naked Sparse: jede Iteration frei top-k routen (Default)
    #   "per_token"      -> SR-Core: r1 routet den Kern, r2..rR verwenden DENSELBEN Kern wieder
    #   "core_satellite" -> SR-Core+Satellit: r1 waehlt (k-1) Kernbloecke (fix), je Iteration
    #                       1 frei geroutetes Satellitenblock dazu
    core_mode: Optional[str] = None


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 64
    seq_len: int = 64
    lr: float = 3e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup: int = 100
    eval_every: int = 200
    eval_batches: int = 8
    # loss weighting across iterations (deep supervision). "equal" or "linear"
    iter_loss_weighting: str = "equal"
    lb_loss_weight: float = 0.01     # moderate load-balancing (Switch aux)
    seed: int = 0
    device: str = "cpu"
    log_routing: bool = True


@dataclass
class DataConfig:
    kind: str = "synthetic"          # "synthetic" | "tinystories"
    n_regimes: int = 4
    base: int = 8                    # symbol arithmetic base for regimes
    seq_len: int = 64
    # tinystories only:
    ts_vocab: int = 8000
    ts_max_docs: int = 20000


@dataclass
class ExpConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    name: str = "exp"

    def to_dict(self):
        return asdict(self)
