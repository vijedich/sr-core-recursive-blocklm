"""Erstellt hm_cont_hm_srcore_b64_k8_R6_s0.pt aus dem 10k-Checkpoint.

Optimizer-State ist kalt (neu initialisiert), Scheduler wird auf
step=10000 eines lr_horizon=15000 Cosine-Schedules vorgespult.
"""
import os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rblm.heteromini import HeteroMiniData
from experiments.tinystories_exp import make_model, _lr, RESULTS

LR_HORIZON = 15000
STEP_FROM  = 10000
SEED       = 0

data = HeteroMiniData()
model, _ = make_model(data.vocab_size, n_blocks=64, k=8, R=6,
                      device="cpu", core_mode="per_token")

weights = torch.load(
    os.path.join(ROOT, "checkpoints/hm_srcore_b64_R6/seed_0/step_10000/model.pt"),
    map_location="cpu", weights_only=True)
model.load_state_dict(weights)
print(f"Modell geladen: {sum(p.numel() for p in model.parameters())/1e6:.1f}M Params")

opt   = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr(s, 200, LR_HORIZON))

# Scheduler auf step=10000 vorspulen (nur sched.step, kein Training)
for _ in range(STEP_FROM):
    sched.step()
lr_now = opt.param_groups[0]["lr"]
print(f"LR an step {STEP_FROM}: {lr_now:.5f}  (lr_horizon={LR_HORIZON})")

out = os.path.join(RESULTS, "hm_cont_hm_srcore_b64_k8_R6_s0.pt")
torch.save({
    "model":     model.state_dict(),
    "optimizer": opt.state_dict(),
    "scheduler": sched.state_dict(),
    "rng_torch": torch.get_rng_state(),
    "rng_cuda":  None,
    "rng_numpy": np.random.get_state(),
    "step":      STEP_FROM,
    "config": {
        "variant": "sparse", "n_blocks": 64, "k": 8, "R": 6,
        "depth": 8, "core_mode": "per_token", "vocab_size": data.vocab_size,
    },
}, out)
print(f"Cont-Snapshot: {out}")
