"""Regenerate routing traces from the trained C model (seed 0).

A trace is, for a lockstep decode batch of B sequences, the block ids requested
at every (position t, inner iteration r):  array of shape (R, B, T, k).

This is exactly what a realistic batched autoregressive decoder would request:
positions left-to-right are the generation order; the R inner iterations are the
recursive refinement of each token. We feed this stream to the cache simulator.
"""
from __future__ import annotations
import os, copy, json
import numpy as np
import torch

from rblm.config import ExpConfig
from rblm.models import build_model
from rblm.synthetic import SyntheticData
from run_demo import CONFIGS, DATA, RESULTS, base_train


@torch.no_grad()
def extract(n_batches=4, seed=0):
    data = SyntheticData(DATA.n_regimes, DATA.base, seed=999)  # held-out stream
    mcfg = copy.deepcopy(CONFIGS["C_routed"]); mcfg.vocab_size = data.vocab_size
    model = build_model(mcfg)
    model.load_state_dict(torch.load(
        os.path.join(RESULTS, f"model_C_routed_s{seed}.pt"), weights_only=True))
    model.eval()
    tcfg = base_train(0)
    R, k, nb = mcfg.routed_iters, mcfg.k_active, mcfg.n_blocks
    traces, regimes, masks = [], [], []
    for _ in range(n_batches):
        toks, tgt, mask, reg = data.batch(tcfg.batch_size, tcfg.seq_len)
        _, aux = model(toks)
        # (R, B, T, k)
        arr = torch.stack([a["topk_idx"] for a in aux["iters"]], dim=0).cpu().numpy()
        traces.append(arr); regimes.append(reg.cpu().numpy()); masks.append(mask.cpu().numpy())
    meta = dict(R=R, k=k, n_blocks=nb,
                B=traces[0].shape[1], T=traces[0].shape[2])
    np.savez_compressed(os.path.join(RESULTS, "traces_s0.npz"),
                        traces=np.stack(traces), regimes=np.stack(regimes),
                        masks=np.stack(masks), meta=json.dumps(meta))
    print("saved traces:", np.stack(traces).shape, "meta:", meta)
    return meta


if __name__ == "__main__":
    extract()
