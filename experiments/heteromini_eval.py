"""HeteroMini-Eval — alle Metriken fuer die Offloading-/Working-Set-Hypothese.

Pro Modus (contiguous = Dokument-zusammenhaengend, shuffled = voellig durchmischt):
  1. Working Set / Token (mittlere eindeutige Bloecke ueber R Iterationen)
  6. CacheMiss@K (LRU) fuer K=4/8/12/16/32
  8. Reuse-Distanz p50/p90/p99
 10. loss_per_iter / anytime

Nur contiguous (Domaene eindeutig):
  2. aktive Bloecke pro Domaene
  3. Domain-Jaccard (Top-Block-Mengen je Domaenenpaar)
  4. Domain-Routing-Classifier (Domaene aus Blocknutzung vorhersagen) vs. Zufall
  5. Gini + tote Bloecke
  9. theoretische Bytes/Token bei fp16 und int8
"""
from __future__ import annotations
import numpy as np
import torch

from rblm.models import iteration_losses
from experiments.tinystories_exp import gini, _simulate_lru


@torch.no_grad()
def _collect(model, data, mode, n_batches, bs, seq_len, device):
    """Sammelt Routing-Traces + Domaenen + Per-Iteration-Loss in einem Modus."""
    model.eval()
    R = model.n_iters
    k = model.cfg.k_active
    traces, doms = [], []
    loss_sum = torch.zeros(R, device=device)
    tok = 0
    for _ in range(n_batches):
        toks, tgt, mask, dom = data.batch(bs, seq_len, device=device, mode=mode)
        logits, aux = model(toks)
        per = iteration_losses(logits, tgt, mask)
        m = int(mask.sum()); loss_sum += per.detach() * m; tok += m
        # (R, bs, T, k)
        tr = np.stack([a["topk_idx"].cpu().numpy() for a in aux["iters"]])
        traces.append(tr)
        doms.append(dom.cpu().numpy())
    losses = (loss_sum / max(1, tok)).cpu().tolist()
    return traces, doms, losses, R, k


def _working_set(traces):
    """mittlere eindeutige Bloecke/Token ueber alle R Iterationen."""
    ws = []
    for tr in traces:                      # (R, bs, T, k)
        R, B, T, k = tr.shape
        a = tr.transpose(1, 2, 0, 3).reshape(B * T, R * k)   # (N, R*k)
        for row in a:
            ws.append(len(np.unique(row)))
    return float(np.mean(ws))


def _streams(traces):
    """Pro Sequenz die geordnete Block-Anforderungsliste (token-major, dann iter, dann k)."""
    out = []
    for tr in traces:                      # (R, bs, T, k)
        R, B, T, k = tr.shape
        for b in range(B):
            seq = []
            for t in range(T):
                for r in range(R):
                    seq.append(np.unique(tr[r, b, t]))   # eindeutige Bloecke dieser Iter
            out.append(seq)
    return out


def _cache_miss(streams, Ks, nb):
    res = {}
    for K in Ks:
        mr = np.mean([_simulate_lru(s, K, nb) for s in streams])
        res[K] = round(float(mr), 4)
    return res


def _reuse_distance(streams):
    """Abstand (in #Anforderungen) seit letzter Nutzung desselben Blocks."""
    dists = []
    for seq in streams:
        last = {}
        i = 0
        for req in seq:
            for b in req:
                b = int(b)
                if b in last:
                    dists.append(i - last[b])
                last[b] = i
                i += 1
    if not dists:
        return {"p50": 0, "p90": 0, "p99": 0}
    d = np.array(dists)
    return {"p50": float(np.percentile(d, 50)),
            "p90": float(np.percentile(d, 90)),
            "p99": float(np.percentile(d, 99))}


def _domain_usage(traces, doms, nb, n_domains):
    """Pro Domaene: Nutzungs-Histogramm (Anteil Token, die jeden Block nutzen)."""
    counts = np.zeros((n_domains, nb))
    toks = np.zeros(n_domains)
    membership = []   # (N, nb) multi-hot pro Token
    labels = []
    for tr, dom in zip(traces, doms):      # tr (R,bs,T,k), dom (bs,)
        R, B, T, k = tr.shape
        for b in range(B):
            d = int(dom[b])
            if d < 0:
                continue
            for t in range(T):
                blk = np.unique(tr[:, b, t, :])
                mh = np.zeros(nb); mh[blk] = 1.0
                membership.append(mh); labels.append(d)
                counts[d, blk] += 1
                toks[d] += 1
    frac = counts / np.maximum(toks[:, None], 1)   # (n_domains, nb)
    return frac, toks, np.array(membership), np.array(labels)


def _domain_jaccard(frac, cap=8):
    """Jaccard der Top-`cap` genutzten Bloecke je Domaenenpaar (off-diag Mittel)."""
    n = frac.shape[0]
    tops = [set(np.argsort(frac[d])[::-1][:cap].tolist()) for d in range(n)]
    M = np.eye(n)
    vals = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            inter = len(tops[i] & tops[j]); union = len(tops[i] | tops[j])
            M[i, j] = inter / max(1, union)
            if i < j:
                vals.append(M[i, j])
    return M.tolist(), (float(np.mean(vals)) if vals else 0.0)


def _domain_clf(membership, labels, n_domains, device):
    """Einfache multinomiale logistische Regression: Domaene aus Block-Membership.
    Train/Test-Split 50/50, Accuracy vs. Zufall (1/n_domains)."""
    if membership.shape[0] < 4 * n_domains:
        return None, 1.0 / n_domains
    X = torch.tensor(membership, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    perm = torch.randperm(X.shape[0], device=device)
    X, y = X[perm], y[perm]
    n_tr = X.shape[0] // 2
    Xtr, ytr, Xte, yte = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]
    W = torch.zeros(X.shape[1], n_domains, device=device, requires_grad=True)
    b = torch.zeros(n_domains, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xtr @ W + b, ytr) + 1e-3 * (W * W).sum()
        loss.backward(); opt.step()
    with torch.no_grad():
        acc = (((Xte @ W + b).argmax(1)) == yte).float().mean().item()
    return acc, 1.0 / n_domains


def evaluate(model, data, n_batches=8, bs=16, seq_len=128, device="cpu",
             cache_Ks=(4, 8, 12, 16, 32)):
    # Kein globales no_grad: der Modell-Forward ist in _collect bereits no_grad-geschuetzt,
    # aber der Domain-Classifier (_domain_clf) braucht Autograd zum Trainieren.
    nb = model.cfg.n_blocks
    block_params = sum(p.numel() for p in model.bank.blocks[0].parameters())
    n_domains = data.n_domains
    out = {"n_blocks": nb, "block_params": int(block_params), "domains": data.domains}

    # contiguous + shuffled
    cont = _collect(model, data, "contiguous", n_batches, bs, seq_len, device)
    shuf = _collect(model, data, "shuffled", n_batches, bs, seq_len, device)
    traces_c, doms_c, losses_c, R, k = cont
    traces_s, _, _, _, _ = shuf

    out["loss_per_iter"] = [round(x, 4) for x in losses_c]
    out["anytime"] = round(max(losses_c) - min(losses_c), 4)

    ws_c = _working_set(traces_c)
    ws_s = _working_set(traces_s)
    out["working_set"] = {"contiguous": round(ws_c, 3), "shuffled": round(ws_s, 3)}

    streams_c = _streams(traces_c)
    streams_s = _streams(traces_s)
    out["cache_miss"] = {"contiguous": _cache_miss(streams_c, cache_Ks, nb),
                         "shuffled":   _cache_miss(streams_s, cache_Ks, nb)}
    out["reuse_distance"] = {"contiguous": _reuse_distance(streams_c),
                             "shuffled":   _reuse_distance(streams_s)}

    # Bytes/Token (basierend auf contiguous WS)
    out["bytes_per_token"] = {
        "fp16": int(round(ws_c * block_params * 2)),
        "int8": int(round(ws_c * block_params * 1)),
        "note": "WS_mean * block_params * bytes/param",
    }

    # Domaenen-Metriken (contiguous)
    frac, toks, membership, labels = _domain_usage(traces_c, doms_c, nb, n_domains)
    out["active_blocks_per_domain"] = {
        data.domains[d]: int((frac[d] > 0).sum()) for d in range(n_domains)}
    jm, joff = _domain_jaccard(frac)
    out["domain_jaccard_offdiag"] = round(joff, 3)
    out["domain_jaccard_matrix"] = [[round(v, 2) for v in row] for row in jm]
    acc, chance = _domain_clf(membership, labels, n_domains, device)
    out["domain_clf_acc"] = round(acc, 3) if acc is not None else None
    out["domain_clf_chance"] = round(chance, 3)

    # Gini + tote Bloecke (Gesamtnutzung)
    usage = (frac * toks[:, None]).sum(0)
    out["gini"] = round(gini(usage), 4)
    out["dead_blocks"] = int((usage == 0).sum())
    return out
