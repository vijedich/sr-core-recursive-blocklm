"""Kompetenzzentren-Test (Exp4) — Phase-2-Modell auf TinyStories

Prueft: Bildet das Phase-2-Modell ohne Aufgabenlabels spontan verschiedene
funktionale Block-Regionen fuer narrative Aufgabentypen aus?

Kategorisierung: heuristisch aus Textmerkmalen, NIEMALS im Training verwendet.
Labels dienen ausschliesslich der nachtraeglichen Analyse.

Nutzung:
  python -m experiments.competence_centers_exp               # Phase-2 Training + Analyse
  python -m experiments.competence_centers_exp --smoke       # Kurztest (500 Schritte)
  python -m experiments.competence_centers_exp --ckpt PATH   # Checkpoint laden, Training ueberspringen
  python -m experiments.competence_centers_exp --ckpt results/tinystories_b64k4R6_s0_model.pt
"""
from __future__ import annotations
import argparse, json, os, math, time
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from rblm.config import ModelConfig
from rblm.models import build_model, iteration_losses, weighting

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


# =============================================================================
# Heuristische Kategorisierung
# =============================================================================

# Prioritaetsreihenfolge: spezifischere Marker zuerst, allgemeinere spaeter.
# "dialogue" ist in TinyStories sehr haeufig (fast jede Geschichte hat Dialog),
# daher als letzter Filter — damit seltenere Kategorien nicht verdraengt werden.

def _cat(text: str) -> str | None:
    """Gibt erste passende Kategorie oder None (unklassifiziert) zurueck.
    WICHTIG: Kategorien werden NIE ans Modell weitergegeben."""
    t = text.lower()

    # Szenenswechsel: Ortswechsel-Phrasen (spezifisch)
    if any(p in t for p in ["went to", "walked to", "came to", "arrived at",
                             "ran to", "flew to", "moved to", "traveled to"]):
        return "scene_shift"

    # Kausalitaet: explizite Verbindungswoerter
    if any(w in t for w in ["because", "therefore", "that's why", "so that",
                             "in order to", "as a result", "which is why"]):
        return "causality"

    # Zeitfolge: mindestens 2 Zeitwoerter
    tw = sum(1 for w in ["before ", "after ", " then ", " next ",
                          " finally ", " first "] if w in t)
    if tw >= 2:
        return "temporal"

    # Koreferenz: mindestens 3 Pronomen-Vorkommnisse (Figuren-Verfolgung)
    # Normalisiere Satzzeichen damit Wortgrenzen zuverlaessig erkannt werden
    padded = " " + t.replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ") + " "
    pr = sum(padded.count(f" {w} ") for w in ["he", "she", "they", "him", "her", "them"])
    if pr >= 3:
        return "coreference"

    # Emotion: Gefuehlswoerter
    if any(w in t for w in ["happy", "sad", "angry", "scared", "excited",
                             "surprised", "afraid", "worried", "upset",
                             "cried", "tears", "laugh"]):
        return "emotion"

    # Dialog: Anfuehrungszeichen oder Dialog-Verben (haeufigste Kategorie)
    if ('"' in text or '“' in text or '”' in text
            or any(f" {w} " in t for w in ["said", "asked", "replied",
                                            "shouted", "whispered", "told"])):
        return "dialogue"

    return None


CAT_DISPLAY = {
    "scene_shift": "Szene/Ort",
    "causality":   "Kausalitaet",
    "temporal":    "Zeitfolge",
    "coreference": "Koreferenz",
    "emotion":     "Emotion",
    "dialogue":    "Dialog",
}


# =============================================================================
# Daten
# =============================================================================

def load_with_text(vocab: int = 8000, max_docs: int = 20000):
    """Laedt TinyStories als Rohtexte + trainiert Tokenizer."""
    try:
        from datasets import load_dataset
        from tokenizers import ByteLevelBPETokenizer
    except ImportError:
        raise RuntimeError("pip install datasets tokenizers")

    print("[CC] Lade TinyStories ...", flush=True)
    ds = load_dataset("roneneldan/TinyStories", split="train")
    texts = [ds[i]["text"] for i in range(min(max_docs, len(ds)))]
    print(f"[CC] {len(texts)} Geschichten geladen.", flush=True)

    print("[CC] Trainiere Tokenizer ...", flush=True)
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(texts, vocab_size=vocab,
                            special_tokens=["<pad>", "<bos>"])
    print(f"[CC] Vokabular: {tok.get_vocab_size()}", flush=True)
    return texts, tok


def flat_ids(texts, tok) -> torch.Tensor:
    """Flaches Token-Array fuer das Training."""
    ids = []
    for t in texts:
        ids.extend(tok.encode("<bos>" + t).ids)
    return torch.tensor(ids, dtype=torch.long)


def categorized_windows(texts, tok, seq_len: int = 128, stride: int = 64,
                         min_per_cat: int = 100) -> dict[str, list]:
    """
    Baut kategorisierte Token-Fenster.
    Gibt {cat: [tensor(seq_len), ...]} zurueck.
    Kategorisierung: heuristisch ueber Textmerkmale (KEIN Training-Signal).
    """
    print("[CC] Baue kategorisierte Fenster (heuristisch) ...", flush=True)
    cat_wins: dict[str, list] = defaultdict(list)

    for text in texts:
        ids = tok.encode("<bos>" + text).ids
        if len(ids) < seq_len + 1:
            continue
        for start in range(0, len(ids) - seq_len, stride):
            w_ids = ids[start:start + seq_len]
            w_text = tok.decode(w_ids, skip_special_tokens=True)
            c = _cat(w_text)
            if c is not None:
                # Store seq_len+1 tokens: first seq_len as input, last as target shift
                if start + seq_len + 1 <= len(ids):
                    cat_wins[c].append(torch.tensor(
                        ids[start:start + seq_len + 1], dtype=torch.long))

    result = {c: seqs for c, seqs in cat_wins.items() if len(seqs) >= min_per_cat}
    print(f"[CC] Gefundene Kategorien ({len(result)}):", flush=True)
    for c in sorted(result):
        print(f"  {CAT_DISPLAY.get(c, c):16s}: {len(result[c]):5d} Fenster", flush=True)
    return result


class FlatData:
    """Minimale causal-LM Dataset-Klasse fuer Training."""
    def __init__(self, ids: torch.Tensor, vocab_size: int):
        self.ids = ids
        self.vocab_size = vocab_size

    def batch(self, bs: int, seq_len: int, device: str = "cpu", **kw):
        N = self.ids.numel() - 1
        starts = torch.randint(0, N - seq_len, (bs,))
        toks = torch.stack([self.ids[s:s + seq_len] for s in starts]).to(device)
        tgt  = torch.stack([self.ids[s + 1:s + seq_len + 1] for s in starts]).to(device)
        mask = torch.ones_like(toks, dtype=torch.bool)
        return toks, tgt, mask, None


# =============================================================================
# Modell
# =============================================================================

def make_model(vocab_size: int, device: str = "cpu"):
    """Phase-2-Konfiguration: identisch zu tinystories_exp.py."""
    cfg = ModelConfig(
        vocab_size=vocab_size, d_model=256, block_hidden=512,
        n_heads=4, context_layers=1, max_len=256,
        variant="C", n_blocks=64, k_active=4, routed_iters=6,
        key_dim=64, router_noise_std=0.3,
    )
    return build_model(cfg).to(device), cfg


def _lr_schedule(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * p))


def train_phase2(model, data: FlatData, steps: int = 3000, bs: int = 32,
                 seq_len: int = 128, device: str = "cpu", seed: int = 0,
                 eval_every: int = 500):
    """Phase-2-Training: kein Diversity-Zwang, identisch zu tinystories_exp.py."""
    torch.manual_seed(seed)
    R = model.n_iters
    w = weighting(R, "end").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _lr_schedule(s, warmup=200, total=steps))

    model.train()
    t0 = time.time()
    for step in range(steps):
        toks, tgt, mask, _ = data.batch(bs, seq_len, device=device)
        logits, aux = model(toks)
        per = iteration_losses(logits, tgt, mask)
        lb  = sum(a["lb_loss"] for a in aux["iters"]) / R
        loss = (w * per).sum() + 0.01 * lb
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % eval_every == 0 or step == 0:
            elapsed = time.time() - t0
            print(f"  step {step+1:5d}  L1={per[0].item():.3f}"
                  f"  Lfin={per[-1].item():.3f}  {elapsed:.0f}s", flush=True)
            model.train()

    elapsed = time.time() - t0
    print(f"[CC] Training fertig — {elapsed:.0f}s", flush=True)


# =============================================================================
# Routing-Analyse
# =============================================================================

@torch.no_grad()
def collect_routing(model, cat_wins: dict, bs: int = 32,
                    max_windows: int = 1000, device: str = "cpu") -> dict:
    """
    Sammelt P(b | q, r) fuer jede Kategorie q und Iteration r.

    Rueckgabe: {cat: {"freq": ndarray(R, nb), "n_samples": int}}
    Normierung: freq[r].sum() == 1.0 (Wahrscheinlichkeitsverteilung ueber Bloecke)
    """
    model.eval()
    R = model.n_iters
    k = model.cfg.k_active
    nb = model.cfg.n_blocks
    out = {}

    for cat, seqs in sorted(cat_wins.items()):
        sample = seqs[:max_windows]
        freq = np.zeros((R, nb), dtype=np.float64)
        total = np.zeros(R, dtype=np.float64)

        for start in range(0, len(sample), bs):
            batch = sample[start:start + bs]
            # toks = first seq_len tokens; tgt = shift by 1 (stored as seq_len+1)
            toks = torch.stack([s[:-1] for s in batch]).to(device)
            B, T = toks.shape

            _, aux = model(toks)
            for r in range(R):
                topk = aux["iters"][r]["topk_idx"].reshape(-1).cpu().numpy()
                np.add.at(freq[r], topk, 1)
                total[r] += B * T * k

        # Normieren: jede Zeile ist eine Wahrscheinlichkeitsverteilung
        freq /= total[:, None] + 1e-12
        out[cat] = {"freq": freq, "n_samples": len(seqs)}

    return out


@torch.no_grad()
def collect_routing_soft(model, cat_wins: dict, bs: int = 32,
                         max_windows: int = 1000, device: str = "cpu") -> dict:
    """
    Vollstaendige Router-Wahrscheinlichkeiten P(b|q,r) via full_probs.
    Feiner als Top-k-Jaccard — erkennt Unterschiede, die Top-12-Sets verbergen.

    Rueckgabe: {cat: {"soft_freq": ndarray(R, nb), "n_samples": int}}
    """
    model.eval()
    R = model.n_iters
    nb = model.cfg.n_blocks
    out = {}

    for cat, seqs in sorted(cat_wins.items()):
        sample = seqs[:max_windows]
        soft_freq = np.zeros((R, nb), dtype=np.float64)
        n_batches = 0

        for start in range(0, len(sample), bs):
            batch = sample[start:start + bs]
            toks = torch.stack([s[:-1] for s in batch]).to(device)
            _, aux = model(toks)
            for r in range(R):
                fp = aux["iters"][r]["full_probs"].cpu().numpy()  # (B*T, nb)
                soft_freq[r] += fp.mean(axis=0)
            n_batches += 1

        if n_batches > 0:
            soft_freq /= n_batches
        out[cat] = {"soft_freq": soft_freq, "n_samples": len(seqs)}

    return out


def compute_lift_soft(soft_routing: dict) -> np.ndarray:
    """Lift-Variante fuer Soft-Router-Wahrscheinlichkeiten. Fuegt 'soft_lift' hinzu."""
    cats = sorted(soft_routing.keys())
    p_global = np.mean([soft_routing[c]["soft_freq"] for c in cats], axis=0)
    for cat in cats:
        soft_routing[cat]["soft_lift"] = soft_routing[cat]["soft_freq"] / (p_global + 1e-12)
    return p_global


def soft_jaccard_between_cats(soft_routing: dict) -> tuple[dict, list]:
    """
    KL-Divergenz und L1-Distanz zwischen Kategorien-Soft-Verteilungen.
    Feingranularerer Vergleich als Top-k-Jaccard.
    Gibt ({r: ndarray(n_cats, n_cats)}, cats_sorted) zurueck.
    """
    cats = sorted(soft_routing.keys())
    n = len(cats)
    R = soft_routing[cats[0]]["soft_freq"].shape[0]
    l1_matrices = {}
    for r in range(R):
        mat = np.zeros((n, n))
        for i, ci in enumerate(cats):
            pi = soft_routing[ci]["soft_freq"][r] + 1e-12
            for j, cj in enumerate(cats):
                pj = soft_routing[cj]["soft_freq"][r] + 1e-12
                # Symmetrische L1-Distanz zwischen Verteilungen
                mat[i, j] = float(np.abs(pi - pj).sum())
        l1_matrices[r] = mat
    return l1_matrices, cats


def compute_lift(routing: dict) -> np.ndarray:
    """
    Lift = P(b | q, r) / P(b, r) wobei P(b, r) ungewichteter Kategorien-Mittelwert.
    Fuegt 'lift' in-place zu routing[cat] hinzu.
    Gibt p_global (R, nb) zurueck.
    """
    cats = sorted(routing.keys())
    p_global = np.mean([routing[c]["freq"] for c in cats], axis=0)
    for cat in cats:
        routing[cat]["lift"] = routing[cat]["freq"] / (p_global + 1e-12)
    return p_global


def jaccard_between_cats(routing: dict, top_k: int = 12) -> tuple[dict, list]:
    """
    Jaccard-Overlap zwischen Kategorien-Topsets (top_k haeufigste Bloecke).
    Gibt ({r: ndarray(n_cats, n_cats)}, cats_sorted) zurueck.
    """
    cats = sorted(routing.keys())
    n = len(cats)
    R = routing[cats[0]]["freq"].shape[0]
    matrices = {}
    for r in range(R):
        mat = np.zeros((n, n))
        sets = [set(np.argsort(-routing[c]["freq"][r])[:top_k]) for c in cats]
        for i in range(n):
            for j in range(n):
                inter = len(sets[i] & sets[j])
                union = len(sets[i] | sets[j])
                mat[i, j] = inter / max(union, 1)
        matrices[r] = mat
    return matrices, cats


def mutual_information(routing: dict) -> list[float]:
    """
    I(Kategorie; Block) fuer jede Iteration r.
    Approximation: gleichmaessige Kategorienverteilung P(q) = 1/n_cats.
    """
    cats = sorted(routing.keys())
    n_cats = len(cats)
    R = routing[cats[0]]["freq"].shape[0]
    p_q = 1.0 / n_cats
    mi_per_r = []
    for r in range(R):
        # P(b, r) = Mittelwert ueber Kategorien
        p_b = np.mean([routing[c]["freq"][r] for c in cats], axis=0)
        mi = 0.0
        for cat in cats:
            p_bq = routing[cat]["freq"][r]
            joint = p_q * p_bq
            denom = p_q * p_b + 1e-12
            mi += float(np.sum(joint * np.log(joint / denom + 1e-12)))
        mi_per_r.append(mi)
    return mi_per_r


# =============================================================================
# Klassifikator (Naechste-Zentroide, kein sklearn noetig)
# =============================================================================

@torch.no_grad()
def build_clf_features(model, cat_wins: dict, bs: int = 32,
                       max_per_cat: int = 300, device: str = "cpu"):
    """
    Erstellt Sample-Level Routing-Vektoren fuer Klassifikator.
    Feature: Aktivierungsfrequenz pro Block pro Iteration (R * nb Dimensionen).
    """
    model.eval()
    R = model.n_iters
    k = model.cfg.k_active
    nb = model.cfg.n_blocks
    cats = sorted(cat_wins.keys())
    cat_to_idx = {c: i for i, c in enumerate(cats)}

    X, y = [], []
    for cat, seqs in sorted(cat_wins.items()):
        sample = seqs[:max_per_cat]
        for start in range(0, len(sample), bs):
            batch = sample[start:start + bs]
            toks = torch.stack([s[:-1] for s in batch]).to(device)
            B, T = toks.shape
            _, aux = model(toks)

            feat = np.zeros((B, R * nb), dtype=np.float32)
            for r in range(R):
                topk = aux["iters"][r]["topk_idx"]  # (B, T, k)
                for b in range(B):
                    flat = topk[b].reshape(-1).cpu().numpy()
                    cnt = np.bincount(flat, minlength=nb).astype(np.float32)
                    cnt /= max(T * k, 1)
                    feat[b, r * nb:(r + 1) * nb] = cnt
            X.append(feat)
            y.extend([cat_to_idx[cat]] * B)

    return np.vstack(X), np.array(y), cats


def nearest_centroid_accuracy(X: np.ndarray, y: np.ndarray,
                              test_frac: float = 0.3, seed: int = 42) -> float:
    """
    Naechste-Zentroide-Klassifikator (kein sklearn noetig).
    Gibt Genauigkeit auf Testset zurueck.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    n_test = int(n * test_frac)
    idx = rng.permutation(n)
    X_tr, X_te = X[idx[n_test:]], X[idx[:n_test]]
    y_tr, y_te = y[idx[n_test:]], y[idx[:n_test]]

    cats = np.unique(y_tr)
    centroids = np.array([X_tr[y_tr == c].mean(axis=0) for c in cats])

    # Euklidische Distanz zu jedem Zentroiden
    diffs = X_te[:, None, :] - centroids[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    preds = cats[np.argmin(dists, axis=1)]
    return float((preds == y_te).mean())


def bootstrap_clf_accuracy(X: np.ndarray, y: np.ndarray,
                           n_bootstrap: int = 10, test_frac: float = 0.3,
                           seed: int = 42) -> tuple[float, float, float]:
    """
    Balanciertes Bootstrap: alle Kategorien auf min(n_per_cat) begrenzen.
    Verhindert, dass grosse Kategorien (z.B. Dialog) den Klassifikator dominieren.

    Gibt (mean_acc, ci_low_95, ci_high_95) zurueck.
    """
    rng = np.random.default_rng(seed)
    cats = np.unique(y)
    min_per_cat = min(int((y == c).sum()) for c in cats)
    accs = []
    for boot in range(n_bootstrap):
        idx = np.concatenate([
            rng.choice(np.where(y == c)[0], size=min_per_cat, replace=False)
            for c in cats
        ])
        rng.shuffle(idx)
        X_b, y_b = X[idx], y[idx]
        accs.append(nearest_centroid_accuracy(
            X_b, y_b, test_frac=test_frac,
            seed=int(rng.integers(0, 100_000))))
    accs = np.array(accs)
    return float(accs.mean()), float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


# =============================================================================
# Ablationstest
# =============================================================================

@torch.no_grad()
def ablation_test(model, cat_wins: dict, routing: dict,
                  n_top: int = 5, bs: int = 32,
                  max_batches: int = 8, device: str = "cpu"):
    """
    Fuer jede Kategorie q: ablate die n_top Bloecke mit hoechstem Lift.
    Messe Loss-Anstieg pro Kategorie.
    Erwartung: Diagonale dominiert — Ablation von q-Bloecken schadet q am meisten.

    Gibt (delta_matrix, cats, normal_loss) zurueck.
    delta_matrix[i, j] = Loss-Anstieg wenn Kat-i Bloecke ablated, Kat-j evaluiert.
    """
    model.eval()
    nb = model.cfg.n_blocks
    cats = sorted(routing.keys())
    n_cats = len(cats)

    # Baseline-Verlust (kein Ablation) fuer jede Kategorie
    normal_loss: dict[str, float] = {}
    for cat, seqs in sorted(cat_wins.items()):
        tot = 0.0; n = 0
        for start in range(0, min(max_batches * bs, len(seqs)), bs):
            batch = seqs[start:start + bs]
            toks = torch.stack([s[:-1] for s in batch]).to(device)
            tgt  = torch.stack([s[1:]  for s in batch]).to(device)
            mask = torch.ones_like(toks, dtype=torch.bool)
            logits, _ = model(toks)
            per = iteration_losses(logits, tgt, mask)
            tot += per[-1].item(); n += 1
        normal_loss[cat] = tot / max(n, 1)

    # Ablations-Matrix
    delta = np.zeros((n_cats, n_cats))
    for i, ablated_cat in enumerate(cats):
        # Top-Lift-Bloecke der abzulationierenden Kategorie (Mittel ueber Iterationen)
        mean_lift = routing[ablated_cat]["lift"].mean(axis=0)
        top_blocks = np.argsort(-mean_lift)[:n_top]
        abl_mask = torch.zeros(nb, dtype=torch.bool, device=device)
        abl_mask[top_blocks] = True

        for j, eval_cat in enumerate(cats):
            seqs = cat_wins[eval_cat]
            tot = 0.0; n = 0
            for start in range(0, min(max_batches * bs, len(seqs)), bs):
                batch = seqs[start:start + bs]
                toks = torch.stack([s[:-1] for s in batch]).to(device)
                tgt  = torch.stack([s[1:]  for s in batch]).to(device)
                mask = torch.ones_like(toks, dtype=torch.bool)
                logits, _ = model(toks, ablate_mask=abl_mask)
                per = iteration_losses(logits, tgt, mask)
                tot += per[-1].item(); n += 1
            ablated_loss = tot / max(n, 1)
            delta[i, j] = ablated_loss - normal_loss[eval_cat]

    return delta, cats, normal_loss


# =============================================================================
# Gruppenablation mit per-Iterations-Kontrolle (r=1-Kausaltest)
# =============================================================================

@torch.no_grad()
def _per_iter_loss(model, toks: torch.Tensor, tgt: torch.Tensor,
                   masks: list, device: str) -> float:
    """Forward-Pass mit per-Iterations-Ablationsmasken. Gibt finalen Loss zurueck."""
    h0 = model.encode(toks)
    h = h0
    for r in range(model.n_iters):
        m = masks[r] if masks[r] is not None else None
        h, _ = model.bank(h, training=False, ablate_mask=m)
    logits = model.readout(h)
    n = toks.shape[0] * toks.shape[1]
    flat_m = torch.ones(n, dtype=torch.bool, device=device)
    return F.cross_entropy(
        logits.reshape(n, -1)[flat_m],
        tgt.reshape(n)[flat_m],
        reduction="mean",
    ).item()


@torch.no_grad()
def group_r1_ablation_test(model, cat_wins: dict, routing: dict,
                            n_tops: tuple = (5, 10, 15),
                            bs: int = 32, max_batches: int = 8,
                            device: str = "cpu", seed: int = 42):
    """
    Gruppenablation mit 5 Konditionen und Per-Iterations-Kontrolle.

    Konditionen:
      group_all  — Top-N-Lift-Bloecke in ALLEN Iterationen gleichzeitig ablated
      group_r1   — Top-N-Lift-Bloecke NUR in r=1 (0-basiert: erste Iteration)
      group_r2r6 — Top-N-Lift-Bloecke NUR in r=2..R (alle ausser erster Iter.)
      random_all — N zufaellige Bloecke in allen Iterationen (deterministisch)
      usage_all  — Bloecke mit aehnlicher Nutzungsfrequenz aber minimalem Lift

    Gibt (results, cats, normal_loss) zurueck.
      results: {n_top: {kondition: delta_matrix (n_cats x n_cats)}}
      delta_matrix[i, j] = Verlust-Aenderung wenn Kat-i-Bloecke ablated, Kat-j evaluiert
    """
    model.eval()
    rng_seed = np.random.default_rng(seed)
    cats = sorted(routing.keys())
    n_cats = len(cats)
    nb = model.cfg.n_blocks
    R = model.n_iters

    def _bool_mask(blocks):
        m = torch.zeros(nb, dtype=torch.bool, device=device)
        if len(blocks) > 0:
            m[torch.tensor(blocks, dtype=torch.long)] = True
        return m

    def _top_lift(cat, n_top):
        mean_lift = routing[cat]["lift"].mean(axis=0)
        return np.argsort(-mean_lift)[:n_top].tolist()

    # Baseline-Verlust (keine Ablation)
    normal_loss: dict[str, float] = {}
    for cat in cats:
        seqs = cat_wins[cat]
        tot = 0.0; n = 0
        for s in range(0, min(max_batches * bs, len(seqs)), bs):
            batch = seqs[s:s + bs]
            toks = torch.stack([x[:-1] for x in batch]).to(device)
            tgt  = torch.stack([x[1:]  for x in batch]).to(device)
            tot += _per_iter_loss(model, toks, tgt, [None] * R, device)
            n += 1
        normal_loss[cat] = tot / max(n, 1)

    results = {}
    for n_top in n_tops:
        cond_deltas: dict[str, np.ndarray] = {}

        for cond_name in ("group_all", "group_r1", "group_r2r6",
                          "random_all", "usage_all"):
            delta = np.zeros((n_cats, n_cats))

            for i, ablated_cat in enumerate(cats):

                # Bestimme die zu ablierenden Bloecke fuer diese Kategorie + Kondition
                if cond_name == "random_all":
                    blocks = rng_seed.choice(nb, size=n_top, replace=False).tolist()
                elif cond_name == "usage_all":
                    # Bloecke mit aehnlicher Freq wie Top-Lift aber niedrigstem Lift
                    mean_freq = routing[ablated_cat]["freq"].mean(axis=0)
                    mean_lift = routing[ablated_cat]["lift"].mean(axis=0)
                    ref_freq  = mean_freq[_top_lift(ablated_cat, n_top)].mean()
                    closeness = -np.abs(mean_freq - ref_freq) - 0.1 * mean_lift
                    blocks = np.argsort(-closeness)[:n_top].tolist()
                else:
                    blocks = _top_lift(ablated_cat, n_top)

                abl = _bool_mask(blocks)

                # Per-Iterations-Masken je nach Kondition
                if cond_name == "group_r1":
                    masks = [abl] + [None] * (R - 1)
                elif cond_name == "group_r2r6":
                    masks = [None] + [abl] * (R - 1)
                else:  # group_all / random_all / usage_all: alle Iterationen
                    masks = [abl] * R

                # Evalauierung ueber alle Kategorien
                for j, eval_cat in enumerate(cats):
                    seqs = cat_wins[eval_cat]
                    tot = 0.0; n = 0
                    for s in range(0, min(max_batches * bs, len(seqs)), bs):
                        batch = seqs[s:s + bs]
                        toks = torch.stack([x[:-1] for x in batch]).to(device)
                        tgt  = torch.stack([x[1:]  for x in batch]).to(device)
                        tot += _per_iter_loss(model, toks, tgt, masks, device)
                        n += 1
                    delta[i, j] = (tot / max(n, 1)) - normal_loss[eval_cat]

            cond_deltas[cond_name] = delta
        results[n_top] = cond_deltas

    return results, cats, normal_loss


# =============================================================================
# Visualisierung
# =============================================================================

def _cat_labels(cats: list) -> list[str]:
    return [CAT_DISPLAY.get(c, c) for c in cats]


def plot_activation_and_lift(routing: dict, cats: list, tag: str):
    """Zwei Heatmaps: Aktivierungsfrequenz P(b|q,r) und Lift P(b|q,r)/P(b,r)."""
    R = routing[cats[0]]["freq"].shape[0]
    n_cats = len(cats)
    cat_labels = _cat_labels(cats)

    for mode in ("activation", "lift"):
        fig, axes = plt.subplots(1, R, figsize=(max(3 * R, 14), max(n_cats * 0.7 + 1.5, 4)))
        if R == 1:
            axes = [axes]

        for r, ax in enumerate(axes):
            if mode == "activation":
                mat = np.vstack([routing[c]["freq"][r] for c in cats])
                im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0)
            else:
                mat = np.vstack([np.clip(routing[c]["lift"][r], 0.1, 10.0) for c in cats])
                im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                               norm=mcolors.LogNorm(vmin=0.2, vmax=5.0))

            ax.set_title(f"r={r + 1}", fontsize=9)
            ax.set_xlabel("Block-Index", fontsize=8)
            if r == 0:
                ax.set_yticks(range(n_cats))
                ax.set_yticklabels(cat_labels, fontsize=8)
            else:
                ax.set_yticks([])

        plt.colorbar(im, ax=axes[-1],
                     label=("P(b|q,r)" if mode == "activation"
                            else "Lift P(b|q,r)/P(b,r)"))
        title = ("Aktivierungsfrequenz P(b | Kategorie, Iteration)"
                 if mode == "activation"
                 else "Block-Lift pro Kategorie und Iteration")
        plt.suptitle(title, y=1.01, fontsize=10)
        plt.tight_layout()
        fname = os.path.join(RESULTS, f"{tag}_{mode}.png")
        plt.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[CC] Grafik gespeichert: {fname}", flush=True)


def plot_jaccard(jacc_matrices: dict, cats: list, tag: str):
    """Jaccard-Overlap-Matrix zwischen Kategorien pro Iteration."""
    R = len(jacc_matrices)
    n_cats = len(cats)
    cat_labels = _cat_labels(cats)

    fig, axes = plt.subplots(1, R, figsize=(max(2.5 * R, 12), max(n_cats * 0.7 + 1.5, 4)))
    if R == 1:
        axes = [axes]
    im = None
    for r, ax in enumerate(axes):
        im = ax.imshow(jacc_matrices[r], vmin=0, vmax=1, cmap="Blues")
        ax.set_title(f"r={r + 1}", fontsize=9)
        ax.set_xticks(range(n_cats))
        ax.set_xticklabels(cat_labels, rotation=45, fontsize=7, ha="right")
        if r == 0:
            ax.set_yticks(range(n_cats))
            ax.set_yticklabels(cat_labels, fontsize=7)
        else:
            ax.set_yticks([])

    plt.colorbar(im, ax=axes[-1], label="Jaccard (top-12 Bloecke)")
    plt.suptitle("Jaccard-Overlap zwischen Kategorien (wie aehnlich ist das Routing?)",
                 y=1.01, fontsize=10)
    plt.tight_layout()
    fname = os.path.join(RESULTS, f"{tag}_jaccard.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[CC] Grafik gespeichert: {fname}", flush=True)


def plot_summary(mi_per_r: list, clf_results: dict, cats: list, tag: str):
    """Zusammenfassungs-Plot: MI pro Iteration + Klassifikator-Genauigkeit."""
    R = len(mi_per_r)
    chance = 1.0 / len(cats)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Mutual Information
    ax = axes[0]
    ax.plot(range(1, R + 1), mi_per_r, "o-", color="#2563eb")
    ax.set_xlabel("Iteration r"); ax.set_ylabel("MI (nats)")
    ax.set_title("Mutual Information I(Kategorie; Block) pro Iteration")
    ax.grid(alpha=0.3)

    # Klassifikator-Genauigkeit
    ax = axes[1]
    keys = list(clf_results.keys())
    vals = list(clf_results.values())
    colors = ["#b91c1c" if "alle" in k else "#2563eb" for k in keys]
    bars = ax.bar(range(len(keys)), vals, color=colors, alpha=0.8)
    ax.axhline(chance, color="gray", ls=":", label=f"Zufall ({chance:.2f})")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Genauigkeit"); ax.set_ylim(0, 1)
    ax.set_title("Klassifikator: Kategorie aus Routing-Muster?")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fname = os.path.join(RESULTS, f"{tag}_summary.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[CC] Grafik gespeichert: {fname}", flush=True)


def plot_ablation_matrix(delta: np.ndarray, cats: list, normal_loss: dict, tag: str):
    """
    Ablationsmatrix: Zeile = ablationierte Kategorie, Spalte = evaluierte Kategorie.
    Diagonale sollte dominieren wenn echte Kompetenzzentren existieren.
    """
    cat_labels = _cat_labels(cats)
    n = len(cats)
    # Normiert durch Baseline-Verlust (relativer Loss-Anstieg)
    base = np.array([normal_loss[c] for c in cats])
    rel_delta = delta / (base[None, :] + 1e-10)

    fig, ax = plt.subplots(figsize=(max(n * 1.3, 7), max(n * 1.1, 6)))
    im = ax.imshow(rel_delta, cmap="Reds", vmin=0, vmax=rel_delta.max() * 0.9)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"Eval:\n{c}" for c in cat_labels], fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"Ablat:\n{c}" for c in cat_labels], fontsize=8)

    for i in range(n):
        for j in range(n):
            txt = f"{rel_delta[i, j]:.3f}"
            color = "white" if rel_delta[i, j] > rel_delta.max() * 0.6 else "black"
            # Diagonale hervorheben
            weight = "bold" if i == j else "normal"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, color=color, fontweight=weight)

    plt.colorbar(im, label="Relativer Loss-Anstieg (delta/baseline)")
    ax.set_title(
        "Ablationsmatrix: Zeilenkategorie-Bloecke deaktiviert → Spaltenverlust gemessen\n"
        "(Diagonale sollte dominieren wenn Kompetenzzentren real)")
    plt.tight_layout()
    fname = os.path.join(RESULTS, f"{tag}_ablation.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[CC] Grafik gespeichert: {fname}", flush=True)


def plot_group_ablation(group_results: dict, cats: list,
                        normal_loss: dict, tag: str):
    """
    Visualisiert Gruppenablations-Ergebnisse: Diagonale vs. Ausserdiagonale
    fuer alle 5 Konditionen und mehrere n_top-Werte.
    """
    cat_labels = _cat_labels(cats)
    n_cats = len(cats)
    n_tops = sorted(group_results.keys())
    conds = list(next(iter(group_results.values())).keys())
    base = np.array([normal_loss.get(c, 1.0) for c in cats])

    # --- Plot 1: Diagonale vs. Außerdiagonale pro Kondition (fuer n_top=5) ---
    n_top_show = min(5, n_tops[0])
    if n_tops:
        n_top_show = n_tops[0]
    delta_dict = group_results[n_top_show]

    fig, axes = plt.subplots(1, len(conds), figsize=(4 * len(conds), 4), sharey=True)
    if len(conds) == 1:
        axes = [axes]

    for ax, cond in zip(axes, conds):
        delta = delta_dict[cond]
        rel = delta / (base[None, :] + 1e-10)
        diag = np.diag(rel)
        offdiag_mean = np.array(
            [(rel[i, :].sum() - rel[i, i]) / max(n_cats - 1, 1) for i in range(n_cats)])
        x = np.arange(n_cats)
        ax.bar(x - 0.2, diag * 100, width=0.35, color="#b91c1c", alpha=0.8, label="Eigen")
        ax.bar(x + 0.2, offdiag_mean * 100, width=0.35,
               color="#93c5fd", alpha=0.8, label="Andere Ø")
        ax.set_xticks(x)
        ax.set_xticklabels(cat_labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(cond, fontsize=9)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_ylabel("Rel. Loss-Anstieg (%)")
        if ax == axes[0]:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Gruppenablation (n_top={n_top_show}): Eigen- vs. Fremd-Schaden pro Kondition",
                 fontsize=10)
    plt.tight_layout()
    fname = os.path.join(RESULTS, f"{tag}_group_ablation_conds.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[CC] Grafik gespeichert: {fname}", flush=True)

    # --- Plot 2: group_r1 vs group_r2r6 Diagonale fuer alle n_top ---
    if "group_r1" in conds and "group_r2r6" in conds:
        fig, ax = plt.subplots(figsize=(8, 4))
        xs = np.arange(n_cats)
        for idx, n_top in enumerate(n_tops):
            d_r1  = group_results[n_top]["group_r1"]
            d_r26 = group_results[n_top]["group_r2r6"]
            diag_r1  = np.diag(d_r1  / (base[None, :] + 1e-10)) * 100
            diag_r26 = np.diag(d_r26 / (base[None, :] + 1e-10)) * 100
            offset = (idx - len(n_tops) / 2) * 0.25
            ax.bar(xs + offset - 0.12, diag_r1,  width=0.12,
                   color="#b91c1c", alpha=0.5 + 0.15 * idx, label=f"r1 n={n_top}")
            ax.bar(xs + offset + 0.01, diag_r26, width=0.12,
                   color="#2563eb", alpha=0.5 + 0.15 * idx, label=f"r2-6 n={n_top}")
        ax.set_xticks(xs)
        ax.set_xticklabels(cat_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Diag. Rel. Loss-Anstieg (%)")
        ax.set_title("r=1-Kausaltest: Schaden durch Ablation NUR r=1 vs. NUR r=2-6")
        ax.axhline(0, color="gray", lw=0.5)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        fname = os.path.join(RESULTS, f"{tag}_r1_causal.png")
        plt.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[CC] Grafik gespeichert: {fname}", flush=True)


# =============================================================================
# Ausgabe
# =============================================================================

def print_results(routing: dict, jacc_matrices: dict, mi_per_r: list,
                  clf_results: dict, delta: np.ndarray, cats: list,
                  normal_loss: dict, tag: str = ""):
    n_cats = len(cats)
    nb = routing[cats[0]]["freq"].shape[1]
    R  = routing[cats[0]]["freq"].shape[0]
    chance = 1.0 / n_cats

    print("\n" + "=" * 72, flush=True)
    print(f"KOMPETENZZENTREN-ANALYSE -- {tag} (TinyStories)", flush=True)
    print("=" * 72, flush=True)

    print(f"\nKategorien ({n_cats}): {', '.join(cats)}", flush=True)
    print("Stichproben:", flush=True)
    for cat in cats:
        print(f"  {CAT_DISPLAY.get(cat, cat):16s}: {routing[cat]['n_samples']:5d} Fenster",
              flush=True)

    # Top-5 Lift-Bloecke pro Kategorie
    print("\n--- Top-5 Bloecke mit hoechstem Lift (gemittelt ueber Iterationen) ---",
          flush=True)
    for cat in cats:
        mean_lift = routing[cat]["lift"].mean(axis=0)
        top5 = np.argsort(-mean_lift)[:5]
        lifts = mean_lift[top5]
        top5_clean = [int(b) for b in top5]
        print(f"  {CAT_DISPLAY.get(cat, cat):16s}: Bloecke {top5_clean}"
              f"  (Lift: {[f'{l:.2f}' for l in lifts]})", flush=True)

    # Jaccard-Overlap (r=2, die stabilste Iteration nach der Eingangsphase)
    r_show = min(1, R - 1)
    print(f"\n--- Jaccard-Overlap zwischen Kategorien (Iteration r={r_show + 1}) ---",
          flush=True)
    jm = jacc_matrices[r_show]
    header = " " * 18 + "".join(f"{c[:7]:9s}" for c in cats)
    print(header, flush=True)
    for i, ci in enumerate(cats):
        row = f"  {CAT_DISPLAY.get(ci, ci)[:14]:16s}" + "".join(
            f"{jm[i, j]:9.3f}" for j in range(n_cats))
        print(row, flush=True)

    # Jaccard-Interpretation
    diag_jacc = np.diag(jm).mean()
    offdiag_jacc = (jm.sum() - np.trace(jm)) / max(n_cats * (n_cats - 1), 1)
    print(f"\n  Diagonale (Selbst-Overlap): {diag_jacc:.3f}",
          flush=True)
    print(f"  Ausserdiagonale (Kategorie-Kreuzung): {offdiag_jacc:.3f}",
          flush=True)
    if offdiag_jacc < 0.7:
        print("  >> Kategorien zeigen VERSCHIEDENE Block-Praeferenzen [OK]", flush=True)
    else:
        print("  >> Kategorien zeigen AEHNLICHE Block-Praeferenzen (universaler Kern?)",
              flush=True)

    # Mutual Information
    print("\n--- Mutual Information I(Kategorie; Block) pro Iteration ---", flush=True)
    for r, mi in enumerate(mi_per_r):
        print(f"  r={r + 1}: {mi:.4f} nats", flush=True)
    if len(mi_per_r) > 1:
        trend = "steigt" if mi_per_r[-1] > mi_per_r[0] else "faellt"
        print(f"  Trend r=1 -> r={R}: {mi_per_r[0]:.4f} -> {mi_per_r[-1]:.4f} ({trend})",
              flush=True)

    # Klassifikator
    print("\n--- Klassifikator (Naechste-Zentroide auf Routing-Muster) ---", flush=True)
    print(f"  Zufalls-Baseline: {chance:.3f}", flush=True)
    for desc, acc in clf_results.items():
        diff = acc - chance
        sign = "+" if diff >= 0 else ""
        print(f"  {desc:25s}: {acc:.3f}  ({sign}{diff:.3f} ggue. Zufall)", flush=True)

    # Ablationsdiagonale
    print("\n--- Ablationstest: Selbstschaden vs. Fremdschaden ---", flush=True)
    for i, cat in enumerate(cats):
        diag = delta[i, i]
        offdiag = (delta[i].sum() - diag) / max(n_cats - 1, 1)
        marker = "[OK] EIGEN > ANDERE" if diag > offdiag else "[--] ANDERE >= EIGEN"
        print(f"  Ablate {CAT_DISPLAY.get(cat, cat):16s}: "
              f"eigen={diag:+.4f}  andere={offdiag:+.4f}  {marker}", flush=True)

    # Normale Verluste
    print("\n--- Baseline-Verlust pro Kategorie ---", flush=True)
    for cat in cats:
        print(f"  {CAT_DISPLAY.get(cat, cat):16s}: {normal_loss[cat]:.4f} Nats",
              flush=True)

    # Gesamtinterpretation
    print("\n--- Interpretation ---", flush=True)
    n_eigen_gt = sum(1 for i in range(n_cats)
                     if delta[i, i] > (delta[i].sum() - delta[i, i]) / max(n_cats - 1, 1))
    best_clf = max(clf_results.values())

    if offdiag_jacc < 0.65 and best_clf > chance + 0.1:
        print("  ERGEBNIS B wahrscheinlich: Gemeinsamer Kern + versetzte Satelliten", flush=True)
        print("  Kategorien teilen Hub-Bloecke aber haben aufgaben-spezifische Regionen.",
              flush=True)
    elif offdiag_jacc >= 0.8:
        print("  ERGEBNIS A wahrscheinlich: Universaler Kern", flush=True)
        print("  Alle Kategorien nutzen dieselben Bloecke — kein messbarer Unterschied.",
              flush=True)
    else:
        print(f"  ERGEBNIS GEMISCHT: Jaccard-Overlap={offdiag_jacc:.3f}, "
              f"Klassifikator={best_clf:.3f}", flush=True)

    print(f"\n  Ablation-Diagonale: {n_eigen_gt}/{n_cats} Kategorien mit Selbstschaden > Fremdschaden",
          flush=True)
    print(f"  Bester Klassifikator: {best_clf:.3f} (Zufall: {chance:.3f})",
          flush=True)


def print_group_ablation(group_results: dict, cats: list, normal_loss: dict):
    """Gibt Gruppenablations-Ergebnisse lesbar aus."""
    cat_labels = _cat_labels(cats)
    n_cats = len(cats)
    base = np.array([normal_loss.get(c, 1.0) for c in cats])
    chance = 1.0 / n_cats

    print("\n" + "=" * 72, flush=True)
    print("GRUPPENABLATION + r=1-KAUSALTEST", flush=True)
    print("=" * 72, flush=True)

    for n_top, cond_dict in sorted(group_results.items()):
        print(f"\n--- n_top={n_top} ---", flush=True)
        for cond, delta in cond_dict.items():
            rel = delta / (base[None, :] + 1e-10)
            diag = np.diag(rel)
            offdiag = np.array(
                [(rel[i, :].sum() - rel[i, i]) / max(n_cats - 1, 1)
                 for i in range(n_cats)])
            n_dom = int((diag > offdiag).sum())
            mean_diag    = float(diag.mean() * 100)
            mean_offdiag = float(offdiag.mean() * 100)
            marker = "[OK]" if n_dom > n_cats // 2 else "[--]"
            print(f"  {cond:12s}: Diag={mean_diag:+.3f}%  Ausserdiag={mean_offdiag:+.3f}%"
                  f"  Dominanz={n_dom}/{n_cats}  {marker}", flush=True)

    # Schlussfolgerung r=1
    if 5 in group_results:
        d5 = group_results[5]
        if "group_r1" in d5 and "group_r2r6" in d5:
            rel_r1  = np.diag(d5["group_r1"]   / (base[None, :] + 1e-10)).mean() * 100
            rel_r26 = np.diag(d5["group_r2r6"] / (base[None, :] + 1e-10)).mean() * 100
            rel_all = np.diag(d5["group_all"]  / (base[None, :] + 1e-10)).mean() * 100
            print(f"\nr=1-Kausaltest (Diagonale-Mittel n_top=5):", flush=True)
            print(f"  group_r1   : {rel_r1:+.3f}%  (Schaden durch NUR r=1-Ablation)", flush=True)
            print(f"  group_r2r6 : {rel_r26:+.3f}%  (Schaden durch NUR r=2-6-Ablation)", flush=True)
            print(f"  group_all  : {rel_all:+.3f}%  (Schaden durch ALLE Iter. Ablation)", flush=True)
            frac_r1 = rel_r1 / max(abs(rel_all), 1e-6)
            if frac_r1 > 0.5:
                print(f"  >> r=1 traegt {frac_r1*100:.0f}% des Gesamtschadens — "
                      f"KAUSALER Adressierungsmechanismus [OK]", flush=True)
            else:
                print(f"  >> r=1 traegt {frac_r1*100:.0f}% des Gesamtschadens — "
                      f"Kern r=2-6 dominiert", flush=True)


# =============================================================================
# Hauptfunktion
# =============================================================================

def _load_checkpoint(model, ckpt_path: str, device: str):
    """Laedt Checkpoint aus .pt-Datei oder versioniertem Verzeichnis."""
    import os
    if ckpt_path.endswith(".pt") and os.path.isfile(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f"[CC] Checkpoint geladen: {ckpt_path}", flush=True)
    elif os.path.isdir(ckpt_path):
        pt_path = os.path.join(ckpt_path, "model.pt")
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"model.pt nicht gefunden in: {ckpt_path}")
        # SHA256-Pruefung wenn vorhanden
        sha_path = os.path.join(ckpt_path, "sha256.txt")
        if os.path.exists(sha_path):
            import hashlib
            h = hashlib.sha256()
            with open(pt_path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            with open(sha_path, encoding="utf-8") as f:
                expected = f.read().strip()
            if h.hexdigest() != expected:
                raise RuntimeError(f"SHA256-Mismatch fuer {pt_path}!")
            print(f"[CC] SHA256 OK: {h.hexdigest()[:16]}...", flush=True)
        model.load_state_dict(torch.load(pt_path, map_location=device, weights_only=True))
        # Metadaten anzeigen
        meta_path = os.path.join(ckpt_path, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            print(f"[CC] Checkpoint geladen: {meta.get('experiment','?')} "
                  f"seed={meta.get('seed','?')} step={meta.get('step','?')} "
                  f"({meta.get('timestamp','?')[:10]})", flush=True)
        else:
            print(f"[CC] Checkpoint geladen: {ckpt_path}", flush=True)
    else:
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {ckpt_path}")


def run(steps: int = 3000, device: str = "cuda", seed: int = 0,
        ckpt: str | None = None, smoke: bool = False,
        vocab: int = 8000, max_docs: int = 20000,
        min_per_cat: int = 100, max_clf: int = 300,
        analysis: bool = False):

    actual_steps = 500 if smoke else steps
    if analysis and ckpt:
        import os as _os
        _ckpt_exp = _os.path.basename(_os.path.dirname(_os.path.dirname(ckpt)))
        analysis_suffix = f"_analysis_{_ckpt_exp}"
        # Seed aus Checkpoint-Pfad-Struktur lesen (.../seed_N/step_N)
        _seed_dir = _os.path.basename(_os.path.dirname(ckpt))
        if _seed_dir.startswith("seed_"):
            try:
                seed = int(_seed_dir.split("_")[1])
            except (IndexError, ValueError):
                pass
    elif analysis:
        analysis_suffix = "_analysis"
    else:
        analysis_suffix = ""
    tag = f"competence_b64k4R6_s{seed}" + ("_smoke" if smoke else "") + analysis_suffix

    # --- Daten ---
    texts, tok = load_with_text(vocab=vocab, max_docs=max_docs)
    cat_wins = categorized_windows(
        texts, tok, seq_len=128, stride=64,
        min_per_cat=min_per_cat if not smoke else 50)

    if len(cat_wins) < 2:
        raise RuntimeError(
            f"Zu wenige Kategorien gefunden: {list(cat_wins.keys())}. "
            "Mehr Daten oder kleineres min_per_cat versuchen.")

    cats = sorted(cat_wins.keys())

    # --- Modell ---
    ids = flat_ids(texts, tok)
    data = FlatData(ids, tok.get_vocab_size())
    model, cfg = make_model(data.vocab_size, device=device)

    auto_ckpt = os.path.join(RESULTS, f"{tag}_model.pt")
    phase2_ckpt = os.path.join(RESULTS, "tinystories_b64k4R6_s0_model.pt")

    trained_in_this_run = False
    if ckpt and os.path.exists(ckpt):
        _load_checkpoint(model, ckpt, device)
    elif os.path.exists(auto_ckpt):
        print(f"[CC] Vorhandener Checkpoint: {auto_ckpt}", flush=True)
        model.load_state_dict(torch.load(auto_ckpt, map_location=device, weights_only=True))
    elif os.path.exists(phase2_ckpt):
        print(f"[CC] Phase-2-Checkpoint aus tinystories_exp: {phase2_ckpt}", flush=True)
        print("[CC] HINWEIS: Tokenizer koennte abweichen — Analyse als Naehrung.",
              flush=True)
        model.load_state_dict(torch.load(phase2_ckpt, map_location=device, weights_only=True))
    else:
        if analysis:
            raise RuntimeError(
                "--analysis gesetzt aber kein Checkpoint gefunden. "
                "Bitte --ckpt PATH angeben.")
        print(f"[CC] Kein Checkpoint gefunden — trainiere Phase-2 ({actual_steps} Schritte).",
              flush=True)
        train_phase2(model, data, steps=actual_steps, device=device, seed=seed,
                     eval_every=50 if smoke else 500)
        os.makedirs(RESULTS, exist_ok=True)
        torch.save(model.state_dict(), auto_ckpt)
        print(f"[CC] Checkpoint gespeichert: {auto_ckpt}", flush=True)
        trained_in_this_run = True

    # --- Routing-Analyse ---
    max_win = 200 if smoke else 1000
    print(f"\n[CC] Sammle Routing-Daten (max {max_win} Fenster/Kat) ...", flush=True)
    routing = collect_routing(model, cat_wins, bs=32, max_windows=max_win, device=device)
    p_global = compute_lift(routing)

    # --- Metriken ---
    print("[CC] Berechne Metriken ...", flush=True)
    jacc_matrices, cats_ordered = jaccard_between_cats(routing, top_k=12)
    mi_per_r = mutual_information(routing)

    # Klassifikator-Features
    print("[CC] Baue Klassifikator-Features ...", flush=True)
    max_clf_actual = 100 if smoke else max_clf
    X, y, cat_names = build_clf_features(
        model, cat_wins, bs=32, max_per_cat=max_clf_actual, device=device)

    nb = model.cfg.n_blocks
    R  = model.n_iters
    clf_results: dict[str, float] = {}

    # Alle Iterationen zusammen
    clf_results["alle Iter (R*nb)"] = nearest_centroid_accuracy(X, y, seed=seed)
    # Per Iteration
    for r in range(R):
        X_r = X[:, r * nb:(r + 1) * nb]
        clf_results[f"r={r + 1}"] = nearest_centroid_accuracy(X_r, y, seed=seed)

    # Balanciertes Bootstrap (alle Kats gleich gross, 95%-KI)
    print("[CC] Balanciertes Bootstrap ...", flush=True)
    n_boot = 5 if smoke else 10
    boot_mean, boot_lo, boot_hi = bootstrap_clf_accuracy(X, y, n_bootstrap=n_boot, seed=seed)
    clf_results["bootstrap_mean"] = boot_mean
    clf_results["bootstrap_ci_lo"] = boot_lo
    clf_results["bootstrap_ci_hi"] = boot_hi

    # Soft-Router-Gewichte
    print("[CC] Soft-Router-Analyse ...", flush=True)
    soft_routing = collect_routing_soft(
        model, cat_wins, bs=32, max_windows=200 if smoke else 500, device=device)
    compute_lift_soft(soft_routing)
    soft_l1_mats, _ = soft_jaccard_between_cats(soft_routing)

    # Ablationstest (Einzelblock, Originaltest)
    print("[CC] Einzelblock-Ablationstest ...", flush=True)
    n_abl = 4 if smoke else 10
    delta, cats_abl, normal_loss = ablation_test(
        model, cat_wins, routing, n_top=5, bs=32, max_batches=n_abl, device=device)

    # Gruppenablation + r=1-Kausaltest
    print("[CC] Gruppenablation + r=1-Kausaltest ...", flush=True)
    n_tops_grp = (5,) if smoke else (5, 10, 15)
    group_results, _grp_cats, grp_normal_loss = group_r1_ablation_test(
        model, cat_wins, routing,
        n_tops=n_tops_grp, bs=32,
        max_batches=4 if smoke else 8, device=device, seed=seed)

    # --- Grafiken ---
    print("[CC] Erstelle Grafiken ...", flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    plot_activation_and_lift(routing, cats_ordered, tag)
    plot_jaccard(jacc_matrices, cats_ordered, tag)
    plot_summary(mi_per_r, clf_results, cats_ordered, tag)
    plot_ablation_matrix(delta, cats_abl, normal_loss, tag)
    plot_group_ablation(group_results, cats_abl, normal_loss, tag)

    # r=1-Kausal-Score aus Gruppenablation berechnen
    r1_causal = None
    if 5 in group_results:
        _d5 = group_results[5]
        if all(k in _d5 for k in ("group_r1", "group_r2r6", "group_all")):
            _gc = sorted(grp_normal_loss)
            _gb = np.array([grp_normal_loss[c] for c in _gc])
            _rel_r1  = float(np.diag(_d5["group_r1"]   / (_gb[None, :] + 1e-10)).mean() * 100)
            _rel_r26 = float(np.diag(_d5["group_r2r6"] / (_gb[None, :] + 1e-10)).mean() * 100)
            _rel_all = float(np.diag(_d5["group_all"]  / (_gb[None, :] + 1e-10)).mean() * 100)
            _frac    = _rel_r1 / max(abs(_rel_all), 1e-6)
            r1_causal = {
                "group_r1_pct":   round(_rel_r1,  4),
                "group_r2r6_pct": round(_rel_r26, 4),
                "group_all_pct":  round(_rel_all, 4),
                "frac_r1":        round(_frac, 4),
                "causal":         bool(_frac > 0.5),
            }

    # --- Speichern (vor print_results, damit JSON immer gespeichert wird) ---
    result = {
        "tag": tag,
        "categories": cats_ordered,
        "n_samples": {c: routing[c]["n_samples"] for c in cats_ordered},
        "freq": {c: routing[c]["freq"].tolist() for c in cats_ordered},
        "lift": {c: routing[c]["lift"].tolist() for c in cats_ordered},
        "p_global": p_global.tolist(),
        "jaccard_r2": jacc_matrices[min(1, R - 1)].tolist(),
        "soft_l1_r2": soft_l1_mats[min(1, R - 1)].tolist(),
        "mi_per_iter": mi_per_r,
        "clf_accuracy": clf_results,
        "bootstrap": {"mean": boot_mean, "ci_lo": boot_lo, "ci_hi": boot_hi},
        "ablation_delta": delta.tolist(),
        "ablation_cats": cats_abl,
        "normal_loss": normal_loss,
        "group_ablation": {
            str(n_top): {cond: mat.tolist() for cond, mat in cond_dict.items()}
            for n_top, cond_dict in group_results.items()
        },
        "r1_causal": r1_causal,
    }
    with open(os.path.join(RESULTS, f"{tag}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[CC] Ergebnisse gespeichert: results/{tag}.json", flush=True)

    # Versionierter Checkpoint (nur wenn Modell in diesem Lauf trainiert wurde)
    if trained_in_this_run:
        try:
            import rblm.checkpoint as _ckpt
            _ckpt.save(
                model       = model,
                experiment  = f"competence_exp4_phase2{'_smoke' if smoke else ''}",
                config      = {
                    "model": {
                        "vocab_size":       data.vocab_size,
                        "d_model":          cfg.d_model,
                        "block_hidden":     cfg.block_hidden,
                        "n_heads":          cfg.n_heads,
                        "context_layers":   cfg.context_layers,
                        "max_len":          cfg.max_len,
                        "n_blocks":         cfg.n_blocks,
                        "k_active":         cfg.k_active,
                        "routed_iters":     cfg.routed_iters,
                        "key_dim":          cfg.key_dim,
                        "router_noise_std": cfg.router_noise_std,
                    },
                    "training": {
                        "steps":          actual_steps,
                        "bs":             32,
                        "seq_len":        128,
                        "lr":             0.002,
                        "weight_decay":   0.01,
                        "warmup":         200,
                        "diverse":        False,
                        "loss_weighting": "end",
                        "lb_loss_weight": 0.01,
                    },
                    "data": {
                        "dataset":    "tinystories",
                        "max_docs":   max_docs,
                        "vocab_size": vocab,
                        "seq_len":    128,
                    },
                },
                metrics     = {
                    "training_steps":     actual_steps,
                    "clf_accuracy_best":  max(clf_results.values()),
                    "clf_results":        clf_results,
                    "mi_per_iter":        mi_per_r,
                    "categories":         cats_ordered,
                    "n_samples":          result["n_samples"],
                },
                routing_stats = {
                    "jaccard_r2":     result["jaccard_r2"],
                    "mi_per_iter":    mi_per_r,
                    "clf_accuracy":   clf_results,
                    "normal_loss":    normal_loss,
                },
                seed      = seed,
                step      = actual_steps,
                val_loss  = None,
            )
        except Exception as _e:
            print(f"[CKPT] Warnung: Checkpoint-Speicherung fehlgeschlagen: {_e}", flush=True)

    # --- Ausgabe ---
    try:
        print_results(routing, jacc_matrices, mi_per_r, clf_results,
                      delta, cats_abl, normal_loss, tag=tag)
    except Exception as _pe:
        print(f"[CC] print_results Fehler (JSON bereits gespeichert): {_pe}", flush=True)
    try:
        print_group_ablation(group_results, cats_abl, normal_loss)
    except Exception as _pe:
        print(f"[CC] print_group_ablation Fehler: {_pe}", flush=True)

    # Balanciertes Bootstrap-Ergebnis ausgeben
    print(f"\n--- Balanciertes Bootstrap (10 Draws, balanciert) ---", flush=True)
    print(f"  Accuracy: {boot_mean:.3f}  95%-KI: [{boot_lo:.3f}, {boot_hi:.3f}]",
          flush=True)
    print(f"  Zufall: {1.0 / len(cats_abl):.3f}", flush=True)
    if r1_causal:
        print(f"\n--- r=1-Kausal-Score (n_top=5) ---", flush=True)
        print(f"  group_r1  : {r1_causal['group_r1_pct']:+.3f}%", flush=True)
        print(f"  group_r2r6: {r1_causal['group_r2r6_pct']:+.3f}%", flush=True)
        print(f"  group_all : {r1_causal['group_all_pct']:+.3f}%", flush=True)
        verdict = "KAUSALER Adressierungsmechanismus [OK]" if r1_causal["causal"] \
                  else "Kern r=2-6 dominiert [--]"
        print(f"  frac_r1   : {r1_causal['frac_r1']:.3f}  -> {verdict}", flush=True)

    print("[CC] Fertig.", flush=True)
    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kompetenzzentren-Test auf Phase-2-Modell (TinyStories)")
    parser.add_argument("--steps", type=int, default=3000,
                        help="Trainingsschritte wenn kein Checkpoint vorhanden (Standard: 3000)")
    parser.add_argument("--device", default="cuda",
                        help="Geraet: cuda oder cpu (Standard: cuda)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", default=None,
                        help="Checkpoint-Verzeichnis oder .pt-Datei (Verzeichnis empfohlen)")
    parser.add_argument("--smoke", action="store_true",
                        help="Kurztest: 500 Trainingsschritte, weniger Daten")
    parser.add_argument("--vocab", type=int, default=8000)
    parser.add_argument("--max_docs", type=int, default=20000)
    parser.add_argument("--min_per_cat", type=int, default=100,
                        help="Mindest-Fensteranzahl pro Kategorie")
    parser.add_argument("--max_clf", type=int, default=300,
                        help="Max. Samples pro Kategorie fuer Klassifikator")
    parser.add_argument("--analysis", action="store_true",
                        help="Analyse-Modus: Checkpoint laden, kein Training. "
                             "Fuehrt Gruppenablation + Soft-Router + Bootstrap durch.")
    args = parser.parse_args()
    run(**vars(args))
