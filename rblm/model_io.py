"""Zentrales, wiederverwendbares Laden beliebiger trainierter Modelle.

Damit Benchmarks NICHT pro Modell hartkodiert werden muessen: jeder Benchmark nimmt
Checkpoint-Pfade (oder ein Glob) und ruft load_checkpoint(). Die Architektur (dense vs
sparse, n_blocks, k, R, core_mode, depth) wird aus dem Checkpoint SELBST gelesen.

Unterstuetzte Formate:
  - Continuation-Snapshots (hm_cont_*.pt):     {"model", "config", "step"}
  - Selbstbeschreibende Checkpoints (neu):     {"state_dict", "arch", "meta"}
  - Roh-state_dict + explizites arch:          ueber load_state(path, arch=...)
"""
from __future__ import annotations
import glob as _glob
import os
import torch

from .config import ModelConfig
from .models import build_model

# Backbone-Defaults des Projekts (gelten fuer alle bisherigen Modelle)
DEFAULTS = dict(d_model=256, block_hidden=512, n_heads=4, context_layers=1,
                max_len=256, key_dim=64, router_noise_std=0.3, coord_dim=3)


def _norm_kind(v):
    """'dense'/'A' -> 'dense';  'sparse'/'routed'/'C' -> 'sparse'."""
    return "dense" if str(v).lower() in ("dense", "a") else "sparse"


def canonical_arch(src, vocab_size=8000):
    """Beliebige Config (cont-'config' oder arch) -> kanonisches arch-Dict."""
    kind = _norm_kind(src.get("kind", src.get("variant", "sparse")))
    a = dict(DEFAULTS)
    a.update(kind=kind,
             vocab_size=src.get("vocab_size", vocab_size),
             n_blocks=src.get("n_blocks", 64),
             k=src.get("k", src.get("k_active", 4)),
             R=src.get("R", src.get("routed_iters", 6)),
             dense_depth=src.get("dense_depth", src.get("depth", 24)),
             core_mode=src.get("core_mode"))
    # explizit gesetzte Backbone-Werte uebernehmen
    for key in DEFAULTS:
        if key in src:
            a[key] = src[key]
    return a


def build_from_arch(arch, device="cpu"):
    if arch["kind"] == "dense":
        cfg = ModelConfig(vocab_size=arch["vocab_size"], d_model=arch["d_model"],
                          block_hidden=arch["block_hidden"], n_heads=arch["n_heads"],
                          context_layers=arch["context_layers"], max_len=arch["max_len"],
                          variant="A", dense_depth=arch["dense_depth"])
    else:
        cfg = ModelConfig(vocab_size=arch["vocab_size"], d_model=arch["d_model"],
                          block_hidden=arch["block_hidden"], n_heads=arch["n_heads"],
                          context_layers=arch["context_layers"], max_len=arch["max_len"],
                          variant="C", n_blocks=arch["n_blocks"], k_active=arch["k"],
                          routed_iters=arch["R"], key_dim=arch["key_dim"],
                          router_noise_std=arch["router_noise_std"],
                          coord_dim=arch["coord_dim"], core_mode=arch["core_mode"])
    return build_model(cfg).to(device), cfg


def label(arch, step=None):
    """Lesbarer, stabiler Modellname aus der Architektur (nicht aus dem Dateinamen)."""
    if arch["kind"] == "dense":
        base = f"dense_d{arch['dense_depth']}"
    else:
        tag = {None: "naked", "per_token": "srcore", "core_satellite": "srsat"}.get(
            arch["core_mode"], str(arch["core_mode"]))
        k = arch.get("k", 4)
        k_tag = f"_k{k}" if k != 4 else ""
        base = f"{tag}_b{arch['n_blocks']}{k_tag}_R{arch['R']}"
    return base + (f"@{step}" if step else "")


def load_checkpoint(path, vocab_size=8000, device="cpu"):
    """-> (model, arch, step). Architektur kommt aus dem Checkpoint selbst."""
    ck = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ck, dict) and "arch" in ck:                      # self-describing
        arch = canonical_arch(ck["arch"], vocab_size)
        state = ck.get("state_dict", ck.get("model"))
        step = (ck.get("meta") or {}).get("step")
    elif isinstance(ck, dict) and "config" in ck and "model" in ck:  # cont-Snapshot
        arch = canonical_arch(ck["config"], vocab_size)
        state = ck["model"]
        step = ck.get("step")
    else:
        raise ValueError(
            f"{path}: kein arch/config im Checkpoint. Roh-state_dict -> load_state(path, arch=...) "
            "nutzen oder im self-describing Format speichern.")
    model, cfg = build_from_arch(arch, device)
    model.load_state_dict(state)
    model.eval()
    return model, arch, step


def load_state(path, arch, vocab_size=8000, device="cpu"):
    """Roh-state_dict laden, Architektur explizit angegeben (arch-Dict)."""
    arch = canonical_arch(arch, vocab_size)
    model, cfg = build_from_arch(arch, device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model, arch, None


def save_checkpoint(path, model, arch, meta=None):
    """Self-describing Checkpoint: laedt spaeter ohne externes Wissen ueber die Architektur."""
    torch.save({"state_dict": model.state_dict(),
                "arch": canonical_arch(arch, arch.get("vocab_size", 8000)),
                "meta": meta or {}}, path)


def discover(pattern):
    """Checkpoint-Pfade per Glob finden, z.B. 'results/hm_cont_*.pt'."""
    return sorted(_glob.glob(pattern))


def blocks_of(model, arch):
    """ModuleList der Bloecke — dense: model.blocks, sparse: model.bank.blocks."""
    return model.blocks if arch["kind"] == "dense" else model.bank.blocks


def is_dense(arch):
    return arch["kind"] == "dense"
