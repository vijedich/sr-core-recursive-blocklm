"""Offloading-Simulator auf gespeicherten HeteroMini-Modellen (Traces aus dem Modell).

Beantwortet: Wie gross waere der theoretische Speicher-/Transfer-Vorteil im Offloading-
Regime (Bank im langsamen Speicher, kleiner Cache im schnellen)? Misst BEWEGTE BYTES, nicht
FLOPs.

Pro (Modell, Modus contiguous/shuffled, Cache-Kapazitaet K):
  miss@K (LRU)            Misses pro Block-Anforderung
  misses/token           Block-Ladungen pro Token
  bytes/token fp16/int8  misses/token * block_bytes
  stall_bytes/token      LRU ohne Prefetch -> jede Miss stallt (= bytes/token)
  miss@K (Belady)        ideale Eviction (untere Schranke)
  resident_working_set   mittlere eindeutige Bloecke/Token (was resident bleiben muesste)
  reuse p50/p90/p99
  transfer_time/token    bytes/token / Bandbreite (Roofline)

Dense (ModelA) = Layer-Offloading-Baseline: ALLE D Bloecke pro Token.

Wiederverwendbar: nimmt beliebige Checkpoints (--glob oder --checkpoints), Architektur kommt
aus dem Checkpoint selbst (rblm.model_io). KEINE hartkodierte Modell-Liste.

Nutzung:
  python -m experiments.offload_sim --glob "results/hm_cont_*.pt"
  python -m experiments.offload_sim --checkpoints results/hm_cont_hm_srcore_b32_R6_s0.pt ...
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch

from rblm.heteromini import HeteroMiniData
from rblm import model_io
from experiments.heteromini_eval import _collect, _streams, _reuse_distance
from experiments.tinystories_exp import RESULTS


def dense_streams(D, n_seq, T):
    """Layer-Offloading: jede Position fordert alle D Bloecke der Reihe nach."""
    req = [np.array([b]) for b in range(D)]
    return [[r for _ in range(T) for r in req] for _ in range(n_seq)]
    # (n_seq Sequenzen, je T Tokens * D Einzel-Block-Anforderungen)


def _flatten(streams):
    """-> (1D Block-Id-Stream, n_tokens). Ein 'request' ist eine Block-Menge pro Iteration."""
    flat = []
    n_tokens = 0
    for seq in streams:
        # Heuristik: bei Sparse ist len(seq) = T*R; bei Dense = T*D. Token-Zahl ueber
        # die Sequenzlaenge nicht direkt ableitbar -> n_tokens separat gezaehlt vom Aufrufer.
        for req in seq:
            flat.extend(int(b) for b in req)
    return flat


def sim_lru(flat, K):
    from collections import OrderedDict
    cache = OrderedDict()
    miss = 0
    for b in flat:
        if b in cache:
            cache.move_to_end(b)
        else:
            miss += 1
            cache[b] = True
            if len(cache) > K:
                cache.popitem(last=False)
    return miss


def sim_belady(flat, K):
    # naechste Nutzung je Position
    n = len(flat)
    next_use = [n] * n
    last = {}
    for i in range(n - 1, -1, -1):
        b = flat[i]
        next_use[i] = last.get(b, n)
        last[b] = i
    cache = {}                # block -> next_use_position (aktuell)
    miss = 0
    for i, b in enumerate(flat):
        if b in cache:
            cache[b] = next_use[i]
        else:
            miss += 1
            if len(cache) >= K:
                victim = max(cache, key=lambda x: cache[x])  # weiteste Zukunft
                if cache[victim] > next_use[i]:
                    del cache[victim]
                    cache[b] = next_use[i]
                # sonst: b waere selbst der weiteste -> nicht aufnehmen (bypass)
                elif len(cache) < K:
                    cache[b] = next_use[i]
            else:
                cache[b] = next_use[i]
    return miss


def run(checkpoints, bandwidth_gbs=16.0, n_batches=8, bs=16, seq_len=128,
        Ks=(4, 8, 16, 32, 64), device="cpu"):
    data = HeteroMiniData()
    bw = bandwidth_gbs * 1e9  # bytes/s
    rows = []
    dense_ref = {}  # (mode,K) -> bytes/token fp16 fuer relative Vergleich

    for path in checkpoints:
        m, arch, mstep = model_io.load_checkpoint(path, data.vocab_size, device)
        name = model_io.label(arch, mstep)
        variant = "dense" if model_io.is_dense(arch) else "sparse"
        print(f"[OffloadSim] {name}: geladen aus {os.path.basename(path)}", flush=True)
        block_params = sum(p.numel() for p in model_io.blocks_of(m, arch)[0].parameters())
        modes = ["contiguous"] if variant == "dense" else ["contiguous", "shuffled"]
        for mode in modes:
            if variant == "dense":
                D = m.cfg.dense_depth
                n_tokens = n_batches * bs * seq_len
                streams = dense_streams(D, n_batches * bs, seq_len)
                ws_mean = float(D)  # alle D pro Token
            else:
                traces, _, _, R, k = _collect(m, data, mode, n_batches, bs, seq_len, device)
                streams = _streams(traces)
                n_tokens = sum(tr.shape[1] * tr.shape[2] for tr in traces)  # bs*T pro batch
                # WS = mittlere eindeutige Bloecke/Token
                wsl = []
                for tr in traces:
                    Rr, B, T, kk = tr.shape
                    a = tr.transpose(1, 2, 0, 3).reshape(B * T, Rr * kk)
                    wsl.extend(len(np.unique(r)) for r in a)
                ws_mean = float(np.mean(wsl))
            flat = _flatten(streams)
            rd = _reuse_distance(streams)
            for K in Ks:
                miss = sim_lru(flat, K)
                miss_bel = sim_belady(flat, K)
                miss_per_tok = miss / n_tokens
                bpt16 = miss_per_tok * block_params * 2
                bpt8 = miss_per_tok * block_params * 1
                if variant == "dense":
                    dense_ref[(mode, K)] = bpt16
                rows.append({
                    "model": name, "mode": mode, "K": K, "block_params": int(block_params),
                    "miss_at_K_lru": round(miss / max(1, len(flat)), 4),
                    "miss_at_K_belady": round(miss_bel / max(1, len(flat)), 4),
                    "misses_per_token": round(miss_per_tok, 3),
                    "bytes_per_token_fp16": int(round(bpt16)),
                    "bytes_per_token_int8": int(round(bpt8)),
                    "stall_bytes_per_token_fp16": int(round(bpt16)),
                    "resident_working_set": round(ws_mean, 2),
                    "reuse_p50": rd["p50"], "reuse_p90": rd["p90"], "reuse_p99": rd["p99"],
                    "transfer_us_per_token_fp16": round(bpt16 / bw * 1e6, 2),
                })
        print(f"[OffloadSim] {name} fertig", flush=True)

    # relative vs dense (gleiches K, dense nur contiguous -> als Referenz fuer beide Modi)
    for r in rows:
        ref = dense_ref.get(("contiguous", r["K"]))
        r["rel_vs_dense_fp16"] = round(r["bytes_per_token_fp16"] / ref, 3) if ref else None

    out = {"bandwidth_gbs": bandwidth_gbs, "n_tokens_per_cfg": n_batches * bs * seq_len,
           "rows": rows}
    with open(os.path.join(RESULTS, "offload_sim.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Haupttabelle
    print("\n=== OFFLOADING-SIM  (Bandbreite %.0f GB/s) ===" % bandwidth_gbs)
    hdr = f'{"model":22} {"mode":11} {"K":3} {"miss@K":7} {"by/tok16":9} {"by/tok8":8} {"rel.dense":9} {"reuseP90":8} {"us/tok":7}'
    print(hdr); print("-" * len(hdr))
    for r in rows:
        rel = f'{r["rel_vs_dense_fp16"]:.3f}' if r["rel_vs_dense_fp16"] is not None else "-"
        print(f'{r["model"]:22} {r["mode"]:11} {r["K"]:<3} {r["miss_at_K_lru"]:<7.3f} '
              f'{r["bytes_per_token_fp16"]/1024:<9.1f} {r["bytes_per_token_int8"]/1024:<8.1f} '
              f'{rel:9} {r["reuse_p90"]:<8.0f} {r["transfer_us_per_token_fp16"]:<7.2f}')
    print("(by/tok in KB; rel.dense = bytes/token relativ zur Dense-Baseline bei gleichem K)")
    return out


@torch.no_grad()
def measure_ws(model, arch, data, n_batches, bs, seq_len, device):
    """Mittlere eindeutige Bloecke/Token (Working Set) — die regime-stabile Geometrie."""
    if model_io.is_dense(arch):
        return float(model.cfg.dense_depth)
    traces, _, _, R, k = _collect(model, data, "contiguous", n_batches, bs, seq_len, device)
    wsl = []
    for tr in traces:
        Rr, B, T, kk = tr.shape
        a = tr.transpose(1, 2, 0, 3).reshape(B * T, Rr * kk)
        wsl.extend(len(np.unique(r)) for r in a)
    return float(np.mean(wsl))


def project_scale(models, targets_b=(6.0, 13.0), n_blocks_grid=(32, 256, 1024, 8192),
                  bandwidth_gbs=16.0, vram_gb=6.0, dtype_bytes=2):
    """Projiziert die GEMESSENE Geometrie (WS) auf Zielmodellgroessen.

    Annahmen (klar als PROJEKTION markiert): WS ist regime-stabil (hielt ueber 64->256
    Bloecke); Transfer-limitiert (Roofline, Compute ignoriert); sparse laedt pro Token den
    Working Set 'kalt' (ohne Cross-Token-Cache = konservativ); Dense-Layer-Offload streamt
    das GANZE Modell pro Token. Verhaeltnis = n_blocks / WS (blockgroessen-unabhaengig).
    """
    bw = bandwidth_gbs * 1e9
    dt = "fp16" if dtype_bytes == 2 else "int8"
    print(f"\n=== SKALEN-PROJEKTION  (Bandbreite {bandwidth_gbs:.0f} GB/s, {dt}, "
          f"VRAM {vram_gb:.0f} GB) ===")
    print("PROJEKTION auf gemessener WS-Geometrie — Dense-Baseline = Layer-Offloading "
          "(ganzes Modell/Token).")
    for P_b in targets_b:
        P = P_b * 1e9
        dense_bytes = P * dtype_bytes
        dense_toks = bw / dense_bytes
        fits = "JA" if dense_bytes <= vram_gb * 1e9 else f"NEIN ({dense_bytes/1e9:.0f}GB>{vram_gb:.0f})"
        print(f"\n-- Ziel {P_b:.0f}B Parameter --  Dense passt in VRAM? {fits};  "
              f"Dense-Offload: {dense_toks:.1f} Tok/s")
        print(f'  {"model":20} {"WS":4} {"n_blk":6} {"block":9} {"sparse Tok/s":12} {"x vs Dense":10}')
        for name, ws in models:
            for nb in n_blocks_grid:
                block_params = P / nb
                block_bytes = block_params * dtype_bytes
                sparse_bytes = ws * block_bytes          # kalt: WS Bloecke/Token
                sparse_toks = bw / sparse_bytes
                speedup = sparse_toks / dense_toks       # = nb / ws
                bs_str = (f"{block_bytes/1e6:.1f}MB" if block_bytes < 1e9
                          else f"{block_bytes/1e9:.2f}GB")
                print(f'  {name:20} {ws:<4.1f} {nb:<6} {bs_str:9} {sparse_toks:<12.0f} {speedup:<10.0f}')
    print("\nVerhaeltnis sparse/dense = n_blocks/WS (Blockgroesse kuerzt sich raus). "
          "int8 verdoppelt beide Tok/s, Verhaeltnis bleibt. Regime A=wenige grosse Bloecke "
          "(kleine n_blk), B=viele kleine (grosse n_blk).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/hm_cont_*.pt",
                    help="Glob fuer Checkpoints (Default: alle Continuation-Snapshots)")
    ap.add_argument("--checkpoints", nargs="*", default=None,
                    help="explizite Checkpoint-Pfade (ueberschreibt --glob)")
    ap.add_argument("--bandwidth_gbs", type=float, default=16.0, help="RAM->VRAM GB/s (PCIe~16)")
    ap.add_argument("--n_batches", type=int, default=8)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--project", action="store_true",
                    help="Skalen-Projektion (Tok/s sparse vs Dense-Offload bei Zielgroessen)")
    ap.add_argument("--targets_b", type=float, nargs="*", default=[6.0, 13.0],
                    help="Zielmodellgroessen in Mrd. Parametern")
    ap.add_argument("--n_blocks_grid", type=int, nargs="*", default=[32, 256, 1024, 8192],
                    help="Block-Regime (klein=wenige grosse, gross=viele kleine)")
    ap.add_argument("--vram_gb", type=float, default=6.0)
    ap.add_argument("--int8", action="store_true", help="int8 statt fp16 projizieren")
    a = ap.parse_args()
    cks = a.checkpoints or model_io.discover(a.glob)
    if not cks:
        raise SystemExit(f"Keine Checkpoints gefunden (glob={a.glob!r}).")
    if a.project:
        data = HeteroMiniData()
        models = []
        for path in cks:
            m, arch, step = model_io.load_checkpoint(path, data.vocab_size, a.device)
            if model_io.is_dense(arch):
                continue  # Dense ist die Baseline, nicht projiziert
            ws = measure_ws(m, arch, data, a.n_batches, a.bs, a.seq_len, a.device)
            models.append((model_io.label(arch, step), ws))
        project_scale(models, targets_b=a.targets_b, n_blocks_grid=a.n_blocks_grid,
                      bandwidth_gbs=a.bandwidth_gbs, vram_gb=a.vram_gb,
                      dtype_bytes=1 if a.int8 else 2)
    else:
        run(cks, bandwidth_gbs=a.bandwidth_gbs, n_batches=a.n_batches, bs=a.bs,
            seq_len=a.seq_len, device=a.device)
