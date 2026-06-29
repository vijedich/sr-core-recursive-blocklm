"""Realer RAM->VRAM Streaming-Prototyp — Engine-Geruest (Stage 0).

Design: docs/streaming_prototype.md

Stage 0 (DIESE Datei, CPU-testbar): BlockStore + LRU-Block-Cache + Routing-Replay.
  Korrektheits-Anker: die Fetch-/Byte-Zahlen MUESSEN die LRU-Miss-Zahlen aus
  experiments/offload_sim.py reproduzieren (gleiche Traces, gleicher Cache K).
  Damit ist bewiesen, dass die Engine das Routing faithful abspielt — bevor echte
  CUDA-Transfers drankommen.

Stage 1/2 (GPU, kuehler Abend): BlockStore.fetch() bekommt einen echten
  copy_(non_blocking=True) aus pinned CPU-Speicher in einen VRAM-Slot; dann
  CUDA-Stream-Overlap (Prefetch naechster Token waehrend Compute aktueller).
  Die Haken dafuer sind unten als TODO(stage1)/TODO(stage2) markiert.

Nutzung (Stage-0-Validierung, CPU):
  python -m experiments.streaming_prototype --checkpoint results/hm_cont_hm_srcore_b64_k8_R6_asw_s0.pt --K 16
"""
from __future__ import annotations
import argparse, os
from collections import OrderedDict

from rblm.heteromini import HeteroMiniData
from rblm import model_io
from experiments.offload_sim import _collect, _streams, _flatten, sim_lru
from experiments.tinystories_exp import RESULTS


# ----------------------------------------------------------------------------- Engine

class BlockStore:
    """Haelt die Blockgewichte im 'langsamen' Speicher (CPU). Stage 0 zaehlt nur Fetches;
    Stage 1 fuegt echten pinned->VRAM copy_(non_blocking=True) hinzu."""

    def __init__(self, blocks, dtype_bytes=2):
        self.n = len(blocks)
        self.block_params = sum(p.numel() for p in blocks[0].parameters())
        self.block_bytes = self.block_params * dtype_bytes
        self.fetch_count = 0
        # TODO(stage1): self.cpu_w = [ {name: p.detach().half().pin_memory()} for blk in blocks ]
        #               self.vram_slots = pre-allokierte VRAM-Buffer (K Stueck)

    def fetch(self, block_id):
        """Stage 0: registriert einen Transfer. Stage 1: echte async H2D-Kopie in freien Slot."""
        self.fetch_count += 1
        # TODO(stage1): slot = cache.free_slot(); copy_stream: vram_slots[slot].copy_(
        #               cpu_w[block_id], non_blocking=True); record event -> compute wartet drauf
        return block_id


class LRUBlockCache:
    """K VRAM-Slots, LRU-Eviction. request() == was der Token an Bloecken braucht."""

    def __init__(self, K, store: BlockStore):
        self.K = K
        self.store = store
        self.resident = OrderedDict()  # block_id -> True (Stage 1: -> slot index)

    def request(self, block_ids):
        """Stellt sicher, dass block_ids resident sind. Misses -> store.fetch()."""
        for b in block_ids:
            if b in self.resident:
                self.resident.move_to_end(b)
            else:
                self.store.fetch(b)
                self.resident[b] = True
                if len(self.resident) > self.K:
                    self.resident.popitem(last=False)  # LRU raus


def replay(flat_stream, K, store: BlockStore):
    """Spielt den (token-geordneten) Block-Anforderungs-Stream durch den Cache.
    Rueckgabe: fetch_count. Identisch zur offload_sim-LRU-Logik => Stage-0-Anker."""
    cache = LRUBlockCache(K, store)
    for b in flat_stream:
        cache.request([b])
    return store.fetch_count


# ----------------------------------------------------------------------------- Stage 0 Run

def run_stage0(checkpoint, K=16, n_batches=4, bs=16, seq_len=128, device="cpu"):
    data = HeteroMiniData()
    m, arch, step = model_io.load_checkpoint(checkpoint, data.vocab_size, device)
    name = model_io.label(arch, step)
    if model_io.is_dense(arch):
        raise SystemExit("Stage 0 erwartet ein sparse SR-Core-Modell (Bank).")

    blocks = model_io.blocks_of(m, arch)
    store = BlockStore(blocks, dtype_bytes=2)
    print(f"[Stream] {name}: n_blocks={store.n}, block={store.block_params/1e6:.2f}M "
          f"= {store.block_bytes/1024:.1f} KB fp16")

    # Routing-Traces aus dem echten Modell (wie offload_sim)
    traces, _, _, R, k = _collect(m, data, "contiguous", n_batches, bs, seq_len, device)
    streams = _streams(traces)
    flat = _flatten(streams)
    n_tokens = sum(tr.shape[1] * tr.shape[2] for tr in traces)
    print(f"[Stream] R={R} k={k}  n_tokens={n_tokens}  Block-Anforderungen={len(flat)}")

    # Engine-Replay
    fetches = replay(flat, K, store)
    bytes_per_token = fetches * store.block_bytes / n_tokens

    # KORREKTHEITS-ANKER: muss exakt der offload_sim-LRU-Miss-Zahl entsprechen
    ref_miss = sim_lru(flat, K)
    ok = (fetches == ref_miss)
    print(f"[Stream] K={K}:  Engine-Fetches={fetches}   offload_sim-LRU-Miss={ref_miss}   "
          f"{'[MATCH]' if ok else '[MISMATCH]'}")
    print(f"[Stream] bytes/token (fp16) = {bytes_per_token/1024:.2f} KB  "
          f"(Stage 1 misst daraus echte Tokens/s)")
    if not ok:
        raise SystemExit("Replay weicht von offload_sim ab — Engine ist NICHT faithful.")
    return {"model": name, "K": K, "fetches": fetches, "bytes_per_token_fp16": bytes_per_token,
            "n_blocks": store.n, "block_bytes": store.block_bytes}


# ----------------------------------------------------------------------------- Stage 1 (GPU)

import copy
import time as _time
import torch


def _calibrate_h2d_bandwidth(block_bytes, device, iters=200):
    """Echte H2D-Bandbreite mit block-grossen pinned Copies messen (NICHT 16 GB/s annehmen)."""
    n = block_bytes // 2  # fp16-Elemente
    src = torch.empty(n, dtype=torch.float16).pin_memory()
    dst = torch.empty(n, dtype=torch.float16, device=device)
    torch.cuda.synchronize()
    t0 = _time.perf_counter()
    for _ in range(iters):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    dt = _time.perf_counter() - t0
    gbs = (block_bytes * iters) / dt / 1e9
    return gbs


class StreamingEngineGPU:
    """Blockgewichte pinned im CPU-RAM; K vorab-allokierte VRAM-Slot-Module; LRU.
    fetch() = echter copy_(non_blocking=True) pinned->VRAM (der gemessene Transfer)."""

    def __init__(self, blocks, K, device):
        self.K = K
        self.device = device
        self.block_params = sum(p.numel() for p in blocks[0].parameters())
        self.block_bytes = self.block_params * 2  # fp16
        # pinned CPU-Kopien aller Bloecke (fp16)
        self.cpu_blocks = []
        for blk in blocks:
            self.cpu_blocks.append({nm: p.detach().half().contiguous().pin_memory()
                                    for nm, p in blk.named_parameters()})
        # K GPU-Slot-Module (fp16), Architektur eines echten Blocks
        self.slots = [copy.deepcopy(blocks[0]).half().to(device).eval() for _ in range(K)]
        self.slot_params = [dict(s.named_parameters()) for s in self.slots]
        self.resident = OrderedDict()      # block_id -> slot_idx (LRU)
        self.free = list(range(K))
        self.fetch_count = 0

    @torch.no_grad()
    def ensure(self, block_id):
        """Gibt das Slot-Modul fuer block_id zurueck; laedt es bei Miss real ins VRAM."""
        if block_id in self.resident:
            self.resident.move_to_end(block_id)
            return self.slots[self.resident[block_id]]
        if self.free:
            slot = self.free.pop()
        else:
            _, slot = self.resident.popitem(last=False)  # LRU-Opfer
        for nm, p in self.slot_params[slot].items():     # ECHTER H2D-Transfer
            p.data.copy_(self.cpu_blocks[block_id][nm], non_blocking=True)
        self.resident[block_id] = slot
        self.fetch_count += 1
        return self.slots[slot]


@torch.no_grad()
def _serve(engine, per_token_blocks, R, d_model, device, k_for_avg):
    """Autoregressives Batch=1-Serving: pro Token k Bloecke holen, R Rekursionsschritte."""
    h = torch.randn(1, d_model, dtype=torch.float16, device=device)
    for bids in per_token_blocks:
        mods = [engine.ensure(int(b)) for b in bids]     # Misses -> echter Transfer
        for _ in range(R):
            acc = mods[0](h)
            for m in mods[1:]:
                acc = acc + m(h)
            h = torch.tanh(h + acc / max(len(mods), 1))   # bounded, Werte egal (nur Timing)
    return h


def run_stage1(checkpoint, K=16, n_tokens=512, n_batches=4, bs=16, seq_len=128,
               dense_depth=24, device="cuda"):
    if not torch.cuda.is_available():
        raise SystemExit("Stage 1 braucht CUDA.")
    data = HeteroMiniData()
    m, arch, step = model_io.load_checkpoint(checkpoint, data.vocab_size, "cpu")
    name = model_io.label(arch, step)
    blocks = model_io.blocks_of(m, arch)
    d_model = m.cfg.d_model

    # Routing pro Token (r=1-Auswahl, in r2..R wiederverwendet) aus echten Traces
    traces, _, _, R, k = _collect(m, data, "contiguous", n_batches, bs, seq_len, "cpu")
    per_token = []
    for tr in traces:                       # tr: (R,B,T,k)
        r1 = tr[0]                           # (B,T,k) — die wiederverwendete Auswahl
        B, T, kk = r1.shape
        for b in range(B):
            for t in range(T):
                per_token.append(r1[b, t])
    per_token = per_token[:n_tokens]
    n = len(per_token)

    block_bytes = sum(p.numel() for p in blocks[0].parameters()) * 2  # fp16
    bw = _calibrate_h2d_bandwidth(block_bytes, device)
    print(f"[Stage1] {name}  d_model={d_model} R={R} k={k}  n_tokens={n}")
    print(f"[Stage1] gemessene H2D-Bandbreite (pinned, block-gross): {bw:.1f} GB/s  "
          f"(Sim nahm 16 an)")

    def bench(per_tok, R_use, K_use, label_):
        eng = StreamingEngineGPU(blocks, K_use, device)
        _serve(eng, per_tok[:32], R_use, d_model, device, k)        # Warmup
        eng.fetch_count = 0
        torch.cuda.synchronize(); t0 = _time.perf_counter()
        _serve(eng, per_tok, R_use, d_model, device, k)
        torch.cuda.synchronize(); dt = _time.perf_counter() - t0
        fpt = eng.fetch_count / len(per_tok)
        bpt = fpt * eng.block_bytes
        toks = len(per_tok) / dt
        print(f"  {label_:28} K={K_use:<3} {toks:8.0f} tok/s  "
              f"fetch/tok={fpt:5.2f}  by/tok={bpt/1024:7.1f}KB  "
              f"transfer-bound={bpt/(bw*1e9)*1e6:5.1f}us/tok")
        return {"label": label_, "K": K_use, "tok_per_s": round(toks, 1),
                "fetch_per_tok": round(fpt, 3), "bytes_per_tok": int(bpt)}

    print("\n=== Stage 1: gemessene Tokens/s (Batch=1-Serving, synchron, KEIN Overlap) ===")
    rows = []
    rows.append(bench(per_token, R, K, f"SR-Core stream (k={k},R={R})"))
    rows.append(bench(per_token, R, len(blocks), f"SR-Core all-resident (Ceiling)"))
    # Dense-Layer-Offload-Baseline: alle dense_depth Bloecke pro Token, R=1
    dense_tok = [list(range(dense_depth)) for _ in range(n)]
    rows.append(bench(dense_tok, 1, K, f"Dense-offload (D={dense_depth})"))
    print("\n(all-resident = Compute-Ceiling ohne Transfer; Differenz zu 'stream' = Transfer-Kosten.")
    print(" Batch=1-Python-Loop hat Kernel-Launch-Overhead -> Absolutwerte konservativ; "
          "der SR-Core-vs-Dense-Vergleich ist die Aussage.)")
    out = {"model": name, "K": K, "R": R, "k": k, "n_tokens": n,
           "bandwidth_gbs_measured": round(bw, 1), "rows": rows}
    import json
    with open(os.path.join(RESULTS, "streaming_stage1.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Gespeichert: {os.path.join(RESULTS, 'streaming_stage1.json')}")
    return out


# ----------------------------------------------------------------------------- Stage 2 (grouped)

class GroupedStreamingEngine:
    """Wie StreamingEngineGPU, aber der VRAM-Cache ist GESTAPELT (K, ...) im bmm-Layout.
    Fetch kopiert direkt in die Slot-Zeile; pro Token werden die k aktiven Slots per
    index_select gegriffen und die Rekursion als gruppierter bmm gerechnet (statt k Calls)."""

    def __init__(self, blocks, K, device):
        self.K = K
        self.device = device
        b0 = blocks[0]
        self.d = b0.fc1.in_features
        self.h = b0.fc1.out_features
        self.block_bytes = sum(p.numel() for p in b0.parameters()) * 2
        d, h = self.d, self.h
        # gestapelter VRAM-Cache im bmm-Layout (W1: d->h, W2: h->d transponiert)
        self.NW = torch.empty(K, d, dtype=torch.float16, device=device)
        self.NB = torch.empty(K, d, dtype=torch.float16, device=device)
        self.W1 = torch.empty(K, d, h, dtype=torch.float16, device=device)
        self.B1 = torch.empty(K, h, dtype=torch.float16, device=device)
        self.W2 = torch.empty(K, h, d, dtype=torch.float16, device=device)
        self.B2 = torch.empty(K, d, dtype=torch.float16, device=device)
        # pinned CPU-Kopien pro Block, gleiches Layout -> Fetch ist reiner copy_
        self.cpu = []
        for blk in blocks:
            self.cpu.append(dict(
                nw=blk.norm.weight.detach().half().contiguous().pin_memory(),
                nb=blk.norm.bias.detach().half().contiguous().pin_memory(),
                w1=blk.fc1.weight.detach().t().half().contiguous().pin_memory(),   # (d,h)
                b1=blk.fc1.bias.detach().half().contiguous().pin_memory(),
                w2=blk.fc2.weight.detach().t().half().contiguous().pin_memory(),   # (h,d)
                b2=blk.fc2.bias.detach().half().contiguous().pin_memory()))
        self.resident = OrderedDict()  # block_id -> slot
        self.free = list(range(K))
        self.fetch_count = 0

    @torch.no_grad()
    def ensure(self, block_id):
        if block_id in self.resident:
            self.resident.move_to_end(block_id)
            return self.resident[block_id]
        slot = self.free.pop() if self.free else self.resident.popitem(last=False)[1]
        c = self.cpu[block_id]
        self.NW[slot].copy_(c["nw"], non_blocking=True)   # ECHTER H2D
        self.NB[slot].copy_(c["nb"], non_blocking=True)
        self.W1[slot].copy_(c["w1"], non_blocking=True)
        self.B1[slot].copy_(c["b1"], non_blocking=True)
        self.W2[slot].copy_(c["w2"], non_blocking=True)
        self.B2[slot].copy_(c["b2"], non_blocking=True)
        self.resident[block_id] = slot
        self.fetch_count += 1
        return slot


@torch.no_grad()
def _serve_grouped(eng, per_token_blocks, R, device):
    d = eng.d
    h = torch.randn(1, d, dtype=torch.float16, device=device)
    for bids in per_token_blocks:
        slots = torch.tensor([eng.ensure(int(b)) for b in bids], device=device)
        nw = eng.NW.index_select(0, slots)   # (k,d)
        nb = eng.NB.index_select(0, slots)
        w1 = eng.W1.index_select(0, slots)   # (k,d,h)
        b1 = eng.B1.index_select(0, slots)   # (k,h)
        w2 = eng.W2.index_select(0, slots)   # (k,h,d)
        b2 = eng.B2.index_select(0, slots)
        kk = slots.shape[0]
        for _ in range(R):
            base = torch.nn.functional.layer_norm(h, (d,))      # (1,d), norm gleich fuer alle
            hn = (base * nw + nb).unsqueeze(1)                  # (k,1,d)
            a = torch.baddbmm(b1.unsqueeze(1), hn, w1)          # (k,1,h)
            a = torch.nn.functional.gelu(a)
            y = torch.baddbmm(b2.unsqueeze(1), a, w2).squeeze(1)  # (k,d)
            h = torch.tanh(h + y.sum(0, keepdim=True) / kk)
    return h


def run_stage2(checkpoint, K=16, n_tokens=512, n_batches=4, bs=16, seq_len=128,
               dense_depth=24, device="cuda"):
    if not torch.cuda.is_available():
        raise SystemExit("Stage 2 braucht CUDA.")
    data = HeteroMiniData()
    m, arch, step = model_io.load_checkpoint(checkpoint, data.vocab_size, "cpu")
    name = model_io.label(arch, step)
    blocks = model_io.blocks_of(m, arch)
    d_model = m.cfg.d_model
    traces, _, _, R, k = _collect(m, data, "contiguous", n_batches, bs, seq_len, "cpu")
    per_token = []
    for tr in traces:
        r1 = tr[0]
        B, T, kk = r1.shape
        for b in range(B):
            for t in range(T):
                per_token.append(r1[b, t])
    per_token = per_token[:n_tokens]
    block_bytes = sum(p.numel() for p in blocks[0].parameters()) * 2
    bw = _calibrate_h2d_bandwidth(block_bytes, device)
    print(f"[Stage2] {name}  d_model={d_model} R={R} k={k}  n_tokens={len(per_token)}  "
          f"H2D={bw:.1f} GB/s")

    def bench_grouped(per_tok, R_use, K_use, label_):
        eng = GroupedStreamingEngine(blocks, K_use, device)
        _serve_grouped(eng, per_tok[:32], R_use, device)
        eng.fetch_count = 0
        torch.cuda.synchronize(); t0 = _time.perf_counter()
        _serve_grouped(eng, per_tok, R_use, device)
        torch.cuda.synchronize(); dt = _time.perf_counter() - t0
        fpt = eng.fetch_count / len(per_tok)
        toks = len(per_tok) / dt
        print(f"  {label_:34} K={K_use:<3} {toks:8.0f} tok/s  fetch/tok={fpt:5.2f}  "
              f"by/tok={fpt*eng.block_bytes/1024:7.1f}KB")
        return {"label": label_, "K": K_use, "tok_per_s": round(toks, 1),
                "fetch_per_tok": round(fpt, 3)}

    def bench_perblock(per_tok, R_use, K_use, label_):  # Stage-1-Pfad zum Vergleich
        eng = StreamingEngineGPU(blocks, K_use, device)
        _serve(eng, per_tok[:32], R_use, d_model, device, k)
        eng.fetch_count = 0
        torch.cuda.synchronize(); t0 = _time.perf_counter()
        _serve(eng, per_tok, R_use, d_model, device, k)
        torch.cuda.synchronize(); dt = _time.perf_counter() - t0
        toks = len(per_tok) / dt
        print(f"  {label_:34} K={K_use:<3} {toks:8.0f} tok/s")
        return {"label": label_, "K": K_use, "tok_per_s": round(toks, 1)}

    print("\n=== Stage 2: gruppierter bmm vs. Stage-1 per-Block (Batch=1, synchron) ===")
    rows = []
    rows.append(bench_perblock(per_token, R, K, "SR-Core per-block (Stage 1)"))
    rows.append(bench_grouped(per_token, R, K, "SR-Core grouped (Stage 2)"))
    rows.append(bench_grouped(per_token, R, len(blocks), "SR-Core grouped all-resident"))
    dense_tok = [list(range(dense_depth)) for _ in range(len(per_token))]
    rows.append(bench_grouped(dense_tok, 1, K, f"Dense-offload grouped (D={dense_depth})"))
    out = {"model": name, "K": K, "R": R, "k": k, "bandwidth_gbs_measured": round(bw, 1),
           "rows": rows}
    import json
    with open(os.path.join(RESULTS, "streaming_stage2.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nGespeichert: {os.path.join(RESULTS, 'streaming_stage2.json')}")
    return out


# ----------------------------------------------------------------------------- Stage 3 (overlap)

def schedule_lru(per_token, K):
    """Offline-LRU -> pro Token: copies=[(block_id,slot)], slots=[slot je aktivem Block].
    Beim Planen der Kopien eines Tokens wird das AKTIVE Set des VORtokens von der Verdraengung
    ausgeschlossen — so ueberschreibt der Prefetch (copy t+1 || compute t) nie einen In-Use-Slot."""
    slot_of = {}                 # block_id -> slot
    lru = OrderedDict()          # block_id -> True (LRU-Reihenfolge)
    free = list(range(K))
    sched = []
    prev_active = set()
    for bids in per_token:
        cur = [int(b) for b in bids]
        copies = []
        for b in cur:
            if b in slot_of:
                lru.move_to_end(b)
                continue
            if free:
                s = free.pop()
            else:
                victim = next((vb for vb in lru if vb not in prev_active and vb not in cur), None)
                if victim is None:
                    raise SystemExit(f"K={K} zu klein fuer 1-ahead-Prefetch (|cur∪prev|>K). "
                                     f"Groesseres K waehlen.")
                del lru[victim]; s = slot_of.pop(victim)
            slot_of[b] = s; lru[b] = True
            copies.append((b, s))
        sched.append({"copies": copies, "slots": [slot_of[b] for b in cur]})
        prev_active = set(cur)
    return sched


@torch.no_grad()
def run_stage3(checkpoint, K=32, n_tokens=512, n_batches=4, bs=16, seq_len=128,
               device="cuda"):
    if not torch.cuda.is_available():
        raise SystemExit("Stage 3 braucht CUDA.")
    data = HeteroMiniData()
    m, arch, step = model_io.load_checkpoint(checkpoint, data.vocab_size, "cpu")
    name = model_io.label(arch, step)
    blocks = model_io.blocks_of(m, arch)
    traces, _, _, R, k = _collect(m, data, "contiguous", n_batches, bs, seq_len, "cpu")
    per_token = []
    for tr in traces:
        r1 = tr[0]; B, T, kk = r1.shape
        for b in range(B):
            for t in range(T):
                per_token.append(r1[b, t])
    per_token = per_token[:n_tokens]
    n = len(per_token)
    d = blocks[0].fc1.in_features
    bw = _calibrate_h2d_bandwidth(sum(p.numel() for p in blocks[0].parameters()) * 2, device)
    print(f"[Stage3] {name}  R={R} k={k}  n_tokens={n}  K={K}  H2D={bw:.1f} GB/s")

    eng = GroupedStreamingEngine(blocks, K, device)

    def copy_block(bid, slot):
        c = eng.cpu[bid]
        eng.NW[slot].copy_(c["nw"], non_blocking=True); eng.NB[slot].copy_(c["nb"], non_blocking=True)
        eng.W1[slot].copy_(c["w1"], non_blocking=True); eng.B1[slot].copy_(c["b1"], non_blocking=True)
        eng.W2[slot].copy_(c["w2"], non_blocking=True); eng.B2[slot].copy_(c["b2"], non_blocking=True)

    def grouped_step(h, slots_t):
        nw = eng.NW.index_select(0, slots_t); nb = eng.NB.index_select(0, slots_t)
        w1 = eng.W1.index_select(0, slots_t); b1 = eng.B1.index_select(0, slots_t)
        w2 = eng.W2.index_select(0, slots_t); b2 = eng.B2.index_select(0, slots_t)
        kk = slots_t.shape[0]
        for _ in range(R):
            base = torch.nn.functional.layer_norm(h, (d,))
            hn = (base * nw + nb).unsqueeze(1)
            a = torch.nn.functional.gelu(torch.baddbmm(b1.unsqueeze(1), hn, w1))
            y = torch.baddbmm(b2.unsqueeze(1), a, w2).squeeze(1)
            h = torch.tanh(h + y.sum(0, keepdim=True) / kk)
        return h

    def run_overlap(sched):
        msteps = len(sched)
        slot_tensors = [torch.tensor(s["slots"], device=device) for s in sched]
        copy_stream = torch.cuda.Stream()
        events = [torch.cuda.Event() for _ in range(msteps)]
        # Token 0 vorab laden
        with torch.cuda.stream(copy_stream):
            for bid, slot in sched[0]["copies"]:
                copy_block(bid, slot)
        events[0].record(copy_stream)
        h = torch.randn(1, d, dtype=torch.float16, device=device)
        torch.cuda.synchronize(); t0 = _time.perf_counter()
        for t in range(msteps):
            if t + 1 < msteps:                             # Prefetch t+1 auf Copy-Stream
                with torch.cuda.stream(copy_stream):
                    for bid, slot in sched[t + 1]["copies"]:
                        copy_block(bid, slot)
                events[t + 1].record(copy_stream)
            torch.cuda.current_stream().wait_event(events[t])   # warte auf t's Kopien
            h = grouped_step(h, slot_tensors[t])
        torch.cuda.synchronize(); return _time.perf_counter() - t0

    def run_serial(sched):  # gleiche Engine, KEIN Overlap (Stage-2-Pfad) zum Vergleich
        msteps = len(sched)
        slot_tensors = [torch.tensor(s["slots"], device=device) for s in sched]
        h = torch.randn(1, d, dtype=torch.float16, device=device)
        torch.cuda.synchronize(); t0 = _time.perf_counter()
        for t in range(msteps):
            for bid, slot in sched[t]["copies"]:
                copy_block(bid, slot)
            h = grouped_step(h, slot_tensors[t])
        torch.cuda.synchronize(); return _time.perf_counter() - t0

    sched = schedule_lru(per_token, K)
    fetches = sum(len(s["copies"]) for s in sched)
    # Warmup beider Pfade
    run_serial(sched[:32]); run_overlap(sched[:32])
    dt_serial = run_serial(sched)
    dt_overlap = run_overlap(sched)
    tok_serial, tok_overlap = n / dt_serial, n / dt_overlap
    print("\n=== Stage 3: Stream-Overlap (Prefetch t+1 || Compute t) ===")
    print(f"  grouped, KEIN Overlap (Stage 2)   {tok_serial:8.0f} tok/s")
    print(f"  grouped, MIT Overlap (Stage 3)    {tok_overlap:8.0f} tok/s   "
          f"({tok_overlap/tok_serial:.2f}x)")
    print(f"  fetch/tok={fetches/n:.2f}  (Ceiling aus Stage 2 war ~364 tok/s)")
    out = {"model": name, "K": K, "R": R, "k": k, "n_tokens": n,
           "tok_per_s_serial": round(tok_serial, 1), "tok_per_s_overlap": round(tok_overlap, 1),
           "speedup": round(tok_overlap / tok_serial, 3), "bandwidth_gbs_measured": round(bw, 1)}
    import json
    with open(os.path.join(RESULTS, "streaming_stage3.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Gespeichert: {os.path.join(RESULTS, 'streaming_stage3.json')}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="sparse SR-Core Checkpoint")
    ap.add_argument("--K", type=int, default=16, help="VRAM-Cache-Kapazitaet (Bloecke)")
    ap.add_argument("--stage", type=int, default=0, choices=[0, 1, 2, 3],
                    help="0=CPU-Anker, 1=GPU per-Block, 2=gruppiert, 3=gruppiert+Overlap")
    ap.add_argument("--n_tokens", type=int, default=512, help="Stage 1: Tokens fuers Timing")
    ap.add_argument("--n_batches", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--dense_depth", type=int, default=24)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if a.stage == 3:
        run_stage3(a.checkpoint, K=a.K, n_tokens=a.n_tokens, n_batches=a.n_batches,
                   bs=a.bs, seq_len=a.seq_len, device=a.device or "cuda")
    elif a.stage == 2:
        run_stage2(a.checkpoint, K=a.K, n_tokens=a.n_tokens, n_batches=a.n_batches,
                   bs=a.bs, seq_len=a.seq_len, dense_depth=a.dense_depth,
                   device=a.device or "cuda")
    elif a.stage == 1:
        run_stage1(a.checkpoint, K=a.K, n_tokens=a.n_tokens, n_batches=a.n_batches,
                   bs=a.bs, seq_len=a.seq_len, dense_depth=a.dense_depth,
                   device=a.device or "cuda")
    else:
        run_stage0(a.checkpoint, K=a.K, n_batches=a.n_batches, bs=a.bs,
                   seq_len=a.seq_len, device=a.device or "cpu")
