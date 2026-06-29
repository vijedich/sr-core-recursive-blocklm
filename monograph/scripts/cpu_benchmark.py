"""CPU-Benchmark: dense vs. sparse Inferenzkosten (ohne Streaming, alles im RAM).

Frage: Laesst sich das kleine aktive Working Set schon OHNE Streaming in einen realen
CPU-Speedup uebersetzen? Misst Wall-Time/Token auf der CPU fuer:

  ModelC-Bank (b64, k=4), pro Variante und fuer R=6 UND R=2:
    A  dense_all   : alle n_blocks Bloecke auf alle Tokens (kein Routing)
    B  sparse_route: echter Router + Top-k-Dispatch (normaler Forward)
    C  sparse_pre  : vorberechnete Route, nur Dispatch (Router-Kosten = B - C)
    D  fixed_core  : nur die ~WS haeufigsten Bloecke (fester kleiner Kern)
  ModelA dense-Baselines (falls trainiert): echter Forward von depth=24 und depth=8.

Nur die Rechenzeit zaehlt (keine Qualitaet). Random-Tokens genuegen — die Dispatch-/
Router-Kosten haengen nicht vom konkreten Token ab.

Wiederverwendbar: nimmt beliebige Checkpoints (--glob/--checkpoints), Architektur kommt aus
dem Checkpoint selbst (rblm.model_io). Sparse-Modelle -> A/B/C/D-Varianten; Dense -> Kern-Zeit.

Nutzung:
  python scripts/cpu_benchmark.py --glob "results/hm_cont_*.pt"
"""
from __future__ import annotations
import argparse, os, time, statistics, json
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU erzwingen
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rblm import model_io

DEV = "cpu"


def time_fn(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)




@torch.no_grad()
def bench_bank(m, toks, R, iters):
    """A/B/C/D-Varianten auf der ModelC-Bank, fuer gegebenes R."""
    bank = m.bank
    d = m.cfg.d_model
    nb = m.cfg.n_blocks
    h0 = m.encode(toks)
    B, T, _ = h0.shape

    # --- B: sparse + echter Router ---
    def variant_B():
        h = h0
        for _ in range(R):
            hf = h.reshape(-1, d)
            ti, g, _ = bank.route(hf, False)
            out, _ = bank.dispatch(hf, ti, g)
            h = h + out.reshape(B, T, d)
        return h
    tB = time_fn(variant_B, iters)

    # Route einmal vorberechnen (pro Iteration), fuer C
    h = h0; routes = []
    for _ in range(R):
        hf = h.reshape(-1, d)
        ti, g, _ = bank.route(hf, False)
        routes.append((ti, g))
        out, _ = bank.dispatch(hf, ti, g)
        h = h + out.reshape(B, T, d)

    # --- C: vorberechnete Route, nur Dispatch ---
    def variant_C():
        h = h0
        for r in range(R):
            hf = h.reshape(-1, d)
            ti, g = routes[r]
            out, _ = bank.dispatch(hf, ti, g)
            h = h + out.reshape(B, T, d)
        return h
    tC = time_fn(variant_C, iters)

    # --- A: dense, ALLE Bloecke ---
    def variant_A():
        h = h0
        for _ in range(R):
            hf = h.reshape(-1, d)
            acc = torch.zeros_like(hf)
            for b in range(nb):
                acc = acc + bank.blocks[b](hf)
            h = h + acc.reshape(B, T, d)
        return h
    tA = time_fn(variant_A, max(2, iters // 4))  # teuer -> weniger Wiederholungen

    # --- D: fester kleiner Kern (die WS haeufigsten Bloecke) ---
    # Usage aus einem Routing-Pass bestimmen
    usage = torch.zeros(nb)
    h = h0
    for _ in range(R):
        hf = h.reshape(-1, d)
        ti, g, _ = bank.route(hf, False)
        usage += torch.bincount(ti.reshape(-1), minlength=nb).float()
        out, _ = bank.dispatch(hf, ti, g)
        h = h + out.reshape(B, T, d)
    ws = int(round(_ws_estimate(m, toks, R)))
    core_blocks = usage.topk(max(1, ws)).indices.tolist()

    def variant_D():
        h = h0
        for _ in range(R):
            hf = h.reshape(-1, d)
            acc = torch.zeros_like(hf)
            for b in core_blocks:
                acc = acc + bank.blocks[b](hf)
            h = h + acc.reshape(B, T, d)
        return h
    tD = time_fn(variant_D, iters)

    return {"A_dense_all": tA, "B_sparse_route": tB, "C_sparse_pre": tC,
            "D_fixed_core": tD, "router_cost": tB - tC, "core_blocks": len(core_blocks)}


@torch.no_grad()
def _ws_estimate(m, toks, R):
    """Mittlere eindeutige Bloecke/Token (Working Set) fuer R Iterationen."""
    bank = m.bank; d = m.cfg.d_model; nb = m.cfg.n_blocks
    h = m.encode(toks); B, T, _ = h.shape
    per_tok = [set() for _ in range(B * T)]
    for _ in range(R):
        hf = h.reshape(-1, d)
        ti, g, _ = bank.route(hf, False)
        for i, row in enumerate(ti.tolist()):
            per_tok[i].update(row)
        out, _ = bank.dispatch(hf, ti, g)
        h = h + out.reshape(B, T, d)
    return sum(len(s) for s in per_tok) / len(per_tok)


@torch.no_grad()
def bench_dense(m, toks, iters):
    """Nur der Dense-KERN (Blockstapel ab h0, OHNE Deep-Supervision-Readouts) — fair
    gegen die Sparse-Bank-Varianten, die ebenfalls nur den Kern messen. Bei Inferenz
    braucht man nur EINEN finalen Readout (fuer beide Modelle identische Konstante)."""
    h0 = m.encode(toks)

    def core():
        h = h0
        for blk in m.blocks:
            h = h + blk(h)
        return h
    return time_fn(core, iters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/hm_cont_*.pt",
                    help="Glob fuer Checkpoints (Default: alle Continuation-Snapshots)")
    ap.add_argument("--checkpoints", nargs="*", default=None,
                    help="explizite Checkpoint-Pfade (ueberschreibt --glob)")
    ap.add_argument("--vocab", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--threads", type=int, default=0, help="0 = torch default")
    ap.add_argument("--sparse_Rs", type=int, nargs="*", default=None,
                    help="zusaetzliche R-Werte fuer Sparse (Default: trainiertes R + 2)")
    a = ap.parse_args()
    if a.threads > 0:
        torch.set_num_threads(a.threads)

    cks = a.checkpoints or model_io.discover(os.path.join(ROOT, a.glob) if not os.path.isabs(a.glob) else a.glob)
    if not cks:
        cks = model_io.discover(a.glob)
    if not cks:
        raise SystemExit(f"Keine Checkpoints gefunden (glob={a.glob!r}).")

    n_tok = a.bs * a.seq
    toks = torch.randint(0, a.vocab, (a.bs, a.seq))
    print(f"CPU-Benchmark: bs={a.bs} seq={a.seq} ({n_tok} Tokens/Forward), "
          f"torch_threads={torch.get_num_threads()}, median ueber {a.iters} Laeufe\n", flush=True)
    results = {"config": {"bs": a.bs, "seq": a.seq, "n_tok": n_tok,
                          "threads": torch.get_num_threads()}, "models": {}}

    for path in cks:
        m, arch, step = model_io.load_checkpoint(path, a.vocab, "cpu")
        name = model_io.label(arch, step)
        if model_io.is_dense(arch):
            t = bench_dense(m, toks, a.iters)
            results["models"][name] = {"kind": "dense", "core_ms": t * 1000}
            print(f"--- {name} (dense, depth={m.cfg.dense_depth}) ---")
            print(f"  core            : {t*1000:8.2f} ms/fwd   {n_tok/t:8.0f} tok/s\n")
        else:
            Rs = a.sparse_Rs or sorted({m.cfg.routed_iters, 2})
            ws = _ws_estimate(m, toks, m.cfg.routed_iters)
            print(f"--- {name} (sparse, n_blocks={m.cfg.n_blocks}, k={m.cfg.k_active}, "
                  f"WS={ws:.1f}) ---")
            results["models"][name] = {"kind": "sparse", "working_set": round(ws, 2), "by_R": {}}
            for R in Rs:
                r = bench_bank(m, toks, R, a.iters)
                results["models"][name]["by_R"][f"R{R}"] = r
                print(f"  R={R} (Kern={r['core_blocks']}):")
                for kk in ("A_dense_all", "B_sparse_route", "C_sparse_pre", "D_fixed_core"):
                    print(f"    {kk:16}: {r[kk]*1000:8.2f} ms/fwd   {n_tok/r[kk]:8.0f} tok/s")
                print(f"    router_cost     : {r['router_cost']*1000:8.2f} ms/fwd "
                      f"({100*r['router_cost']/max(r['B_sparse_route'],1e-9):.0f}% von B)")
            print()

    out = os.path.join(ROOT, "results", "cpu_benchmark.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Gespeichert: results/cpu_benchmark.json")


if __name__ == "__main__":
    main()
