"""Eval-Suite fuer Tests 3, 4, 5, 8 auf einem bestehenden Checkpoint.

Kein Training. Laeuft auf jedem Curriculum-C-kompatiblen Checkpoint.

Nutzung:
  python scripts/eval_suite.py --ckpt checkpoints/tinystories_curriculum_fromIter2/seed_0/step_3000
  python scripts/eval_suite.py --ckpt ... --tests depth_truncation state_reset
  python scripts/eval_suite.py --ckpt ... --tests all

Tests:
  depth_truncation  — L[r=1..R] Kurve (Test 3)
  state_reset       — Normal vs. State-Reset Loss (Test 4)
  forced_diversity  — Normal vs. Forced-Diversity Loss (Test 5)
  cache             — Working Set + Cache-Miss-Kurve (Test 8)
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.competence_centers_exp import make_model, _load_checkpoint, load_with_text, flat_ids, FlatData
from experiments.tinystories_exp import full_eval, cache_sim, gini, iteration_diagnostics
from rblm.models import iteration_losses

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
ALL_TESTS = ["depth_truncation", "state_reset", "forced_diversity", "cache"]


# ---------- data ----------

def load_data(vocab=8000, max_docs=20000, seq_len=128, device="cpu"):
    print("[Suite] Lade TinyStories ...", flush=True)
    texts, tok = load_with_text(vocab=vocab, max_docs=max_docs)
    ids = flat_ids(texts, tok)
    data = FlatData(ids, tok.get_vocab_size())
    print(f"[Suite] {ids.numel():,} Tokens, vocab={tok.get_vocab_size()}", flush=True)
    return data, tok.get_vocab_size()


# ---------- eval helpers ----------

@torch.no_grad()
def eval_loss_per_iter(model, data, n_batches, bs, seq_len, device, **kw):
    """Gibt (R,)-Vektor der mittleren CE-Losses pro Iteration zurueck."""
    model.eval()
    R = model.n_iters
    nb = model.cfg.n_blocks
    loss_sum = torch.zeros(R, device=device)
    tok_count = 0
    for _ in range(n_batches):
        toks, tgt, mask, _ = data.batch(bs, seq_len, device=device)
        logits, _ = model(toks, **kw)
        per = iteration_losses(logits, tgt, mask)
        m = int(mask.sum())
        loss_sum += per.detach() * m
        tok_count += m
    return (loss_sum / max(1, tok_count)).cpu().tolist()


# ---------- Test 3: Depth Truncation ----------

def test_depth_truncation(model, data, n_batches=16, bs=32, seq_len=128, device="cpu"):
    """L[r=1..R] — Anytime-Qualitaetskurve."""
    print("\n" + "="*60, flush=True)
    print("TEST 3 — DEPTH TRUNCATION (Anytime-Kurve)", flush=True)
    print("="*60, flush=True)

    losses = eval_loss_per_iter(model, data, n_batches, bs, seq_len, device)
    R = len(losses)
    chance = math.log(model.cfg.vocab_size)

    print(f"\n{'Iter':>6} {'Loss':>8} {'vs r=1':>8} {'vs Zufall':>10}", flush=True)
    print("-" * 40, flush=True)
    for r, l in enumerate(losses, 1):
        delta = l - losses[0]
        vs_chance = l - chance
        print(f"  r={r}    {l:.4f}   {delta:+.4f}    {vs_chance:+.4f}", flush=True)

    anytime_delta = losses[0] - min(losses)
    best_r = losses.index(min(losses)) + 1
    print(f"\n  Tiefengewinn d(L1-Lmin) = {anytime_delta:.4f} Nats  (best r={best_r})", flush=True)
    print(f"  Zufalls-Baseline: {chance:.4f} Nats", flush=True)

    if anytime_delta < 0.01:
        print("  >> Anytime-Kurve fast flach — kein sichtbarer Tiefennutzen", flush=True)
    elif anytime_delta < 0.05:
        print("  >> Schwacher Tiefennutzen vorhanden", flush=True)
    else:
        print("  >> Klarer Tiefennutzen — Iterationen sind produktiv", flush=True)

    return {"loss_per_iter": losses, "anytime_delta": round(anytime_delta, 5),
            "best_r": best_r}


# ---------- Test 4: State Reset ----------

def test_state_reset(model, data, n_batches=16, bs=32, seq_len=128, device="cpu"):
    """Vergleich Normal vs. State-Reset zwischen Iterationen.

    State Reset: h wird nach jeder Iteration auf h0 zurueckgesetzt.
    Wenn der Tiefengewinn verschwindet -> spätere Iterationen bauen
    tatsächlich auf dem akkumulierten Zustand auf.
    """
    print("\n" + "="*60, flush=True)
    print("TEST 4 — STATE-RESET-TEST", flush=True)
    print("="*60, flush=True)

    normal = eval_loss_per_iter(model, data, n_batches, bs, seq_len, device)
    reset  = eval_loss_per_iter(model, data, n_batches, bs, seq_len, device,
                                state_reset=True)
    R = len(normal)

    print(f"\n{'Iter':>6} {'Normal':>8} {'Reset':>8} {'Δ(Reset−Normal)':>16}", flush=True)
    print("-" * 44, flush=True)
    for r in range(R):
        delta = reset[r] - normal[r]
        print(f"  r={r+1}    {normal[r]:.4f}   {reset[r]:.4f}   {delta:+.4f}", flush=True)

    delta_final = reset[-1] - normal[-1]
    print(f"\n  Δ bei r={R} (final): {delta_final:+.4f} Nats", flush=True)

    if delta_final < 0.005:
        print("  >> Kein Unterschied — State-Reset hat keinen Effekt", flush=True)
        print("     Spätere Iterationen bauen NICHT auf akkumuliertem Zustand auf.", flush=True)
    elif delta_final < 0.05:
        print("  >> Kleiner Unterschied — schwache Zustandsakkumulation", flush=True)
    else:
        print("  >> Klarer Unterschied — echte iterative Zustandsakkumulation nachgewiesen!", flush=True)
        print("     State-Reset zerstört den Tiefengewinn — starkes Architekturargument.", flush=True)

    return {"loss_normal": normal, "loss_reset": reset,
            "delta_final": round(delta_final, 5)}


# ---------- Test 5: Forced Diversity ----------

def test_forced_diversity(model, data, n_batches=8, bs=32, seq_len=128, device="cpu"):
    """Normal-Eval vs. Forced-Diversity-Eval.

    Forced: bei Iteration r werden die k häufigsten Blöcke aus r-1 gesperrt
    (identisches Signal wie beim Training mit --diverse).

    Wenn Normal ≈ Forced -> Modell hat Diversity-Potential vollständig internalisiert.
    Wenn Forced < Normal -> Diversity würde bei Eval noch helfen (nicht vollständig gelernt).
    Wenn Forced > Normal -> Forced-Diversity schadet bei Eval (echte gelernte Struktur).
    """
    print("\n" + "="*60, flush=True)
    print("TEST 5 — INFERENCE MIT UND OHNE DIVERSITY-ZWANG", flush=True)
    print("="*60, flush=True)

    print("\n  Laufe iteration_diagnostics (run_forced=True) ...", flush=True)
    diag = iteration_diagnostics(
        model, data, n_batches=n_batches, bs=bs, seq_len=seq_len,
        device=device, run_forced=True,
    )

    normal = diag["loss_normal"]
    forced = diag["loss_forced"]
    delta  = diag["forced_minus_normal"]
    R = len(normal)

    print(f"\n{'Iter':>6} {'Normal':>8} {'Forced':>8} {'Δ(F−N)':>8} {'Bedeutung':>12}",
          flush=True)
    print("-" * 55, flush=True)
    for r in range(R):
        fn = forced[r]
        fn_str = f"{fn:.4f}" if not (isinstance(fn, float) and math.isnan(fn)) else "  nan "
        d  = delta[r]
        d_str  = f"{d:+.4f}" if not (isinstance(d, float) and math.isnan(d)) else "   nan"
        sig = "Forced hilft" if (not math.isnan(d) and d < -0.005) else \
              "Forced schadet" if (not math.isnan(d) and d > 0.005) else "~gleich"
        print(f"  r={r+1}    {normal[r]:.4f}   {fn_str}   {d_str}   {sig}", flush=True)

    state_change = diag["state_rel_change"]
    print(f"\n  Relative Zustandsaenderung ||Δh|| / ||h|| pro Iteration:", flush=True)
    for r, sc in enumerate(state_change, 1):
        print(f"    r={r}: {sc:.4f}", flush=True)

    nan_count = sum(1 for d in delta if isinstance(d, float) and math.isnan(d))
    if nan_count > 0:
        print(f"\n  HINWEIS: {nan_count} NaN-Werte — kumulative Ablation sperrt zu viele Blöcke.",
              flush=True)
        print("  (Bekanntes Problem bei nahezu uniform geroutetem Modell; Gini≈0.13)", flush=True)

    return {"loss_normal": normal, "loss_forced": forced,
            "delta": delta, "state_rel_change": state_change}


# ---------- Test 8: Cache + Working Set ----------

def test_cache(model, data, n_batches=12, bs=32, seq_len=128, device="cpu"):
    """Working Set pro Token + Cache-Miss-Kurve (gelernt vs. Random).

    Messung auf Routing-Traces: wie viele einzigartige Blöcke braucht ein Token?
    Wie oft fehlen benötigte Blöcke im Cache?
    """
    print("\n" + "="*60, flush=True)
    print("TEST 8 — CACHE + WORKING SET", flush=True)
    print("="*60, flush=True)

    nb = model.cfg.n_blocks
    print(f"\n  Laufe full_eval ({n_batches} Batches) ...", flush=True)
    routing, unique_stats, traces_arr = full_eval(
        model, data, nb, n_batches=n_batches,
        bs=bs, seq_len=seq_len, device=device,
    )

    usage = np.array(routing["usage_per_block"])
    hub_gini = gini(usage)
    top5_share = float(np.sort(usage)[-5:].sum() / max(usage.sum(), 1))

    print(f"\n  Working Set pro Token:", flush=True)
    print(f"    mean={unique_stats['mean']:.2f}  p50={unique_stats['p50']:.1f}"
          f"  p90={unique_stats['p90']:.1f}  max_possible={unique_stats['max_possible']}",
          flush=True)
    print(f"  Transfer-Reduktion vs. Layer-Offloading:", flush=True)
    print(f"    {nb} Blöcke → {unique_stats['reduction_64']:.1f}×  "
          f"(bei 256 Blöcken: {unique_stats['reduction_256']:.1f}×)", flush=True)

    print(f"\n  Hub-Struktur:", flush=True)
    print(f"    Gini={hub_gini:.4f}  Top-5-Anteil={top5_share*100:.1f}%", flush=True)

    caps = [max(model.cfg.k_active, nb // 8), nb // 4, nb // 2]
    rng = np.random.default_rng(42)
    print(f"\n  Cache-Simulation (LRU, 3 Kapazitäten) ...", flush=True)
    cache = cache_sim(traces_arr, nb, caps, rng)

    print(f"\n  {'Kapazität':>12} {'Gelernt':>10} {'Zufall':>10} {'Faktor':>8}", flush=True)
    print("  " + "-" * 44, flush=True)
    for i, cap in enumerate(caps):
        pct = f"{cap}/{nb} ({100*cap//nb}%)"
        mr_l = cache["learned"][i]
        mr_r = cache["random"][i]
        factor = mr_r / max(mr_l, 1e-6)
        print(f"  {pct:>12}   {mr_l:.4f}     {mr_r:.4f}    {factor:.1f}×", flush=True)

    hot_blocks = int(np.argsort(usage)[-4:].max())  # top-4 blocks
    print(f"\n  Jaccard (r→r+1): {routing['jaccard_consecutive']}", flush=True)

    return {
        "unique_stats": unique_stats,
        "hub_gini": round(hub_gini, 4),
        "top5_share": round(top5_share, 4),
        "cache_caps": caps,
        "cache_learned": cache["learned"],
        "cache_random": cache["random"],
        "jaccard_consecutive": routing["jaccard_consecutive"],
    }


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="Pfad zum Checkpoint-Ordner (z.B. checkpoints/tinystories_curriculum_fromIter2/seed_0/step_3000)")
    ap.add_argument("--tests", nargs="+", default=["all"],
                    choices=ALL_TESTS + ["all"],
                    help="Welche Tests laufen. 'all' = alle vier.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--vocab", type=int, default=8000)
    ap.add_argument("--max_docs", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--n_batches", type=int, default=16,
                    help="Batches für depth_truncation und state_reset (mehr = stabiler)")
    args = ap.parse_args()

    tests = ALL_TESTS if "all" in args.tests else args.tests

    # Checkpoint laden
    print(f"\n[Suite] Lade Checkpoint: {args.ckpt}", flush=True)
    model, cfg = make_model(vocab_size=args.vocab, device=args.device)
    _load_checkpoint(model, args.ckpt, device=args.device)
    model.eval()
    print(f"[Suite] n_blocks={model.cfg.n_blocks}  k={model.cfg.k_active}"
          f"  R={model.n_iters}  device={args.device}", flush=True)

    # Daten laden
    data, vocab_size = load_data(args.vocab, args.max_docs, args.seq_len, args.device)

    results = {"ckpt": args.ckpt, "tests": tests}

    if "depth_truncation" in tests:
        results["depth_truncation"] = test_depth_truncation(
            model, data, n_batches=args.n_batches, bs=args.bs,
            seq_len=args.seq_len, device=args.device)

    if "state_reset" in tests:
        results["state_reset"] = test_state_reset(
            model, data, n_batches=args.n_batches, bs=args.bs,
            seq_len=args.seq_len, device=args.device)

    if "forced_diversity" in tests:
        results["forced_diversity"] = test_forced_diversity(
            model, data, n_batches=max(8, args.n_batches // 2), bs=args.bs,
            seq_len=args.seq_len, device=args.device)

    if "cache" in tests:
        results["cache"] = test_cache(
            model, data, n_batches=12, bs=args.bs,
            seq_len=args.seq_len, device=args.device)

    # Ergebnisse speichern
    ckpt_tag = os.path.basename(os.path.dirname(os.path.dirname(args.ckpt)))
    out_path = os.path.join(RESULTS, f"eval_suite_{ckpt_tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Suite] Ergebnisse gespeichert: {out_path}", flush=True)

    print("\n" + "="*60, flush=True)
    print("EVAL-SUITE ABGESCHLOSSEN", flush=True)
    print("="*60, flush=True)


if __name__ == "__main__":
    main()
