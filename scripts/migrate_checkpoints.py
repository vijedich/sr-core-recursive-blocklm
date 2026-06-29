"""Migriert vorhandene Modell-Dateien in den versionierten Checkpoint-Workspace.

Fuer jeden Checkpoint wird ein vollstaendiger Eintrag mit Metadaten,
SHA256-Hash und Registry-Eintrag erstellt. Niemals ueberschreibt dieses
Skript einen bereits migrierten Checkpoint.

Ausfuehren:
  python scripts/migrate_checkpoints.py
  python scripts/migrate_checkpoints.py --dry-run    # nur anzeigen, nicht kopieren
"""
from __future__ import annotations
import argparse, json, os, shutil, sys

# Projekt-Root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rblm.checkpoint import CKPT_ROOT, _sha256_file, _code_version, _to_yaml, _append_registry
from datetime import datetime


def migrate_one(
    src_model: str,
    experiment: str,
    seed: int,
    step: int,
    config: dict,
    metrics: dict,
    routing_stats: dict | None = None,
    dry_run: bool = False,
    note: str = "",
):
    """Kopiert ein bestehendes model.pt in die versionierte Struktur."""
    ckpt_dir = os.path.join(CKPT_ROOT, experiment, f"seed_{seed}", f"step_{step}")
    model_dst = os.path.join(ckpt_dir, "model.pt")

    if os.path.exists(model_dst):
        print(f"  [SKIP] Bereits migriert: {experiment}/seed_{seed}/step_{step}")
        return

    if not os.path.exists(src_model):
        print(f"  [WARN] Quelldatei nicht gefunden: {src_model}")
        return

    print(f"  [MIGS] {os.path.basename(src_model)}")
    print(f"         -> {experiment}/seed_{seed}/step_{step}/")
    if note:
        print(f"         Hinweis: {note}")

    if dry_run:
        return

    os.makedirs(ckpt_dir, exist_ok=True)

    # Modell kopieren (nicht verschieben — Original in results/ bleibt erhalten)
    shutil.copy2(src_model, model_dst)

    # SHA256
    sha = _sha256_file(model_dst)
    with open(os.path.join(ckpt_dir, "sha256.txt"), "w", encoding="utf-8") as f:
        f.write(sha + "\n")

    # config.yaml
    yaml_str = "# Konfiguration (migriert aus bestehendem Lauf)\n" + _to_yaml(config)
    with open(os.path.join(ckpt_dir, "config.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml_str + "\n")

    # metrics.json
    with open(os.path.join(ckpt_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # routing_stats.json
    if routing_stats:
        with open(os.path.join(ckpt_dir, "routing_stats.json"), "w", encoding="utf-8") as f:
            json.dump(routing_stats, f, indent=2, ensure_ascii=False)

    # metadata.json
    import torch
    code_ver = _code_version()
    meta = {
        "experiment":       experiment,
        "seed":             seed,
        "step":             step,
        "timestamp":        datetime.now().isoformat(),
        "migrated_from":    src_model,
        "migration_note":   note,
        "code_version":     code_ver,
        "sha256_model":     sha,
        "python":           sys.version.split()[0],
        "torch":            torch.__version__,
        "ckpt_dir":         ckpt_dir,
    }
    with open(os.path.join(ckpt_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # latest.json
    parent = os.path.dirname(ckpt_dir)
    with open(os.path.join(parent, "latest.json"), "w", encoding="utf-8") as f:
        json.dump({"step": step, "path": ckpt_dir,
                   "timestamp": meta["timestamp"]}, f, indent=2)

    # val_loss aus metrics ermitteln und best_by_val_loss setzen
    val_loss = metrics.get("final_loss") or metrics.get("val_loss")
    if val_loss is not None:
        best_path = os.path.join(parent, "best_by_val_loss.json")
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump({"step": step, "val_loss": val_loss,
                       "path": ckpt_dir,
                       "timestamp": meta["timestamp"]}, f, indent=2)

    # Globales Registry
    _append_registry({
        "experiment": experiment,
        "seed":       seed,
        "step":       step,
        "timestamp":  meta["timestamp"],
        "sha256":     sha,
        "val_loss":   val_loss,
        "path":       ckpt_dir,
        "migrated":   True,
    })

    print(f"         SHA256: {sha}")
    print(f"         [OK]")


def main(dry_run: bool = False):
    results_dir = os.path.join(ROOT, "results")

    print("=" * 70)
    print("CHECKPOINT-MIGRATION")
    print("Kopiert bestehende Modelle in den versionierten Workspace.")
    print("Originale in results/ werden nicht veraendert.")
    if dry_run:
        print("DRY-RUN: Keine Aenderungen werden vorgenommen.")
    print("=" * 70)
    print()

    # -------------------------------------------------------------------------
    # 1. Fruehe synthetische Experimente (exp_all.py / run_demo.py)
    #    model_C_routed_s0.pt — 4 Iterationen, 1.67M Parameter, synthetische Regime
    # -------------------------------------------------------------------------
    src = os.path.join(results_dir, "model_C_routed_s0.pt")
    # Metriken aus C_routed_s0_end.json (das vollstaendigste)
    c_json_path = os.path.join(results_dir, "C_routed_s0_end.json")
    c_metrics = {"final_loss": 0.5497, "note": "Synthetische Regime, 4 Iterationen"}
    c_routing = None
    if os.path.exists(c_json_path):
        with open(c_json_path, encoding="utf-8") as f:
            c_data = json.load(f)
        final = c_data.get("final", {})
        c_metrics = {
            "final_loss": final.get("final_loss"),
            "loss_per_iter": final.get("loss_per_iter"),
        }

    migrate_one(
        src_model   = src,
        experiment  = "synthetic_exp_initial",
        seed        = 0,
        step        = 2000,  # Trainingsschritte aus fruehen Experimenten (TrainConfig-Default)
        config      = {
            "model": {
                "vocab_size": 16,
                "d_model":    128,
                "n_blocks":   24,
                "k_active":   4,
                "routed_iters": 4,
                "key_dim":    32,
                "router_noise_std": 0.1,
            },
            "training": {
                "steps": 2000,
                "bs":    64,
                "seq_len": 64,
                "dataset": "synthetic_4regimes",
            },
        },
        metrics       = c_metrics,
        routing_stats = c_routing,
        dry_run       = dry_run,
        note          = "Fruehe synthetische Experimente (run_demo / exp_all). "
                        "Architektur kleiner als TinyStories-Modelle.",
    )

    # -------------------------------------------------------------------------
    # 2. Phase-2-Modell TinyStories (via competence_centers_exp.py)
    #    competence_b64k4R6_s0_model.pt — 3000 Schritte, kein Diversity
    #    Ausgabe: L1=3.659  Lfin=3.658  bei Schritt 3000
    # -------------------------------------------------------------------------
    src = os.path.join(results_dir, "competence_b64k4R6_s0_model.pt")

    # Vollstaendige Metriken aus dem Kompetenz-Analyse-Lauf
    comp_json = os.path.join(results_dir, "competence_b64k4R6_s0.json")
    comp_metrics = {
        "final_loss":    3.658,
        "L1":            3.659,
        "Lfin":          3.658,
        "training_steps": 3000,
        "note": ("Phase-2-Training innerhalb von competence_centers_exp.py. "
                 "Kein Diversity-Zwang."),
    }
    comp_routing = None
    if os.path.exists(comp_json):
        with open(comp_json, encoding="utf-8") as f:
            cd = json.load(f)
        comp_routing = {
            "n_samples_per_cat":     cd.get("n_samples"),
            "jaccard_r2":            cd.get("jaccard_r2"),
            "mi_per_iter":           cd.get("mi_per_iter"),
            "normal_loss_per_cat":   cd.get("normal_loss"),
            "clf_accuracy":          cd.get("clf_accuracy"),
        }
        comp_metrics["competence_analysis"] = {
            "clf_accuracy_best":     max(cd["clf_accuracy"].values()) if cd.get("clf_accuracy") else None,
            "categories":            cd.get("categories"),
        }

    migrate_one(
        src_model   = src,
        experiment  = "tinystories_phase2",
        seed        = 0,
        step        = 3000,
        config      = {
            "model": {
                "vocab_size":       8000,
                "d_model":          256,
                "block_hidden":     512,
                "n_heads":          4,
                "context_layers":   1,
                "max_len":          256,
                "n_blocks":         64,
                "k_active":         4,
                "routed_iters":     6,
                "key_dim":          64,
                "router_noise_std": 0.3,
                "coord_dim":        3,
            },
            "training": {
                "steps":        3000,
                "bs":           32,
                "seq_len":      128,
                "lr":           0.002,
                "weight_decay": 0.01,
                "warmup":       200,
                "div_w":        0.0,
                "coord_w":      0.0,
                "diverse":      False,
                "loss_weighting": "end",
                "lb_loss_weight": 0.01,
            },
            "data": {
                "dataset":    "tinystories",
                "max_docs":   20000,
                "vocab_size": 8000,
                "seq_len":    128,
            },
        },
        metrics       = comp_metrics,
        routing_stats = comp_routing,
        dry_run       = dry_run,
        note          = ("Trainiert durch competence_centers_exp.py (Exp4). "
                         "Identische Konfiguration zu tinystories_exp.py Phase 2. "
                         "Phase-3-Modell aus tinystories_exp.py nicht verfuegbar "
                         "(Training erfolgte vor Einfuehrung des Checkpoint-Systems)."),
    )

    # -------------------------------------------------------------------------
    # 3. Hinweis auf fehlende Phase-3-Modelle
    # -------------------------------------------------------------------------
    phase3_json = os.path.join(results_dir,
                               "tinystories_b64k4R6_s0_div0.0_crd0.05_diverse.json")
    if os.path.exists(phase3_json) and not dry_run:
        lost_path = os.path.join(CKPT_ROOT, "tinystories_phase3",
                                 "seed_0", "MODELL_NICHT_VERFUEGBAR.txt")
        os.makedirs(os.path.dirname(lost_path), exist_ok=True)
        with open(lost_path, "w", encoding="utf-8") as f:
            f.write(
                "Phase-3-Modellgewichte sind nicht verfuegbar.\n\n"
                "Das Modell wurde trainiert bevor das Checkpoint-System eingefuehrt wurde.\n"
                "Ergebnisse sind in:\n"
                "  results/tinystories_b64k4R6_s0_div0.0_crd0.05_diverse.json\n\n"
                "Zum Wiederherstellen:\n"
                "  python -m experiments.tinystories_exp \\\n"
                "    --steps 3000 --diverse --coord_w 0.05 --device cuda\n\n"
                "Dies speichert das Modell dann automatisch als:\n"
                "  checkpoints/tinystories_phase3/seed_0/step_3000/\n"
            )
        print("\n  [INFO] Phase-3-Modell: Gewichte nicht verfuegbar.")
        print("         Hinweisdatei erstellt: checkpoints/tinystories_phase3/seed_0/")
        print("         Zum Wiederherstellen Phase-3-Training erneut ausfuehren.")

    print()
    print("Migration abgeschlossen.")
    print(f"Checkpoint-Verzeichnis: {CKPT_ROOT}")

    # Abschliessende Uebersicht
    if not dry_run:
        from rblm.checkpoint import print_registry
        print()
        print_registry()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migriert bestehende Modell-Checkpoints in den versionierten Workspace.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Nur anzeigen, keine Dateien kopieren")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
