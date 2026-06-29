"""Versioniertes Checkpoint-Management.

Verzeichnisstruktur:
  checkpoints/
    {experiment}/          z.B. "tinystories_phase2", "tinystories_phase3"
      seed_{seed}/
        step_{step}/
          model.pt           Modell-Gewichte (state_dict)
          config.yaml        Vollstaendige Konfiguration (Modell + Training + Daten)
          metrics.json       Finale Metriken (Loss, Gini, Jaccard, ...)
          routing_stats.json Routing-Statistiken (optional)
          metadata.json      Experiment-Metadaten (Zeit, Code-Version, ...)
          sha256.txt         SHA256-Hash von model.pt (Integritaetscheck)
        latest.json          Zeiger auf neuesten Checkpoint (wird ueberschrieben)
        best_by_val_loss.json Zeiger auf besten Checkpoint nach Verlust
    registry.json            Globales Register aller gespeicherten Checkpoints

Wichtige Invarianten:
  - model.pt wird NIEMALS ueberschrieben (schuetzt historische Laeufe)
  - SHA256 wird beim Laden verifiziert
  - registry.json akkumuliert alle Checkpoints (append-only)
"""
from __future__ import annotations
import hashlib, json, os, sys
from datetime import datetime


CKPT_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
REGISTRY_PATH = os.path.join(CKPT_ROOT, "registry.json")


# =============================================================================
# Internes Hilfsprogramm
# =============================================================================

def _sha256_file(path: str) -> str:
    """SHA256-Hash einer Datei (blockweise fuer grosse Dateien)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _code_version() -> str:
    """Gibt Git-Commit-Hash oder einen Quellcode-Hash zurueck."""
    try:
        import subprocess
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True,
            cwd=os.path.dirname(CKPT_ROOT), timeout=5,
        )
        if r.returncode == 0:
            return "git:" + r.stdout.strip()
    except Exception:
        pass
    # Kein Git-Repo: Hash der zentralen Quelldateien
    h = hashlib.sha256()
    src_root = os.path.dirname(CKPT_ROOT)
    for rel in ["rblm/models.py", "rblm/router.py", "rblm/config.py",
                 "rblm/blocks.py"]:
        p = os.path.join(src_root, rel)
        if os.path.exists(p):
            with open(p, "rb") as f:
                h.update(f.read())
    return "src-hash:" + h.hexdigest()[:16]


# Minimaler YAML-Serialisierer (keine Abhaengigkeit von PyYAML)
def _yaml_scalar(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    s = str(v)
    # Anführungszeichen wenn noetig
    needs_quote = any(c in s for c in ':#{}[]|>\'",' ) or "\n" in s
    needs_quote = needs_quote or s.lower() in ("true", "false", "null", "yes", "no", "on", "off")
    return f'"{s}"' if needs_quote else s


def _to_yaml(obj, indent: int = 0) -> str:
    pad = "  " * indent
    if isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v:
                lines.append(f"{pad}{k}:")
                lines.append(_to_yaml(v, indent + 1))
            else:
                lines.append(f"{pad}{k}: {_yaml_scalar(v)}")
        return "\n".join(lines)
    if isinstance(obj, list):
        lines = []
        for item in obj:
            if isinstance(item, dict):
                first = True
                for k, v in item.items():
                    prefix = f"{pad}- " if first else f"{pad}  "
                    first = False
                    if isinstance(v, (dict, list)):
                        lines.append(f"{prefix}{k}:")
                        lines.append(_to_yaml(v, indent + 2))
                    else:
                        lines.append(f"{prefix}{k}: {_yaml_scalar(v)}")
            else:
                lines.append(f"{pad}- {_yaml_scalar(item)}")
        return "\n".join(lines)
    return pad + _yaml_scalar(obj)


# =============================================================================
# Registry
# =============================================================================

def _load_registry() -> list:
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def _append_registry(entry: dict):
    os.makedirs(CKPT_ROOT, exist_ok=True)
    registry = _load_registry()
    registry.append(entry)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


# =============================================================================
# Oeffentliche API
# =============================================================================

def save(
    model,
    experiment: str,
    config: dict,
    metrics: dict,
    routing_stats: dict | None = None,
    seed: int = 0,
    step: int = 0,
    val_loss: float | None = None,
) -> str:
    """
    Speichert einen Checkpoint unveraenderlich in den versionierten Workspace.

    Parameter:
      model        PyTorch-Modell (wird als state_dict gespeichert)
      experiment   Eindeutiger Name, z.B. "tinystories_phase2", "competence_exp4"
      config       dict mit Modell- und Trainingsparametern
      metrics      dict mit finalen Metriken (loss_per_iter, gini, jaccard, ...)
      routing_stats dict mit Routing-Statistiken (optional)
      seed         Random-Seed des Laufs
      step         Trainingsschritt
      val_loss     Validierungsverlust fuer best_by_val_loss-Markierung

    Gibt den Checkpoint-Verzeichnispfad zurueck.
    Wirft keine Exception wenn der Checkpoint schon existiert (Warnung statt Fehler).
    """
    import torch

    ckpt_dir = os.path.join(CKPT_ROOT, experiment, f"seed_{seed}", f"step_{step}")
    model_path = os.path.join(ckpt_dir, "model.pt")

    # Niemals ueberschreiben
    if os.path.exists(model_path):
        print(f"[CKPT] Checkpoint existiert bereits — unveraendert gelassen: {ckpt_dir}",
              flush=True)
        return ckpt_dir

    os.makedirs(ckpt_dir, exist_ok=True)

    # 1. Modell-Gewichte
    torch.save(model.state_dict(), model_path)

    # 2. SHA256-Hash
    sha = _sha256_file(model_path)
    with open(os.path.join(ckpt_dir, "sha256.txt"), "w", encoding="utf-8") as f:
        f.write(sha + "\n")

    # 3. Konfiguration als YAML
    yaml_str = "# Checkpoint-Konfiguration (Modell + Training + Daten)\n" + _to_yaml(config)
    with open(os.path.join(ckpt_dir, "config.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml_str + "\n")

    # 4. Metriken als JSON
    with open(os.path.join(ckpt_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 5. Routing-Statistiken (optional)
    if routing_stats is not None:
        with open(os.path.join(ckpt_dir, "routing_stats.json"), "w", encoding="utf-8") as f:
            json.dump(routing_stats, f, indent=2, ensure_ascii=False)

    # 6. Metadaten
    code_ver = _code_version()
    metadata = {
        "experiment":   experiment,
        "seed":         seed,
        "step":         step,
        "timestamp":    datetime.now().isoformat(),
        "code_version": code_ver,
        "sha256_model": sha,
        "python":       sys.version.split()[0],
        "torch":        torch.__version__,
        "ckpt_dir":     ckpt_dir,
    }
    if val_loss is not None:
        metadata["val_loss"] = val_loss
    with open(os.path.join(ckpt_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 7. latest.json (Zeiger — wird ueberschrieben, Model-Datei aber nicht)
    parent = os.path.dirname(ckpt_dir)
    with open(os.path.join(parent, "latest.json"), "w", encoding="utf-8") as f:
        json.dump({"step": step, "path": ckpt_dir,
                   "timestamp": metadata["timestamp"]}, f, indent=2)

    # 8. best_by_val_loss.json (nur wenn angegeben und besser als bisheriger Bestwert)
    if val_loss is not None:
        best_path = os.path.join(parent, "best_by_val_loss.json")
        current_best = None
        if os.path.exists(best_path):
            with open(best_path, encoding="utf-8") as f:
                current_best = json.load(f).get("val_loss")
        if current_best is None or val_loss < current_best:
            with open(best_path, "w", encoding="utf-8") as f:
                json.dump({"step": step, "val_loss": val_loss,
                           "path": ckpt_dir,
                           "timestamp": metadata["timestamp"]}, f, indent=2)
            print(f"[CKPT] Neuer Bestwert (val_loss={val_loss:.4f}): {ckpt_dir}",
                  flush=True)

    # 9. Globales Registry (append-only)
    _append_registry({
        "experiment": experiment,
        "seed":       seed,
        "step":       step,
        "timestamp":  metadata["timestamp"],
        "sha256":     sha,
        "val_loss":   val_loss,
        "path":       ckpt_dir,
    })

    print(f"[CKPT] Gespeichert: {ckpt_dir}", flush=True)
    print(f"[CKPT] SHA256:      {sha}", flush=True)
    return ckpt_dir


def load(path: str, model, device: str = "cpu", verify: bool = True):
    """
    Laedt Modell-Gewichte aus einem Checkpoint-Verzeichnis oder einer .pt-Datei.
    Verifiziert SHA256 standardmaessig.

    Parameter:
      path    Checkpoint-Verzeichnis oder direkte model.pt-Datei
      model   Initialisiertes Modell (Architektur muss passen)
      device  Geraet: "cpu" oder "cuda"
      verify  SHA256-Verifizierung (empfohlen)
    """
    import torch

    if path.endswith(".pt"):
        model_path = path
        ckpt_dir   = os.path.dirname(path)
    else:
        model_path = os.path.join(path, "model.pt")
        ckpt_dir   = path

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {model_path}")

    # SHA256-Verifizierung
    if verify:
        sha_file = os.path.join(ckpt_dir, "sha256.txt")
        if os.path.exists(sha_file):
            with open(sha_file, encoding="utf-8") as f:
                expected = f.read().strip()
            actual = _sha256_file(model_path)
            if actual != expected:
                raise RuntimeError(
                    f"SHA256-Mismatch — Checkpoint koennte beschaedigt sein!\n"
                    f"  Datei:    {model_path}\n"
                    f"  Erwartet: {expected}\n"
                    f"  Aktuell:  {actual}"
                )
            print(f"[CKPT] SHA256 OK: {actual[:16]}...", flush=True)
        else:
            print(f"[CKPT] Warnung: Keine sha256.txt gefunden — keine Integritaetspruefung.",
                  flush=True)

    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)

    # Metadaten anzeigen falls vorhanden
    meta_path = os.path.join(ckpt_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        print(f"[CKPT] Geladen: {meta.get('experiment', '?')} "
              f"seed={meta.get('seed', '?')} "
              f"step={meta.get('step', '?')} "
              f"({meta.get('timestamp', '?')[:10]})",
              flush=True)
    else:
        print(f"[CKPT] Geladen: {model_path}", flush=True)

    return model


def get_latest(experiment: str, seed: int = 0) -> str | None:
    """Gibt den Pfad zum neuesten Checkpoint-Verzeichnis zurueck."""
    p = os.path.join(CKPT_ROOT, experiment, f"seed_{seed}", "latest.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f).get("path")
    return None


def get_best(experiment: str, seed: int = 0,
             metric: str = "val_loss") -> str | None:
    """Gibt den Pfad zum besten Checkpoint zurueck."""
    p = os.path.join(CKPT_ROOT, experiment, f"seed_{seed}",
                     f"best_by_{metric}.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f).get("path")
    return None


def list_checkpoints(experiment: str | None = None) -> list:
    """Gibt alle Eintraege aus dem globalen Registry zurueck."""
    reg = _load_registry()
    if experiment:
        reg = [r for r in reg if r["experiment"] == experiment]
    return reg


def print_registry():
    """Gibt eine lesbare Uebersicht aller Checkpoints aus."""
    reg = _load_registry()
    if not reg:
        print("[CKPT] Registry ist leer.", flush=True)
        return
    print(f"\n{'Experiment':30s}  {'Seed':4s}  {'Step':6s}  {'val_loss':9s}  "
          f"{'Datum':10s}  SHA256 (kurz)")
    print("-" * 85)
    for r in reg:
        ts = r.get("timestamp", "")[:10]
        vl = f"{r['val_loss']:.4f}" if r.get("val_loss") is not None else "—"
        sha = r.get("sha256", "")[:12] + "..."
        print(f"  {r['experiment']:28s}  {r['seed']:4d}  {r['step']:6d}  "
              f"{vl:9s}  {ts:10s}  {sha}")
