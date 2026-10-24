"""Chargement de poids Ultralytics depuis des chemins Windows."""

import shutil
import tempfile
from pathlib import Path


def load_yolo_model(weights):
    """Charge un checkpoint même si son chemin contient une apostrophe."""
    from ultralytics import YOLO

    weights = Path(weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Poids introuvables : {weights}")
    if "'" not in str(weights):
        return YOLO(str(weights))

    with tempfile.TemporaryDirectory(prefix="mad_counter_") as temporary_dir:
        safe_path = Path(temporary_dir) / "weights.pt"
        shutil.copy2(weights, safe_path)
        return YOLO(str(safe_path))
