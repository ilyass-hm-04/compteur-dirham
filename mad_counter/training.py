"""Entraînement, comparaison et évaluation de détecteurs YOLO."""

import csv
import shutil
from pathlib import Path

import yaml

from .labels import CLASS_NAMES


def _read_dataset_config(data_path):
    data_path = Path(data_path).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Configuration dataset introuvable : {data_path}")

    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Configuration YAML invalide : {data_path}")

    names = config.get("names")
    if isinstance(names, dict):
        actual_names = [names[key] for key in sorted(names, key=lambda key: int(key))]
    elif isinstance(names, list):
        actual_names = names
    else:
        raise ValueError("Le fichier data.yaml doit définir les noms des classes.")

    if tuple(actual_names) != CLASS_NAMES:
        raise ValueError(
            "Les classes de data.yaml doivent correspondre exactement à l'ordre défini "
            f"dans le projet : {list(CLASS_NAMES)}"
        )

    root = Path(config.get("path", data_path.parent))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()

    for split in ("train", "val"):
        split_path = config.get(split)
        if not split_path:
            raise ValueError(f"Le split '{split}' doit être défini dans data.yaml.")
        paths = split_path if isinstance(split_path, list) else [split_path]
        for item in paths:
            image_dir = Path(item)
            if not image_dir.is_absolute():
                image_dir = root / image_dir
            if not image_dir.is_dir() or not any(image_dir.iterdir()):
                raise FileNotFoundError(f"Images du split '{split}' absentes : {image_dir}")
            labels_dir = image_dir.parent / "labels"
            if not labels_dir.is_dir() or not any(labels_dir.glob("*.txt")):
                raise FileNotFoundError(
                    f"Annotations YOLO absentes pour le split '{split}' : {labels_dir}"
                )

    return data_path


def train_and_compare(
    data_path,
    candidates,
    epochs=60,
    image_size=640,
    batch_size=16,
    device="auto",
    output_dir="runs",
):
    """Entraîne les modèles demandés et sauvegarde le meilleur par mAP50-95."""
    from ultralytics import YOLO

    data_path = _read_dataset_config(data_path)
    output_dir = Path(output_dir).resolve()
    runs_dir = output_dir / "train"
    runs_dir.mkdir(parents=True, exist_ok=True)
    device_arg = None if device == "auto" else device
    rows = []

    for weights in candidates:
        run_name = Path(weights).stem
        print(f"\n===== Entraînement de {run_name} =====")
        model = YOLO(weights)
        model.train(
            data=str(data_path),
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            device=device_arg,
            project=str(runs_dir),
            name=run_name,
            exist_ok=True,
            patience=15,
            hsv_h=0.015,
            hsv_s=0.6,
            hsv_v=0.4,
            degrees=180,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            scale=0.5,
            plots=True,
            verbose=False,
        )

        best_weights = runs_dir / run_name / "weights" / "best.pt"
        if not best_weights.is_file():
            raise FileNotFoundError(
                f"Poids best.pt manquants après entraînement : {best_weights}"
            )

        trained_model = YOLO(str(best_weights))
        from .inference import _validate_model_names

        _validate_model_names(trained_model.names)
        metrics = trained_model.val(
            data=str(data_path),
            split="val",
            imgsz=image_size,
            device=device_arg,
            verbose=False,
            plots=False,
        )
        rows.append(
            {
                "model": run_name,
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
                "mAP50": float(metrics.box.map50),
                "mAP50-95": float(metrics.box.map),
                "inference_ms": float(metrics.speed["inference"]),
                "weights": str(best_weights),
            }
        )

    rows.sort(key=lambda row: row["mAP50-95"], reverse=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "model_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    best_destination = models_dir / "best_mad_detector.pt"
    shutil.copy2(rows[0]["weights"], best_destination)

    print("\nClassement par mAP50-95 :")
    for row in rows:
        print(
            f"{row['model']}: mAP50-95={row['mAP50-95']:.4f}, "
            f"mAP50={row['mAP50']:.4f}, precision={row['precision']:.4f}, "
            f"recall={row['recall']:.4f}"
        )
    print(f"Meilleur modèle : {best_destination}")
    print(f"Comparaison : {comparison_path}")
    return rows


def evaluate_model(weights, data_path, image_size=640, device="auto"):
    """Évalue un modèle existant sur le split de validation."""
    from ultralytics import YOLO

    data_path = _read_dataset_config(data_path)
    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Poids introuvables : {weights}")

    model = YOLO(str(weights))
    from .inference import _validate_model_names

    _validate_model_names(model.names)
    device_arg = None if device == "auto" else device
    metrics = model.val(
        data=str(data_path),
        split="val",
        imgsz=image_size,
        device=device_arg,
        verbose=False,
        plots=True,
    )
    print(
        f"Précision: {metrics.box.mp:.4f} | Rappel: {metrics.box.mr:.4f} | "
        f"mAP50: {metrics.box.map50:.4f} | mAP50-95: {metrics.box.map:.4f}"
    )
    return metrics
