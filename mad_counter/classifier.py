"""Classification d'une seule pièce ou d'un seul billet par image."""

import hashlib
import json
import random
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import yaml

from .labels import CLASS_NAMES, CLASS_VALUES_DH

FOLDER_TO_CLASS = {
    "0.1": "coin_10c",
    "0.2": "coin_20c",
    "0.5": "coin_50c",
    "1": "coin_1dh",
    "2": "coin_2dh",
    "5": "coin_5dh",
    "10": "coin_10dh",
    "20": "note_20dh",
    "50": "note_50dh",
    "100": "note_100dh",
    "200": "note_200dh",
}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _read_cv_image(image_path, cv2):
    import numpy as np

    encoded = np.fromfile(str(image_path), dtype=np.uint8)
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def prepare_classification_data(source, destination, seed=42):
    """Déduplique les photos puis crée des dossiers train/val/test."""
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Dossier de données introuvable : {source}")

    found = {name: [] for name in CLASS_NAMES}
    hashes = defaultdict(set)
    for folder_name, class_name in FOLDER_TO_CLASS.items():
        class_dir = source / folder_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Dossier de classe manquant : {class_dir}")
        for image in sorted(class_dir.iterdir()):
            if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
                digest = hashlib.sha256(image.read_bytes()).hexdigest()
                hashes[digest].add(class_name)
                found[class_name].append((digest, image))

    conflicts = [digest for digest, labels in hashes.items() if len(labels) > 1]
    if conflicts:
        raise ValueError(
            f"{len(conflicts)} image(s) identiques sont attribuées à plusieurs classes. "
            "Corrigez ces étiquettes avant l'entraînement."
        )

    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"Le dossier de sortie n'est pas vide : {destination}. "
            "Choisissez un autre --dataset-dir ou videz uniquement ce dossier généré."
        )

    randomizer = random.Random(seed)
    splits = {split: defaultdict(list) for split in ("train", "val", "test")}
    for class_name, images in found.items():
        unique = {}
        for digest, path in images:
            unique.setdefault(digest, path)
        samples = list(unique.values())
        if len(samples) < 3:
            raise ValueError(
                f"Classe {class_name} : 3 images uniques minimum requises, "
                f"{len(samples)} disponible(s)."
            )

        randomizer.shuffle(samples)
        test_count = max(1, round(len(samples) * 0.2))
        val_count = max(1, round((len(samples) - test_count) * 0.2))
        assignments = {
            "test": samples[:test_count],
            "val": samples[test_count : test_count + val_count],
            "train": samples[test_count + val_count :],
        }
        for split, selected in assignments.items():
            splits[split][class_name].extend(selected)

    for split, classes in splits.items():
        for class_name, images in classes.items():
            output_dir = destination / split / class_name
            output_dir.mkdir(parents=True, exist_ok=True)
            for index, image in enumerate(images):
                shutil.copy2(image, output_dir / f"{index:05d}{image.suffix.lower()}")

    counts = {
        split: sum(len(images) for images in classes.values())
        for split, classes in splits.items()
    }
    print(
        f"Dataset dédupliqué : {sum(counts.values())} images "
        f"(train={counts['train']}, val={counts['val']}, test={counts['test']})."
    )
    return destination


def prepare_detection_data(
    source,
    destination,
    image_size=416,
    seed=42,
    train_scenes=72,
    validation_scenes=36,
    test_scenes=72,
):
    """Build a YOLO detection dataset, with held-out synthetic multi-object scenes."""
    import cv2
    import numpy as np

    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if image_size < 64:
        raise ValueError("La taille des images de détection doit être d'au moins 64 pixels.")
    if min(train_scenes, validation_scenes, test_scenes) < 1:
        raise ValueError("Chaque split doit contenir au moins une scène synthétique.")
    if not source.is_dir():
        raise FileNotFoundError(f"Dossier de données introuvable : {source}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"Le dossier de sortie n'est pas vide : {destination}. "
            "Choisissez un autre --dataset-dir ou videz uniquement ce dossier généré."
        )

    randomizer = random.Random(seed)
    samples = {name: {} for name in CLASS_NAMES}
    hashes = defaultdict(set)
    for folder_name, class_name in FOLDER_TO_CLASS.items():
        class_dir = source / folder_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Dossier de classe manquant : {class_dir}")
        for image in sorted(class_dir.iterdir()):
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            hashes[digest].add(class_name)
            samples[class_name].setdefault(digest, image)

    conflicts = [digest for digest, names in hashes.items() if len(names) > 1]
    if conflicts:
        raise ValueError(
            f"{len(conflicts)} image(s) identiques sont attribuées à plusieurs classes. "
            "Corrigez ces étiquettes avant l'entraînement."
        )

    splits = {split: defaultdict(list) for split in ("train", "val", "test")}
    for class_name, unique_images in samples.items():
        images = list(unique_images.values())
        if len(images) < 3:
            raise ValueError(
                f"Classe {class_name} : 3 images uniques minimum requises, "
                f"{len(images)} disponible(s)."
            )
        randomizer.shuffle(images)
        test_count = max(1, round(len(images) * 0.2))
        val_count = max(1, round((len(images) - test_count) * 0.2))
        splits["test"][class_name].extend(images[:test_count])
        splits["val"][class_name].extend(images[test_count : test_count + val_count])
        splits["train"][class_name].extend(images[test_count + val_count :])

    split_class_ids = {
        split: {
            class_id: list(splits[split][class_name])
            for class_id, class_name in enumerate(CLASS_NAMES)
        }
        for split in ("train", "val", "test")
    }
    for class_id, class_name in enumerate(CLASS_NAMES):
        image_dir = destination / "train" / "images"
        label_dir = destination / "train" / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for index, image_path in enumerate(splits["train"][class_name]):
            image = _read_cv_image(image_path, cv2)
            if image is None:
                raise ValueError(f"Impossible de lire l'image source : {image_path}")
            image_name = f"{class_id}_{index:05d}.jpg"
            if not cv2.imwrite(str(image_dir / image_name), image):
                raise OSError(f"Impossible d'écrire l'image : {image_dir / image_name}")
            (label_dir / f"{Path(image_name).stem}.txt").write_text(
                f"{class_id} 0.5 0.5 1.0 1.0\n", encoding="utf-8"
            )

    _write_synthetic_scenes(
        split_class_ids["train"],
        destination / "train",
        "mosaic",
        train_scenes,
        image_size,
        randomizer,
    )
    _write_synthetic_scenes(
        split_class_ids["val"],
        destination / "val",
        "scene",
        validation_scenes,
        image_size,
        randomizer,
    )
    _write_synthetic_scenes(
        split_class_ids["test"],
        destination / "test",
        "scene",
        test_scenes,
        image_size,
        randomizer,
    )

    data_path = destination / "data.yaml"
    data_path.write_text(
        yaml.safe_dump(
            {
                "path": str(destination),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "names": list(CLASS_NAMES),
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    train_count = (
        sum(len(images) for images in splits["train"].values()) + train_scenes
    )
    print(
        f"Dataset YOLO créé : {train_count} images train dont {train_scenes} scènes à 4 objets, "
        f"{validation_scenes} scènes val et {test_scenes} scènes test."
    )
    print("Chaque objet source reçoit une boîte initiale sur tout le cadre de sa photo.")
    return data_path


def _write_synthetic_scenes(class_images, split_dir, prefix, scene_count, image_size, rng):
    import cv2
    import numpy as np

    image_dir = split_dir / "images"
    label_dir = split_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    cell_width = image_size // 2
    cell_height = image_size // 2
    margin_x = max(1, round(image_size * 0.025))
    margin_y = max(1, round(image_size * 0.025))
    max_width = cell_width - 2 * margin_x
    max_height = cell_height - 2 * margin_y
    class_ids = [class_id for class_id, images in class_images.items() if images]

    for scene_index in range(scene_count):
        canvas = np.full((image_size, image_size, 3), 242, dtype=np.uint8)
        chosen_classes = rng.sample(class_ids, k=min(4, len(class_ids)))
        labels = []
        for quadrant, class_id in enumerate(chosen_classes):
            image_path = rng.choice(class_images[class_id])
            image = _read_cv_image(image_path, cv2)
            if image is None:
                raise ValueError(f"Impossible de lire l'image source : {image_path}")
            original_height, original_width = image.shape[:2]
            scale = min(max_width / original_width, max_height / original_height)
            scale *= rng.uniform(0.72, 0.96)
            width = max(1, round(original_width * scale))
            height = max(1, round(original_height * scale))
            resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

            row, column = divmod(quadrant, 2)
            cell_x = column * cell_width
            cell_y = row * cell_height
            max_x = max(margin_x, cell_width - width - margin_x)
            max_y = max(margin_y, cell_height - height - margin_y)
            x_min = cell_x + rng.randint(margin_x, max_x)
            y_min = cell_y + rng.randint(margin_y, max_y)
            x_max = x_min + width
            y_max = y_min + height
            canvas[y_min:y_max, x_min:x_max] = resized

            center_x = (x_min + x_max) / (2 * image_size)
            center_y = (y_min + y_max) / (2 * image_size)
            labels.append(
                f"{class_id} {center_x:.6f} {center_y:.6f} "
                f"{width / image_size:.6f} {height / image_size:.6f}"
            )

        image_name = f"{prefix}_{scene_index:04d}.jpg"
        if not cv2.imwrite(str(image_dir / image_name), canvas):
            raise OSError(f"Impossible d'écrire la scène : {image_dir / image_name}")
        (label_dir / f"{Path(image_name).stem}.txt").write_text(
            "\n".join(labels) + "\n", encoding="utf-8"
        )


def _evaluate_synthetic_counting(model, dataset_dir, confidence, image_size):
    image_dir = Path(dataset_dir) / "test" / "images"
    label_dir = Path(dataset_dir) / "test" / "labels"
    images = sorted(image_dir.glob("*.jpg"))
    results = model.predict(
        [str(image) for image in images],
        imgsz=image_size,
        conf=confidence,
        verbose=False,
    )
    exact_count = 0
    exact_denominations = 0
    detected_objects = 0
    absolute_total_errors = []

    for image, result in zip(images, results):
        true_ids = [
            int(line.split()[0])
            for line in (label_dir / f"{image.stem}.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        predicted_ids = [int(class_id) for class_id in result.boxes.cls.tolist()]
        exact_count += len(predicted_ids) == len(true_ids)
        exact_denominations += sorted(predicted_ids) == sorted(true_ids)
        detected_objects += len(predicted_ids)
        expected_cents = sum(
            round(CLASS_VALUES_DH[CLASS_NAMES[class_id]] * 100)
            for class_id in true_ids
        )
        detected_cents = sum(
            round(CLASS_VALUES_DH[CLASS_NAMES[class_id]] * 100)
            for class_id in predicted_ids
        )
        absolute_total_errors.append(abs(expected_cents - detected_cents) / 100)

    scene_count = len(images)
    return {
        "confidence_threshold": confidence,
        "test_scenes": scene_count,
        "exact_object_count_scenes": exact_count,
        "exact_denomination_count_scenes": exact_denominations,
        "mean_detected_objects_per_scene": detected_objects / scene_count,
        "mean_absolute_total_error_dh": sum(absolute_total_errors) / scene_count,
    }


def train_detector_from_classification(
    source,
    weights="yolo11n.pt",
    epochs=40,
    image_size=416,
    batch_size=8,
    device="cpu",
    output_dir="runs/detection",
    dataset_dir="runs/detection_dataset",
    model_output="models/best_mad_detector.pt",
    train_scenes=72,
    validation_scenes=36,
    test_scenes=72,
):
    """Train and evaluate a detector using synthetic multi-object scenes."""
    from .model_loading import load_yolo_model

    data_path = prepare_detection_data(
        source,
        dataset_dir,
        image_size=image_size,
        train_scenes=train_scenes,
        validation_scenes=validation_scenes,
        test_scenes=test_scenes,
    )
    run_dir = Path(output_dir).resolve()
    if "'" in str(data_path):
        temporary_dir = tempfile.TemporaryDirectory(prefix="mad_counter_detect_")
        safe_dataset_dir = Path(temporary_dir.name) / "dataset"
        shutil.copytree(data_path.parent, safe_dataset_dir)
        data_path = safe_dataset_dir / "data.yaml"
        data_config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
        data_config["path"] = str(safe_dataset_dir)
        data_path.write_text(
            yaml.safe_dump(data_config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    from ultralytics import YOLO

    model = load_yolo_model(weights) if Path(weights).is_file() else YOLO(weights)
    model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        workers=0 if device == "cpu" else 4,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        patience=10,
        seed=42,
        deterministic=True,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        degrees=180,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=5,
        mixup=0.1,
        scale=0.5,
        plots=True,
        verbose=True,
    )

    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"Poids entraînés introuvables : {best_weights}")

    model = load_yolo_model(best_weights)
    actual_names = {model.names[index] for index in sorted(model.names)}
    if actual_names != set(CLASS_NAMES):
        raise ValueError(f"Classes inattendues après entraînement : {sorted(actual_names)}")

    metrics = model.val(
        data=str(data_path),
        split="test",
        imgsz=image_size,
        device=device,
        plots=True,
    )
    counting_metrics = _evaluate_synthetic_counting(
        model, data_path.parent, confidence=0.40, image_size=image_size
    )
    destination = Path(model_output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, destination)

    from .inference import annotate_frame

    import cv2

    demo_path = Path("output") / "multiobject_demo.jpg"
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path = data_path.parent / "test" / "images" / "scene_0000.jpg"
    frame = _read_cv_image(sample_path, cv2)
    if frame is None:
        raise ValueError(f"Scène de test illisible : {sample_path}")
    annotated, total, counts = annotate_frame(
        model, frame, confidence=counting_metrics["confidence_threshold"], image_size=image_size
    )
    if not cv2.imwrite(str(demo_path), annotated):
        raise OSError(f"Impossible d'enregistrer la démonstration : {demo_path}")

    metrics_path = destination.parent / "mad_detector_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "task": "multi-object Moroccan currency detection and frame-by-frame counting",
                "architecture": "YOLO11n",
                "initial_weights": Path(weights).name,
                "epochs": epochs,
                "image_size": image_size,
                "batch_size": batch_size,
                "device": device,
                "source_unique_images": sum(
                    len(class_images)
                    for class_images in _collect_class_samples(source).values()
                ),
                "synthetic_test_scenes": test_scenes,
                "synthetic_train_scenes": train_scenes,
                "synthetic_validation_scenes": validation_scenes,
                "synthetic_objects_per_scene": min(4, len(CLASS_NAMES)),
                "test_metrics": {
                    "precision": float(metrics.box.mp),
                    "recall": float(metrics.box.mr),
                    "map50": float(metrics.box.map50),
                    "map50_95": float(metrics.box.map),
                    "counting": counting_metrics,
                },
                "demo_scene_detections": counts,
                "demo_scene_total_dh": total,
                "limitations": [
                    "The classification dataset does not contain manually drawn object boxes.",
                    "Initial training boxes cover each source image; test scenes are synthetic image composites.",
                    "Real-scene performance needs manually annotated multi-object images.",
                    "Frame totals are not accumulated between video frames.",
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Détection multiobjets test synthétique : precision={metrics.box.mp:.4f}, "
        f"rappel={metrics.box.mr:.4f}, mAP50={metrics.box.map50:.4f}, "
        f"mAP50-95={metrics.box.map:.4f}."
    )
    print(
        f"Comptage à conf={counting_metrics['confidence_threshold']:.2f} : "
        f"{counting_metrics['exact_object_count_scenes']}/{counting_metrics['test_scenes']} "
        f"scènes avec le bon nombre d'objets, erreur moyenne du total "
        f"{counting_metrics['mean_absolute_total_error_dh']:.2f} DH."
    )
    print(f"Exemple compté : {counts}, total détecté {total:.2f} DH.")
    print(f"Modèle détecteur : {destination.resolve()}")
    print(f"Résultat multiobjets local : {demo_path.resolve()}")
    print(f"Métriques : {metrics_path.resolve()}")
    return destination, metrics


def _collect_class_samples(source):
    source = Path(source)
    collected = {name: {} for name in CLASS_NAMES}
    for folder_name, class_name in FOLDER_TO_CLASS.items():
        class_dir = source / folder_name
        for image in class_dir.iterdir():
            if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
                digest = hashlib.sha256(image.read_bytes()).hexdigest()
                collected[class_name].setdefault(digest, image)
    return collected


def train_classifier(
    source,
    weights="yolo11n-cls.pt",
    epochs=20,
    image_size=224,
    batch_size=16,
    device="cpu",
    output_dir="runs/classification",
    dataset_dir="runs/classification_dataset",
    model_output="models/mad_classifier.pt",
):
    """Entraîne YOLO-cls et évalue le modèle sur les images test réservées."""
    dataset_dir = prepare_classification_data(source, dataset_dir)
    run_dir = Path(output_dir).resolve()
    if "'" in str(dataset_dir.resolve()):
        temporary_dir = tempfile.TemporaryDirectory(prefix="mad_counter_data_")
        safe_dataset_dir = Path(temporary_dir.name) / "dataset"
        shutil.copytree(dataset_dir, safe_dataset_dir)
        dataset_dir = safe_dataset_dir

    from ultralytics import YOLO
    from .model_loading import load_yolo_model

    model = YOLO(weights)
    model.train(
        data=str(dataset_dir),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        workers=0 if device == "cpu" else 4,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        patience=5,
        seed=42,
        deterministic=True,
        plots=True,
        verbose=True,
    )

    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"Poids entraînés introuvables : {best_weights}")

    model = load_yolo_model(best_weights)
    actual_names = {model.names[index] for index in sorted(model.names)}
    if actual_names != set(CLASS_NAMES):
        raise ValueError(
            f"Classes inattendues après entraînement : {sorted(actual_names)}"
        )

    metrics = model.val(data=str(dataset_dir), split="test", imgsz=image_size)
    destination = Path(model_output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, destination)
    print(
        f"Précision top-1 sur le jeu de test réservé : {metrics.top1:.4f} "
        f"({metrics.top1 * 100:.1f} %)"
    )
    print(f"Précision top-5 : {metrics.top5:.4f}")
    print(f"Modèle sauvegardé : {destination.resolve()}")
    return destination, metrics


def classify_image(weights, source, image_size=224):
    """Classe une photo contenant un objet isolé et affiche la valeur prédite."""
    import cv2

    from .model_loading import load_yolo_model

    source = Path(source)
    if not source.is_file() or source.suffix.lower() not in IMAGE_SUFFIXES:
        raise FileNotFoundError(f"Image introuvable ou format non pris en charge : {source}")

    image = _read_cv_image(source, cv2)
    if image is None:
        raise ValueError(f"Impossible de lire l'image : {source}")

    model = load_yolo_model(weights)
    class_names = {model.names[index] for index in sorted(model.names)}
    if class_names != set(CLASS_NAMES):
        raise ValueError(f"Classes du modèle incompatibles : {sorted(class_names)}")

    result = model.predict(image, imgsz=image_size, verbose=False)[0]
    class_id = int(result.probs.top1)
    class_name = result.names[class_id]
    confidence = float(result.probs.top1conf)
    amount = CLASS_VALUES_DH[class_name]
    print(
        f"Prédiction : {class_name} — {amount:.2f} DH "
        f"(confiance {confidence:.1%})"
    )
    print("Le classificateur attend une photo recadrée avec un seul objet.")
    return class_name, amount, confidence
