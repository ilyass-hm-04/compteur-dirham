"""Classification d'une seule pièce ou d'un seul billet par image."""

import hashlib
import random
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

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

    image = cv2.imread(str(source))
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
