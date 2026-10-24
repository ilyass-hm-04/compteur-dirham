"""Prédiction sur une image, une vidéo ou une caméra locale."""

from collections import Counter
from pathlib import Path

from .labels import CLASS_NAMES, CLASS_VALUES_DH


def _validate_model_names(names):
    if isinstance(names, dict):
        normalized = {int(class_id): name for class_id, name in names.items()}
    else:
        normalized = dict(enumerate(names))

    expected_ids = set(range(len(CLASS_NAMES)))
    expected_names = dict(enumerate(CLASS_NAMES))
    if normalized != expected_names or set(normalized) != expected_ids:
        received = set(normalized.values())
        missing = sorted(set(CLASS_NAMES) - received)
        unexpected = sorted(received - set(CLASS_NAMES))
        raise ValueError(
            "Les classes des poids ne correspondent pas à l'ordre du projet. "
            f"Manquantes : {missing or 'aucune'} ; inattendues : {unexpected or 'aucune'} ; "
            f"identifiants : {sorted(normalized)}."
        )
    return normalized


def annotate_frame(model, frame, confidence=0.45, image_size=640):
    """Retourne l'image annotée, le total DH et le comptage par classe."""
    import cv2

    result = model.predict(
        frame, conf=confidence, imgsz=image_size, verbose=False
    )[0]
    class_names = _validate_model_names(result.names)
    counts = Counter()
    total_cents = 0

    for box in result.boxes:
        class_id = int(box.cls.item())
        name = class_names[class_id]
        counts[name] += 1
        total_cents += round(CLASS_VALUES_DH[name] * 100)

    annotated = result.plot()
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 48), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        f"TOTAL: {total_cents / 100:.2f} DH  ({sum(counts.values())} objets)",
        (10, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    return annotated, total_cents / 100, dict(counts)


def predict_source(
    model,
    source,
    output=None,
    confidence=0.45,
    image_size=640,
    show=True,
):
    """Traite une image, une vidéo, ou un index de webcam."""
    import cv2

    source = str(source)
    is_camera = source.isdecimal()
    source_path = None if is_camera else Path(source)

    if not is_camera and not source_path.is_file():
        raise FileNotFoundError(f"Source introuvable : {source_path}")

    image_suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    if source_path is not None and source_path.suffix.lower() in image_suffixes:
        frame = cv2.imread(str(source_path))
        if frame is None:
            raise ValueError(f"Impossible de lire l'image : {source_path}")
        annotated, total, counts = annotate_frame(model, frame, confidence, image_size)
        destination = Path(output) if output else Path("output") / f"{source_path.stem}_detected.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), annotated):
            raise OSError(f"Impossible d'enregistrer le résultat : {destination}")
        print(f"Total détecté : {total:.2f} DH ({sum(counts.values())} objets)")
        print(f"Image annotée : {destination}")
        if show:
            cv2.imshow("Compteur MAD - image", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    capture_source = int(source) if is_camera else str(source_path)
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Impossible d'ouvrir la caméra/vidéo : {source}")

    writer = None
    destination = Path(output) if output else (
        Path("output") / f"{source_path.stem}_detected.mp4"
        if source_path is not None
        else None
    )
    frame_count = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                if frame_count == 0:
                    raise RuntimeError(f"Aucune image lisible dans la source : {source}")
                break

            annotated, total, counts = annotate_frame(model, frame, confidence, image_size)
            frame_count += 1

            if destination is not None:
                if writer is None:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        str(destination),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise OSError(f"Impossible de créer la vidéo : {destination}")
                writer.write(annotated)

            if show:
                cv2.imshow("Compteur MAD (q pour quitter)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        print(f"{frame_count} images traitées. Total du dernier instant : {total:.2f} DH.")
        if destination is not None:
            print(f"Vidéo annotée : {destination}")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
