"""Interface en ligne de commande du compteur de dirhams."""

import argparse

DEFAULT_MODELS = ("yolov8n.pt", "yolov8s.pt", "yolo11n.pt", "yolo11s.pt")


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="mad-counter",
        description="Détecter des pièces et billets marocains et calculer leur total.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Entraîner et comparer plusieurs modèles YOLO.")
    train.add_argument("--data", default="dataset/data.yaml", help="Chemin vers le YAML YOLO.")
    train.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Poids YOLO à comparer.")
    train.add_argument("--epochs", type=int, default=60)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--device", default="auto", help="auto, cpu ou index GPU (ex. 0).")
    train.add_argument("--output-dir", default="runs")

    predict = commands.add_parser("predict", help="Prédire sur image, vidéo ou webcam.")
    predict.add_argument("--weights", default="models/best_mad_detector.pt")
    predict.add_argument("--source", required=True, help="Fichier image/vidéo ou numéro de caméra.")
    predict.add_argument("--output", help="Chemin de sortie annoté.")
    predict.add_argument("--conf", type=float, default=0.45)
    predict.add_argument("--imgsz", type=int, default=640)
    predict.add_argument("--no-show", action="store_true", help="Ne pas ouvrir de fenêtre d'affichage.")

    evaluate = commands.add_parser("evaluate", help="Évaluer un modèle sur le split validation.")
    evaluate.add_argument("--weights", default="models/best_mad_detector.pt")
    evaluate.add_argument("--data", default="dataset/data.yaml")
    evaluate.add_argument("--imgsz", type=int, default=640)
    evaluate.add_argument("--device", default="auto")

    export = commands.add_parser("export", help="Exporter un modèle au format ONNX ou TFLite.")
    export.add_argument("--weights", default="models/best_mad_detector.pt")
    export.add_argument("--format", choices=("onnx", "tflite"), default="onnx")
    export.add_argument("--imgsz", type=int, default=640)

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "train":
        if args.epochs < 1 or args.imgsz < 1 or args.batch < 1:
            parser.error("--epochs, --imgsz et --batch doivent être positifs.")
        from .training import train_and_compare

        train_and_compare(
            args.data,
            args.models,
            epochs=args.epochs,
            image_size=args.imgsz,
            batch_size=args.batch,
            device=args.device,
            output_dir=args.output_dir,
        )
    elif args.command == "predict":
        if not 0 < args.conf <= 1:
            parser.error("--conf doit être supérieur à 0 et inférieur ou égal à 1.")
        from ultralytics import YOLO
        from .inference import _validate_model_names, predict_source

        model = YOLO(args.weights)
        _validate_model_names(model.names)
        predict_source(
            model,
            args.source,
            output=args.output,
            confidence=args.conf,
            image_size=args.imgsz,
            show=not args.no_show,
        )
    elif args.command == "evaluate":
        from .training import evaluate_model

        evaluate_model(args.weights, args.data, args.imgsz, args.device)
    elif args.command == "export":
        from pathlib import Path
        from ultralytics import YOLO

        weights = Path(args.weights)
        if not weights.is_file():
            parser.error(f"Poids introuvables : {weights}")
        exported = YOLO(str(weights)).export(format=args.format, imgsz=args.imgsz)
        print(f"Modèle exporté : {exported}")


if __name__ == "__main__":
    main()
