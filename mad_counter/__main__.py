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
    predict.add_argument("--conf", type=float, default=0.40)
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

    train_classifier = commands.add_parser(
        "train-classifier",
        help="Entraîner un modèle de classification sur des photos classées par valeur.",
    )
    train_classifier.add_argument(
        "--source",
        required=True,
        help="Dossier d'images classées dans des sous-dossiers 0.1, 0.2, ..., 200.",
    )
    train_classifier.add_argument("--weights", default="yolo11n-cls.pt")
    train_classifier.add_argument("--epochs", type=int, default=20)
    train_classifier.add_argument("--imgsz", type=int, default=128)
    train_classifier.add_argument("--batch", type=int, default=16)
    train_classifier.add_argument("--device", default="cpu")
    train_classifier.add_argument("--output-dir", default="runs/classification")
    train_classifier.add_argument("--dataset-dir", default="runs/classification_dataset")
    train_classifier.add_argument("--model-output", default="models/mad_classifier.pt")

    train_detector = commands.add_parser(
        "train-detector",
        help="Entraîner un détecteur multiobjets et tester le comptage sur des scènes synthétiques.",
    )
    train_detector.add_argument(
        "--source",
        required=True,
        help="Dossier des images classées par valeur (sous-dossiers 0.1 à 200).",
    )
    train_detector.add_argument("--weights", default="yolo11n.pt")
    train_detector.add_argument("--epochs", type=int, default=40)
    train_detector.add_argument("--imgsz", type=int, default=416)
    train_detector.add_argument("--batch", type=int, default=8)
    train_detector.add_argument("--device", default="cpu")
    train_detector.add_argument("--output-dir", default="runs/detection")
    train_detector.add_argument("--dataset-dir", default="runs/detection_dataset")
    train_detector.add_argument("--model-output", default="models/best_mad_detector.pt")
    train_detector.add_argument("--train-scenes", type=int, default=72)
    train_detector.add_argument("--validation-scenes", type=int, default=36)
    train_detector.add_argument("--test-scenes", type=int, default=72)

    classify = commands.add_parser(
        "classify",
        help="Classer une photo recadrée contenant une seule pièce ou un billet.",
    )
    classify.add_argument("--weights", default="models/mad_classifier.pt")
    classify.add_argument("--source", required=True, help="Chemin vers la photo d'un seul objet.")
    classify.add_argument("--imgsz", type=int, default=128)

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
        from .inference import _validate_model_names, predict_source
        from .model_loading import load_yolo_model

        model = load_yolo_model(args.weights)
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
        from .model_loading import load_yolo_model

        weights = Path(args.weights)
        if not weights.is_file():
            parser.error(f"Poids introuvables : {weights}")
        exported = load_yolo_model(weights).export(format=args.format, imgsz=args.imgsz)
        print(f"Modèle exporté : {exported}")
    elif args.command == "train-classifier":
        if args.epochs < 1 or args.imgsz < 1 or args.batch < 1:
            parser.error("--epochs, --imgsz et --batch doivent être positifs.")
        from .classifier import train_classifier

        train_classifier(
            source=args.source,
            weights=args.weights,
            epochs=args.epochs,
            image_size=args.imgsz,
            batch_size=args.batch,
            device=args.device,
            output_dir=args.output_dir,
            dataset_dir=args.dataset_dir,
            model_output=args.model_output,
        )
    elif args.command == "train-detector":
        if args.epochs < 1 or args.imgsz < 64 or args.batch < 1:
            parser.error("--epochs, --batch doivent être positifs et --imgsz doit être >= 64.")
        if min(args.train_scenes, args.validation_scenes, args.test_scenes) < 1:
            parser.error("--train-scenes, --validation-scenes et --test-scenes doivent être positifs.")
        from .classifier import train_detector_from_classification

        train_detector_from_classification(
            source=args.source,
            weights=args.weights,
            epochs=args.epochs,
            image_size=args.imgsz,
            batch_size=args.batch,
            device=args.device,
            output_dir=args.output_dir,
            dataset_dir=args.dataset_dir,
            model_output=args.model_output,
            train_scenes=args.train_scenes,
            validation_scenes=args.validation_scenes,
            test_scenes=args.test_scenes,
        )
    elif args.command == "classify":
        from pathlib import Path
        from .classifier import classify_image

        if not Path(args.weights).is_file():
            parser.error(f"Poids introuvables : {args.weights}")
        classify_image(args.weights, args.source, args.imgsz)


if __name__ == "__main__":
    main()
