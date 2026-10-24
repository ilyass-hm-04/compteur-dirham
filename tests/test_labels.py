import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

from mad_counter.classifier import FOLDER_TO_CLASS, prepare_classification_data
from mad_counter.inference import _validate_model_names, annotate_frame
from mad_counter.labels import CLASS_NAMES, CLASS_VALUES_DH


class LabelTests(unittest.TestCase):
    def test_class_values_follow_the_model_class_order(self):
        self.assertEqual(tuple(CLASS_VALUES_DH), CLASS_NAMES)
        self.assertEqual(len(CLASS_NAMES), 11)
        self.assertEqual(CLASS_VALUES_DH["coin_10c"], 0.10)
        self.assertEqual(CLASS_VALUES_DH["note_200dh"], 200.0)

    def test_model_names_must_use_the_canonical_class_order(self):
        mapping = dict(enumerate(CLASS_NAMES))
        self.assertEqual(_validate_model_names(mapping), mapping)

        mapping[0], mapping[1] = mapping[1], mapping[0]
        with self.assertRaises(ValueError):
            _validate_model_names(mapping)

    def test_detector_sums_multiple_objects_by_their_money_value(self):
        import numpy as np

        detections = (3, 0, 7, 3)
        names = dict(enumerate(CLASS_NAMES))
        result = SimpleNamespace(
            names=names,
            boxes=[
                SimpleNamespace(cls=SimpleNamespace(item=lambda class_id=class_id: class_id))
                for class_id in detections
            ],
            plot=lambda: np.zeros((80, 240, 3), dtype=np.uint8),
        )
        model = SimpleNamespace(
            predict=lambda *args, **kwargs: [result]
        )

        annotated, total, counts = annotate_frame(
            model, np.zeros((80, 240, 3), dtype=np.uint8)
        )

        self.assertEqual(total, 22.1)
        self.assertEqual(counts, {"coin_1dh": 2, "coin_10c": 1, "note_20dh": 1})
        self.assertEqual(annotated.shape, (80, 240, 3))

    def test_classifier_data_is_deduplicated_before_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "prepared"
            for folder in FOLDER_TO_CLASS:
                class_dir = source / folder
                class_dir.mkdir(parents=True)
                for index in range(3):
                    (class_dir / f"{index}.jpg").write_bytes(
                        f"{folder}-{index}".encode()
                    )
                (class_dir / "duplicate.jpg").write_bytes(
                    f"{folder}-0".encode()
                )

            prepare_classification_data(source, destination)

            for split in ("train", "val", "test"):
                images = list((destination / split).rglob("*.jpg"))
                self.assertEqual(len(images), len(CLASS_NAMES))
            train_hashes = {
                image.read_bytes()
                for image in (destination / "train").rglob("*.jpg")
            }
            test_hashes = {
                image.read_bytes()
                for image in (destination / "test").rglob("*.jpg")
            }
            self.assertFalse(train_hashes & test_hashes)

    def test_classifier_rejects_identical_images_with_conflicting_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            for folder in FOLDER_TO_CLASS:
                class_dir = source / folder
                class_dir.mkdir(parents=True)
                for index in range(3):
                    (class_dir / f"{index}.jpg").write_bytes(
                        f"{folder}-{index}".encode()
                    )
            (source / "0.1" / "same.jpg").write_bytes(b"same")
            (source / "0.2" / "same.jpg").write_bytes(b"same")

            with self.assertRaisesRegex(ValueError, "attribuées à plusieurs classes"):
                prepare_classification_data(source, root / "prepared")

    def test_detection_dataset_has_four_held_out_objects_per_scene(self):
        import cv2
        import numpy as np

        from mad_counter.classifier import prepare_detection_data

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "prepared"
            for class_index, folder in enumerate(FOLDER_TO_CLASS):
                class_dir = source / folder
                class_dir.mkdir(parents=True)
                for index in range(3):
                    image = np.full(
                        (32, 32, 3),
                        (class_index * 19 + index * 3) % 255,
                        dtype=np.uint8,
                    )
                    ok, encoded = cv2.imencode(".jpg", image)
                    self.assertTrue(ok)
                    (class_dir / f"{index}.jpg").write_bytes(encoded.tobytes())

            data_path = prepare_detection_data(
                source,
                destination,
                image_size=128,
                train_scenes=1,
                validation_scenes=3,
                test_scenes=3,
            )

            self.assertTrue(data_path.is_file())
            train_labels = destination / "train" / "labels"
            for class_id in range(len(CLASS_NAMES)):
                labels_path = next(train_labels.glob(f"{class_id}_*.txt"))
                fields = labels_path.read_text(encoding="utf-8").split()
                self.assertEqual(len(fields), 5)
                self.assertEqual(fields[1:], ["0.5", "0.5", "1.0", "1.0"])

            train_scene = train_labels / "mosaic_0000.txt"
            self.assertEqual(len(train_scene.read_text().splitlines()), 4)
            validation_labels = sorted((destination / "val" / "labels").glob("*.txt"))
            self.assertEqual(len(validation_labels), 3)
            for labels_path in validation_labels:
                self.assertEqual(len(labels_path.read_text().splitlines()), 4)
            test_labels = sorted((destination / "test" / "labels").glob("*.txt"))
            self.assertEqual(len(test_labels), 3)
            for labels_path in test_labels:
                labels = labels_path.read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(labels), 4)
                for line in labels:
                    class_id, center_x, center_y, width, height = map(
                        float, line.split()
                    )
                    self.assertGreaterEqual(class_id, 0)
                    self.assertLess(class_id, len(CLASS_NAMES))
                    for coordinate in (center_x, center_y, width, height):
                        self.assertGreater(coordinate, 0)
                        self.assertLessEqual(coordinate, 1)


if __name__ == "__main__":
    unittest.main()
