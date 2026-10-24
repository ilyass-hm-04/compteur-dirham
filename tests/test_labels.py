import unittest
import tempfile
from pathlib import Path

from mad_counter.classifier import FOLDER_TO_CLASS, prepare_classification_data
from mad_counter.inference import _validate_model_names
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


if __name__ == "__main__":
    unittest.main()
