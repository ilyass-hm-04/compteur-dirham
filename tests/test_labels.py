import unittest

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


if __name__ == "__main__":
    unittest.main()
