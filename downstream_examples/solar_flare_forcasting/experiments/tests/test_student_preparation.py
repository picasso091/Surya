"""CPU tests: .venv/bin/python downstream_examples/solar_flare_forcasting/experiments/tests/test_student_preparation.py"""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_student_data as preparation


class StudentPreparationTests(unittest.TestCase):
    def test_four_by_four_means_preserve_locations_and_constant_images(self):
        raw = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
        pooled = preparation.average_blocks(raw)
        expected = np.array([[raw[y:y+4, x:x+4].mean() for x in range(0, 16, 4)]
                             for y in range(0, 16, 4)], dtype=np.float32)
        np.testing.assert_array_equal(pooled, expected)
        np.testing.assert_array_equal(preparation.average_blocks(np.full((16, 16), -3., dtype=np.float32)),
                                      np.full((4, 4), -3., dtype=np.float32))

    def test_invalid_sizes_and_nonfinite_pixels_rejected(self):
        with self.assertRaises(ValueError):
            preparation.average_blocks(np.zeros((15, 16), dtype=np.float32))
        with self.assertRaises(ValueError):
            preparation.average_blocks(np.full((16, 16), np.nan, dtype=np.float32))

    def test_saved_array_shape_dtype_and_checksum(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(preparation, "SIZE", 4):
            path = Path(tmp) / "sample.npy"
            values = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
            np.save(path, values)
            fingerprint = preparation.sha256(path)
            np.testing.assert_array_equal(preparation.verify_array(path, 2, fingerprint), values)
            values[0, 0, 0, 0] += 1
            np.save(path, values)
            with self.assertRaisesRegex(ValueError, "checksum"):
                preparation.verify_array(path, 2, fingerprint)
            np.save(path, values.astype(np.float64))
            with self.assertRaisesRegex(ValueError, "dtype"):
                preparation.verify_array(path, 2)


if __name__ == "__main__":
    unittest.main()
