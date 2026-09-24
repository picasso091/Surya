"""CPU tests: .venv/bin/python downstream_examples/solar_flare_forcasting/experiments/tests/test_student_baseline.py"""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_student_baseline as baseline


class BaselineTests(unittest.TestCase):
    def make_data(self, root):
        directory = root / "student"
        (directory / "arrays").mkdir(parents=True)
        manifest = root / "manifest.csv"
        records = []
        generator = np.random.default_rng(7)
        # Three examples exercise an uneven final batch and sample-weighted loss.
        for i, label in enumerate((0, 1, 0)):
            path = directory / "arrays" / f"sample{i}.npy"
            np.save(path, generator.normal(size=(13, 2, 64, 64)).astype(np.float32))
            records.append({"sample_id": f"sample{i}", "split": "train", "label_max": label,
                            "reference_time_utc": str(i), "previous_time_utc": str(i),
                            "current_time_utc": str(i), "array_path": str(path),
                            "array_sha256": baseline.sha256(path), "size_bytes": path.stat().st_size})
        baseline.save_csv(manifest, records)
        cache_path = root / "teacher_cache" / root.name / "cache.json"
        cache_path.parent.mkdir(parents=True)
        teacher = [{**row, "teacher_probability": 0.2 + 0.6 * row["label_max"]}
                   for row in reversed(records)]
        baseline.save_json(cache_path, {"status": "complete", "predictions": teacher})
        state = {"status": "complete", "expected_samples": 3, "samples": records,
                 "provenance": {"channels": list(range(13)), "student_size": 64,
                                "input_times": ["t-60min", "t"], "dtype": "float32",
                                "manifest_sha256": baseline.sha256(manifest),
                                "teacher_cache_sha256": baseline.sha256(cache_path)}}
        baseline.save_json(directory / "preparation.json", state)
        return directory, manifest, state

    def test_full_training_checkpoint_and_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory, manifest, _ = self.make_data(root)
            args = ["train_student_baseline.py", "--student-dir", str(directory),
                    "--manifest", str(manifest), "--epochs", "1", "--device", "cpu"]
            with patch.object(baseline, "INPUT_SHAPE", (13, 2, 64, 64)), \
                    patch.object(baseline, "EXPERIMENTS", root), patch.object(sys, "argv", args):
                baseline.main()
                output = next((root / "runs").rglob("results.json")).parent
                report = json.loads((output / "results.json").read_text())
                self.assertEqual(report["status"], "complete")
                self.assertEqual(report["completed_epochs"], 1)
                self.assertEqual(len(report["history"]), 2)
                self.assertFalse(report["settings"]["teacher_targets_used"])
                checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
                self.assertEqual(checkpoint["epoch"], 1)
                model = baseline.ResNet18Classifier(in_channels=13, time_steps=2)
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                data = baseline.StudentDataset(directory, manifest)
                loader = baseline.DataLoader(data, batch_size=2)
                metrics, predictions = baseline.run_epoch(model, loader, torch.device("cpu"))
                self.assertFalse(model.training)
                self.assertAlmostEqual(metrics["loss"], report["history"][-1]["train_eval_loss"], places=6)
                with (output / "train_predictions.csv").open() as handle:
                    saved = list(csv.DictReader(handle))
                self.assertEqual([r["sample_id"] for r in saved], [r["sample_id"] for r in predictions])
                np.testing.assert_allclose([float(r["student_logit"]) for r in saved],
                                           [r["student_logit"] for r in predictions], atol=1e-6)
                direct_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    torch.tensor([r["student_logit"] for r in predictions]),
                    torch.tensor([float(r["label_max"]) for r in predictions])).item()
                self.assertAlmostEqual(metrics["loss"], direct_loss, places=6)
                self.assertTrue((output / "comparisons/epoch_000.csv").is_file())
                self.assertEqual((output / "comparisons/epoch_001.csv").read_text(),
                                 (output / "student_errors_vs_teacher.csv").read_text())
                with (output / "student_errors_vs_teacher.csv").open() as handle:
                    errors = list(csv.DictReader(handle))
                expected = [r for r in predictions if r["prediction_at_0.5"] != r["label_max"]]
                self.assertEqual([r["sample_id"] for r in errors], [r["sample_id"] for r in expected])
                for row in errors:
                    self.assertAlmostEqual(float(row["teacher_probability"]),
                                           0.2 + 0.6 * int(row["label_max"]))

    def test_comparison_empty_epoch_clears_previous_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher = {"a": {"teacher_probability": 0.1}}
            prediction = {"sample_id": "a", "label_max": 0, "student_probability": 0.5}
            baseline.save_error_comparison(root, 1, [prediction], teacher)
            with (root / "student_errors_vs_teacher.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["student_prediction_at_0.5"], "1")
            prediction["student_probability"] = 0.2
            baseline.save_error_comparison(root, 2, [prediction], teacher)
            with (root / "student_errors_vs_teacher.csv").open() as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, baseline.COMPARISON_FIELDS)
                self.assertEqual(list(reader), [])
            self.assertTrue((root / "comparisons/epoch_001.csv").is_file())

    def test_rejects_wrong_labels_and_corrupt_arrays(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(baseline, "INPUT_SHAPE", (13, 2, 64, 64)):
            directory, manifest, state = self.make_data(Path(tmp))
            state["samples"][0]["label_max"] = 1
            baseline.save_json(directory / "preparation.json", state)
            with self.assertRaisesRegex(ValueError, "Manifest mismatch"):
                baseline.StudentDataset(directory, manifest)
            state["samples"][0]["label_max"] = 0
            baseline.save_json(directory / "preparation.json", state)
            array_path = directory / "arrays/sample0.npy"
            values = np.load(array_path)
            values[0, 0, 0, 0] += 1
            np.save(array_path, values)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                baseline.StudentDataset(directory, manifest).verify_files()


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
