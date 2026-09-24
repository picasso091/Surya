"""CPU tests: .venv/bin/python downstream_examples/solar_flare_forcasting/experiments/tests/test_student_distillation.py"""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_student_baseline as training
import test_student_baseline as fixtures


class DistillationTests(unittest.TestCase):
    def test_loss_gradient_and_shuffled_teacher_alignment(self):
        logits = torch.tensor([-2., 1., 0.], requires_grad=True)
        labels = torch.tensor([1., 0., 1.])
        teacher = {"a": {"teacher_probability": 0.9}, "b": {"teacher_probability": 0.1},
                   "c": {"teacher_probability": 0.4}}
        loss, hard, soft = training.training_losses(logits, labels, ["b", "c", "a"], teacher)
        targets = torch.tensor([0.1, 0.4, 0.9])
        probability = logits.detach().sigmoid()
        expected_soft = -(targets * probability.log() + (1-targets) * (1-probability).log()).mean()
        torch.testing.assert_close(soft, expected_soft)
        torch.testing.assert_close(loss, 0.5 * hard + 0.5 * soft)
        loss.backward()
        torch.testing.assert_close(logits.grad, (probability - 0.5 * (labels + targets)) / 3)
        baseline_loss, baseline_hard, baseline_soft = training.training_losses(logits, labels, [])
        self.assertIsNone(baseline_soft)
        torch.testing.assert_close(baseline_loss, baseline_hard)
        with self.assertRaises(KeyError):
            training.training_losses(logits, labels, ["missing", "c", "a"], teacher)

    def test_matched_runs_initialization_order_and_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory, manifest, _ = fixtures.BaselineTests().make_data(root)
            args = ["train_student_distillation.py", "--student-dir", str(directory),
                    "--manifest", str(manifest), "--epochs", "1", "--device", "cpu"]
            reports, orders, checkpoints = [], [], []
            original_getitem = training.StudentDataset.__getitem__
            for distillation in (False, True):
                visited = []

                def tracked_getitem(dataset, index):
                    item = original_getitem(dataset, index)
                    visited.append(item["sample_id"])
                    return item

                with patch.object(training, "INPUT_SHAPE", (13, 2, 64, 64)), \
                        patch.object(training, "EXPERIMENTS", root), patch.object(sys, "argv", args), \
                        patch.object(training.StudentDataset, "__getitem__", tracked_getitem):
                    training.main(distillation=distillation)
                mode = "distillation" if distillation else "labels_only"
                output = next((root / "runs" / mode).rglob("results.json")).parent
                reports.append(json.loads((output / "results.json").read_text()))
                orders.append(visited)
                checkpoints.append(torch.load(output / "last.pt", map_location="cpu", weights_only=True))
                self.assertTrue((output / "comparisons/epoch_001.csv").is_file())
                self.assertTrue((output / "student_errors_vs_teacher.csv").is_file())
            self.assertEqual(orders[0], orders[1])
            self.assertEqual(reports[0]["history"][0]["train_eval_loss"],
                             reports[1]["history"][0]["train_eval_loss"])
            distilled = reports[1]
            self.assertEqual(distilled["status"], "complete")
            self.assertTrue(distilled["settings"]["teacher_targets_used"])
            epoch = distilled["history"][1]
            self.assertAlmostEqual(epoch["train_loss"],
                                   0.5 * (epoch["train_label_loss"] + epoch["train_teacher_loss"]), places=6)
            self.assertFalse(torch.equal(checkpoints[0]["model_state_dict"]["classifier.weight"],
                                         checkpoints[1]["model_state_dict"]["classifier.weight"]))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
