"""CPU-only cache tests; no teacher weights or solar images are loaded.

From ~/Surya:
    .venv/bin/python downstream_examples/solar_flare_forcasting/experiments/tests/test_teacher_cache.py
"""

from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cache_teacher_predictions as cache_module


class TeacherCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rows = [dict(sample_id=f"sample{i}", split="train", label_max=str(i),
                          reference_time_utc=f"2011-02-15T1{i}:00:00Z") for i in range(2)]
        self.record = {
            **self.rows[0], "label_max": 0, "teacher_logit": 0., "teacher_probability": .5,
            "prediction_at_0.5": 0, "preparation_seconds": 0., "inference_seconds": 0.,
            "peak_allocated_gpu_gib": 0.,
        }

    def test_provenance_change_rejected(self):
        cache = cache_module.get_cache(self.root, {"weights": "old"}, self.rows)
        cache_module.save_cache(self.root, cache)
        with self.assertRaisesRegex(ValueError, "different inputs"):
            cache_module.get_cache(self.root, {"weights": "new"}, self.rows)

    def test_duplicate_wrong_label_and_bad_probability_rejected(self):
        self.assertEqual(cache_module.validate_cached_rows([self.record], self.rows), {"sample0"})
        bad_cases = [
            [self.record, self.record], [{**self.record, "label_max": 1}],
            [{**self.record, "teacher_probability": .9}],
            [{**self.record, "teacher_logit": float("nan")}],
        ]
        for case in bad_cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                cache_module.validate_cached_rows(case, self.rows)

    def test_interrupted_run_resumes_only_unsaved_samples(self):
        root, rows = self.root, self.rows

        class TinyDataset(torch.utils.data.Dataset):
            def __init__(self, *args):
                self.rows = rows
                self.split = "train"
                self.channels = ["test"]
                self.image_size = 4
                self.manifest_path = root / "manifest.csv"
                self.scalers_path = root / "scalers.yaml"

            def __len__(self):
                return len(rows)

            def __getitem__(self, i):
                return {"ts": torch.full((1, 2, 4, 4), float(i)),
                        "time_delta_input": torch.tensor([1., 0.]), "label": torch.tensor(float(i))}, {
                            "sample_id": rows[i]["sample_id"]}

        calls = []
        interrupt = [True]

        class TinyTeacher(torch.nn.Module):
            def forward(self, inputs):
                i = int(inputs["ts"][0, 0, 0, 0, 0].item())
                calls.append(i)
                if i == 1 and interrupt[0]:
                    raise RuntimeError("simulated interruption")
                return torch.tensor([float(i)])

        config = {"model": {"model_type": "spectformer", "learned_flow": False,
                            "img_size": 4, "in_channels": 1}, "pretrained_path": "base.pt"}
        (root / "config_infer.yaml").write_text(yaml.safe_dump(config))
        args = SimpleNamespace(manifest_dir=root)
        with ExitStack() as stack, redirect_stdout(io.StringIO()):
            stack.enter_context(patch.object(cache_module, "TASK", root))
            stack.enter_context(patch.object(cache_module, "ManifestFlareDataset", TinyDataset))
            stack.enter_context(patch.object(cache_module, "sha256", return_value="test-hash"))
            stack.enter_context(patch.dict(sys.modules, {"infer": SimpleNamespace(load_model=lambda *a: TinyTeacher())}))
            stack.enter_context(patch.object(torch.Tensor, "to", lambda self, *a, **kw: self))
            stack.enter_context(patch.object(torch.cuda, "is_available", return_value=True))
            stack.enter_context(patch.object(torch.cuda, "get_device_name", return_value="mock GPU"))
            for method in ("synchronize", "reset_peak_memory_stats"):
                stack.enter_context(patch.object(torch.cuda, method))
            stack.enter_context(patch.object(torch.cuda, "max_memory_allocated", return_value=0))
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                cache_module.run(args, root)
            saved = json.loads((root / "cache.json").read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(len(saved["predictions"]), 1)
            # CSV loss is recoverable: the JSON is authoritative.
            (root / "predictions.csv").unlink()
            interrupt[0] = False
            cache_module.run(args, root)
            saved = json.loads((root / "cache.json").read_text())
            self.assertEqual(saved["status"], "complete")
            self.assertEqual([x["sample_id"] for x in saved["predictions"]], ["sample0", "sample1"])
            self.assertTrue((root / "predictions.csv").exists())
            self.assertEqual(calls, [0, 1, 1])  # The saved sample 0 was never recomputed.
            cache_module.run(args, root)
            self.assertEqual(calls, [0, 1, 1])  # Complete cache is a no-op.


if __name__ == "__main__":
    unittest.main()
