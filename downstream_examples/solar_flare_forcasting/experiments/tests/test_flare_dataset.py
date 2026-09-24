"""Small CPU tests using synthetic files; no pilot data is changed.

Run from ~/Surya:
    .venv/bin/python downstream_examples/solar_flare_forcasting/experiments/tests/test_flare_dataset.py
"""

import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flare_dataset import ManifestFlareDataset, TASK
from surya.datasets.helio import HelioNetCDFDataset


class FlareDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        config = yaml.safe_load((TASK / "config_infer.yaml").read_text())
        config["data"]["channels"] = ["aia94", "hmi_m"]
        config["data"]["scalers_path"] = str((TASK / config["data"]["scalers_path"]).resolve())
        self.config = self.root / "config.yaml"
        self.config.write_text(yaml.safe_dump(config))
        self.paths, self.raw = [], []
        for t, stamp in enumerate(("20110215_1500", "20110215_1600")):
            raw = np.arange(128, dtype=np.float32).reshape(2, 8, 8) - 64 + t * 10
            path = self.root / f"{stamp}.nc"
            xr.Dataset({ch: (("y", "x"), raw[i]) for i, ch in enumerate(config["data"]["channels"])},
                       attrs={"data_time": stamp}).to_netcdf(path, engine="h5netcdf")
            self.paths.append(path)
            self.raw.append(raw)
        self.row = {
            "sample_id": "20110215_1600", "reference_time_utc": "2011-02-15T16:00:00Z",
            "label_window_end_utc": "2011-02-16T16:00:00Z", "label_max": "1", "split": "train",
            "previous_time_utc": "2011-02-15T15:00:00Z", "current_time_utc": "2011-02-15T16:00:00Z",
        }
        for name, path in zip(("previous", "current"), self.paths):
            self.row[f"{name}_local_path"] = str(path)
            self.row[f"{name}_size_bytes"] = path.stat().st_size
        self.manifest = self.root / "manifest.csv"
        self.write_manifest()

    def write_manifest(self, rows=None):
        with self.manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.row))
            writer.writeheader()
            writer.writerows(rows if rows is not None else [self.row])

    def dataset(self):
        return ManifestFlareDataset(self.manifest, self.config, image_size=8)

    def test_only_two_inputs_and_original_normalization(self):
        dataset = self.dataset()
        with patch("flare_dataset.xr.open_dataset", wraps=xr.open_dataset) as opened:
            batch, metadata = dataset[0]
        self.assertEqual([call.args[0] for call in opened.call_args_list], self.paths)
        self.assertEqual(batch["ts"].shape, (2, 2, 8, 8))
        self.assertEqual(batch["label"].item(), 1)
        self.assertEqual(metadata["sample_id"], "20110215_1600")
        self.assertEqual(set(batch), {"ts", "time_delta_input", "label"})
        torch.testing.assert_close(batch["time_delta_input"], torch.tensor([1., 0.]))
        legacy = HelioNetCDFDataset.__new__(HelioNetCDFDataset)
        legacy.channels, legacy.scalers, legacy.pooling = dataset.channels, dataset.scalers, 1
        for t, raw in enumerate(self.raw):
            expected = legacy.transform_data(raw).astype(np.float32)
            np.testing.assert_array_equal(batch["ts"][:, t].numpy(), expected)

    def test_reversed_input_time_rejected(self):
        self.row["previous_time_utc"] = self.row["current_time_utc"]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Incorrect previous timestamp"):
            self.dataset()

    def test_missing_input_raises_instead_of_substituting(self):
        dataset = self.dataset()
        self.paths[0].unlink()
        with self.assertRaises(FileNotFoundError):
            dataset[0]

    def test_wrong_horizon_rejected(self):
        self.row["label_window_end_utc"] = "2011-02-15T17:00:00Z"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Incorrect label horizon"):
            self.dataset()

    def test_duplicate_sample_rejected(self):
        self.write_manifest([self.row, self.row])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.dataset()


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
