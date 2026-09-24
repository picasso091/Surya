"""Two-input flare dataset for the pilot; this module does not launch training.

Check a real example from ~/Surya with:
    .venv/bin/python downstream_examples/solar_flare_forcasting/experiments/checks/check_flare_dataset.py --manifest-dir downstream_examples/solar_flare_forcasting/experiments/manifests/pilot50_20260919T220418_102479Z

Each item is (batch, metadata). Before batching, batch contains float32 tensors:
ts [channels, 2, height, width], time_delta_input [2], and scalar label.
Standard PyTorch DataLoader collation adds the batch dimension. No future image
is opened. Errors identify the failing sample instead of substituting a neighbor.
This first version uses full-resolution inputs, no pooling, and no augmentation.
"""

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr
import yaml

EXPERIMENTS = Path(__file__).resolve().parent
TASK = EXPERIMENTS.parent
sys.path.insert(0, str(TASK.parents[1]))

from surya.datasets.helio import transform
from surya.utils.data import build_scalers


def parse_utc(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() != timedelta(0):
        raise ValueError(f"Expected an explicit UTC timestamp: {value}")
    return result.astimezone(timezone.utc)


class ManifestFlareDataset(Dataset):
    def __init__(self, manifest_path, config_path=TASK / "config_infer.yaml", *, image_size=4096):
        self.manifest_path = Path(manifest_path).resolve()
        self.config_path = Path(config_path).resolve()
        config = yaml.safe_load(self.config_path.read_text())
        self.channels = list(config["data"]["channels"])
        if not self.channels or len(set(self.channels)) != len(self.channels):
            raise ValueError("Channel list must be nonempty and unique.")
        if config["data"]["time_delta_input_minutes"] != [-60, 0]:
            raise ValueError("This manifest loader expects inputs at t-60 minutes and t.")
        self.image_size = image_size
        self.scalers_path = (self.config_path.parent / config["data"]["scalers_path"]).resolve()
        self.scalers = build_scalers(yaml.safe_load(self.scalers_path.read_text()))
        self.means = np.array([self.scalers[ch].mean for ch in self.channels])
        self.stds = np.array([self.scalers[ch].std for ch in self.channels])
        self.factors = np.array([self.scalers[ch].sl_scale_factor for ch in self.channels])
        self.epsilons = np.array([self.scalers[ch].epsilon for ch in self.channels])
        if not all(np.isfinite(a).all() for a in (self.means, self.stds, self.factors, self.epsilons)):
            raise ValueError("Non-finite normalization settings.")
        if np.any(self.stds + self.epsilons <= 0):
            raise ValueError("Normalization denominator must be positive.")
        with self.manifest_path.open(newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError("Manifest has no examples.")
        ids = set()
        splits = set()
        for row in self.rows:
            reference = parse_utc(row["reference_time_utc"])
            if row["sample_id"] in ids or row["sample_id"] != reference.strftime("%Y%m%d_%H%M"):
                raise ValueError(f"Duplicate or inconsistent sample ID: {row['sample_id']}")
            ids.add(row["sample_id"])
            splits.add(row["split"])
            if row["label_max"] not in ("0", "1"):
                raise ValueError(f"Invalid label for {row['sample_id']}")
            if parse_utc(row["label_window_end_utc"]) != reference + timedelta(hours=24):
                raise ValueError(f"Incorrect label horizon for {row['sample_id']}")
            for name, offset in (("previous", -1), ("current", 0)):
                timestamp = parse_utc(row[f"{name}_time_utc"])
                path = Path(row[f"{name}_local_path"])
                if timestamp != reference + timedelta(hours=offset):
                    raise ValueError(f"Incorrect {name} timestamp for {row['sample_id']}")
                if not path.is_absolute() or path.stem != timestamp.strftime("%Y%m%d_%H%M"):
                    raise ValueError(f"Incorrect {name} path for {row['sample_id']}: {path}")
                if not path.is_file():
                    raise FileNotFoundError(f"Missing input for {row['sample_id']}: {path}")
                if path.stat().st_size != int(row[f"{name}_size_bytes"]):
                    raise ValueError(f"Wrong input size for {row['sample_id']}: {path}")
        if len(splits) != 1 or not splits <= {"train", "validation", "test"}:
            raise ValueError(f"A dataset must contain exactly one recognized split, got {splits}.")
        self.split = next(iter(splits))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        inputs = torch.empty((len(self.channels), 2, self.image_size, self.image_size), dtype=torch.float32)
        for time_index, name in enumerate(("previous", "current")):
            path = Path(row[f"{name}_local_path"])
            timestamp = parse_utc(row[f"{name}_time_utc"])
            with xr.open_dataset(path, engine="h5netcdf", chunks=None, cache=False) as frame:
                if str(frame.attrs.get("data_time", "")) != timestamp.strftime("%Y%m%d_%H%M"):
                    raise ValueError(f"Embedded timestamp mismatch: {path}")
                for channel_index, channel in enumerate(self.channels):
                    variable = frame[channel]
                    if variable.dims != ("y", "x") or variable.shape != (self.image_size, self.image_size):
                        raise ValueError(f"Unexpected shape for {channel}: {path}")
                    raw = variable.to_numpy()
                    if not np.isfinite(raw).all():
                        raise ValueError(f"Non-finite raw values in {channel}: {path}")
                    # The existing normalization is pointwise and independent per channel.
                    # Reuse it one channel at a time to bound temporary host-memory use.
                    i = slice(channel_index, channel_index + 1)
                    normalized = transform(raw[None], self.means[i], self.stds[i], self.factors[i], self.epsilons[i])
                    normalized = normalized.astype(np.float32)
                    if not np.isfinite(normalized).all():
                        raise ValueError(f"Non-finite normalized values in {channel}: {path}")
                    inputs[channel_index, time_index].copy_(torch.from_numpy(normalized[0]))
                    del raw, normalized
        return {
            "ts": inputs,
            "time_delta_input": torch.tensor([1.0, 0.0], dtype=torch.float32),
            "label": torch.tensor(float(row["label_max"]), dtype=torch.float32),
        }, {
            "sample_id": row["sample_id"], "split": row["split"],
            "reference_time_utc": row["reference_time_utc"],
            "input_times_utc": [row["previous_time_utc"], row["current_time_utc"]],
        }
