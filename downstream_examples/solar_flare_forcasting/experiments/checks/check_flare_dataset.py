"""Check one full-resolution flare example on CPU (no model inference).

Run from ~/Surya:
    .venv/bin/python downstream_examples/solar_flare_forcasting/experiments/checks/check_flare_dataset.py --manifest-dir downstream_examples/solar_flare_forcasting/experiments/manifests/pilot50_20260919T220418_102479Z

Use --index N to check a different example. Results go into a timestamped
experiments/reports/<manifest-name>/dataset_checks/ folder. This checks all values
of the selected pair, then compares small raw-image patches against the original
loader's normalization and checks its label against the official label CSV.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flare_dataset import ManifestFlareDataset, TASK, EXPERIMENTS, parse_utc
from surya.datasets.helio import HelioNetCDFDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = args.manifest_dir.resolve() / "manifest.csv"
    output = EXPERIMENTS / "reports" / manifest.parent.name / "dataset_checks" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "manifest": str(manifest), "index": args.index}
    print(f"Results: {output / 'results.json'}", flush=True)
    try:
        dataset = ManifestFlareDataset(manifest)
        if not 0 <= args.index < len(dataset):
            raise ValueError(f"Index must be between 0 and {len(dataset) - 1}")
        row = dataset.rows[args.index]
        print(f"All {len(dataset)} examples have both input files at the expected sizes.", flush=True)
        print(f"Loading and normalizing {row['sample_id']} (two images)...", flush=True)
        start = time.perf_counter()
        # Use the normal DataLoader path to check batching and metadata collation too.
        batch, metadata = next(iter(DataLoader(dataset, batch_size=1, sampler=[args.index], num_workers=0)))
        load_seconds = time.perf_counter() - start
        assert tuple(batch["ts"].shape) == (1, len(dataset.channels), 2, 4096, 4096)
        assert batch["ts"].dtype == torch.float32
        assert torch.equal(batch["time_delta_input"], torch.tensor([[1., 0.]]))
        assert set(batch) == {"ts", "time_delta_input", "label"}
        official = TASK / "assets/surya-bench-flare-forecasting" / f"{dataset.split}.csv"
        with official.open() as handle:
            labels = {r["timestamp"]: int(r["label_max"]) for r in csv.DictReader(handle)}
        label = labels[parse_utc(row["reference_time_utc"]).strftime("%Y-%m-%d %H:%M:%S")]
        assert batch["label"].item() == label
        # Compare against the original multi-channel transform_data method, without
        # constructing its future-image-dependent dataset or reading a future file.
        reference = HelioNetCDFDataset.__new__(HelioNetCDFDataset)
        reference.channels, reference.scalers, reference.pooling = dataset.channels, dataset.scalers, 1
        differences = []
        for t, name in enumerate(("previous", "current")):
            with xr.open_dataset(row[f"{name}_local_path"], engine="h5netcdf", chunks=None, cache=False) as frame:
                for start_pixel in (0, 2044, 4088):
                    window = slice(start_pixel, start_pixel + 8)
                    raw = frame[dataset.channels].isel(y=window, x=window).to_array().to_numpy()
                    expected = reference.transform_data(raw).astype(np.float32)
                    actual = batch["ts"][0, :, t, window, window].numpy()
                    np.testing.assert_array_equal(actual, expected)
                    differences.append(float(np.max(np.abs(actual - expected))))
        stats = {}
        for i, channel in enumerate(dataset.channels):
            values = batch["ts"][0, i]
            assert torch.isfinite(values).all().item()
            stats[channel] = {"min": values.min().item(), "max": values.max().item()}
        report.update({
            "status": "passed", "examples": len(dataset), "sample_id": row["sample_id"],
            "split": dataset.split, "reference_time_utc": row["reference_time_utc"],
            "input_times_utc": [row["previous_time_utc"], row["current_time_utc"]],
            "label": label, "label_matches_official_csv": True,
            "input_shape": list(batch["ts"].shape), "dtype": str(batch["ts"].dtype),
            "time_delta_input_hours": batch["time_delta_input"].tolist(),
            "input_image_count": 2, "uses_future_image": False,
            "load_and_normalize_seconds": load_seconds,
            "normalization_comparison": "Exact float32 match on three 8x8 patches per image, all channels",
            "maximum_patch_difference": max(differences), "normalized_channel_ranges": stats,
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "scalers_sha256": hashlib.sha256(dataset.scalers_path.read_bytes()).hexdigest(),
        })
        print(f"Passed: shape={report['input_shape']}, label={label}, offsets=[[1.0, 0.0]] hours.", flush=True)
        print("Normalized values are finite; sampled patches exactly match the existing loader.", flush=True)
        print(f"Loaded and normalized in {load_seconds:.2f} seconds; no future image used.", flush=True)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
