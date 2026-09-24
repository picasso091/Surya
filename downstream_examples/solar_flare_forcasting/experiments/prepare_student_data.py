"""Prepare and verify 1024x1024 student inputs, using CPU only.

From ~/Surya, prepare the first example:
    .venv/bin/python -u downstream_examples/solar_flare_forcasting/experiments/prepare_student_data.py --manifest-dir downstream_examples/solar_flare_forcasting/experiments/manifests/pilot50_20260919T220418_102479Z

Add --all to process all 50 examples. Reruns verify and reuse completed arrays.
Output: experiments/student_data/1024/<manifest-name>/
  arrays/<sample-id>.npy  Float32 [13, 2, 1024, 1024], already normalized.
  samples.csv            IDs, labels, input timestamps, paths, and teacher logits.
  preparation.json       Settings, hashes, checks, and progress.

Order: raw images -> nonoverlapping 4x4 mean -> existing sign-log normalization.
No crops or augmentation. Do NOT normalize the saved arrays again at training.
Original images and cached teacher predictions are not modified.
"""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import time

import numpy as np
import skimage.measure
import xarray as xr

from flare_dataset import ManifestFlareDataset, EXPERIMENTS, parse_utc
from cache_teacher_predictions import sha256, validate_cached_rows
from surya.datasets.helio import HelioNetCDFDataset, transform


SIZE = 1024
FACTOR = 4
FIELDS = ["sample_id", "split", "reference_time_utc", "previous_time_utc", "current_time_utc",
          "label_max", "teacher_logit", "array_path", "size_bytes", "array_sha256"]


def average_blocks(raw, factor=FACTOR):
    if raw.ndim != 2 or any(n % factor for n in raw.shape):
        raise ValueError("Input must be a 2D image divisible by the pooling factor.")
    if not np.isfinite(raw).all():
        raise ValueError("Raw image contains non-finite values.")
    return skimage.measure.block_reduce(raw, block_size=(factor, factor), func=np.mean)


def verify_array(path, channels, expected_hash=None):
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != (channels, 2, SIZE, SIZE) or array.dtype != np.dtype("float32"):
        raise ValueError(f"Unexpected saved shape or dtype: {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"Saved array has non-finite values: {path}")
    if expected_hash is not None and sha256(path) != expected_hash:
        raise ValueError(f"Saved array checksum changed: {path}")
    return array


def prepare_pair(dataset, row):
    pair = np.empty((len(dataset.channels), 2, SIZE, SIZE), dtype=np.float32)
    max_pool_difference = 0.
    max_normalization_difference = 0.
    channel_ranges = {}
    for t, name in enumerate(("previous", "current")):
        path = Path(row[f"{name}_local_path"])
        with xr.open_dataset(path, engine="h5netcdf", chunks=None, cache=False) as frame:
            stamp = parse_utc(row[f"{name}_time_utc"]).strftime("%Y%m%d_%H%M")
            if frame.attrs.get("data_time") != stamp:
                raise ValueError(f"Embedded timestamp mismatch: {path}")
            for i, channel in enumerate(dataset.channels):
                variable = frame[channel]
                if variable.dims != ("y", "x") or variable.shape != (4096, 4096):
                    raise ValueError(f"Unexpected shape for {channel} in {path}")
                raw = variable.to_numpy()
                if raw.dtype != np.dtype("float32"):
                    raise ValueError(f"Unexpected raw dtype in {path}: {raw.dtype}")
                pooled = average_blocks(raw)
                s = slice(i, i + 1)
                normalized = transform(pooled[None], dataset.means[s], dataset.stds[s],
                                       dataset.factors[s], dataset.epsilons[s])[0].astype(np.float32)
                if not np.isfinite(normalized).all():
                    raise ValueError(f"Non-finite normalized values for {channel}: {path}")
                # Check block location/mean independently at the corner, center, and edge.
                # Compare normalization with the original loader's pooling=4 path.
                reference = HelioNetCDFDataset.__new__(HelioNetCDFDataset)
                reference.channels, reference.scalers, reference.pooling = [channel], dataset.scalers, FACTOR
                for y, x in ((0, 0), (SIZE // 2, SIZE // 2), (SIZE - 1, SIZE - 1)):
                    block = raw[y * FACTOR:(y + 1) * FACTOR, x * FACTOR:(x + 1) * FACTOR]
                    expected_mean = block.astype(np.float64).mean()
                    # Float32 reduction order can differ slightly from the float64 reference.
                    tolerance = 1e-6 * max(1., float(np.abs(block).max()))
                    difference = abs(float(pooled[y, x]) - expected_mean)
                    if difference > tolerance:
                        raise ValueError(f"Block average mismatch for {channel}: {path}")
                    max_pool_difference = max(max_pool_difference, difference)
                    expected = reference.transform_data(block[None]).astype(np.float32)[0, 0, 0]
                    np.testing.assert_allclose(normalized[y, x], expected, rtol=1e-6, atol=1e-6)
                    max_normalization_difference = max(max_normalization_difference,
                                                       abs(float(normalized[y, x]) - float(expected)))
                pair[i, t] = normalized
                channel_ranges[f"{name}/{channel}"] = {"min": float(normalized.min()), "max": float(normalized.max())}
                del raw, pooled, normalized
    return pair, {
        "shape": list(pair.shape), "dtype": str(pair.dtype), "all_finite": True,
        "checked_blocks": len(dataset.channels) * 2 * 3,
        "max_pool_difference_from_float64_mean": max_pool_difference,
        "max_normalization_difference_from_original_loader": max_normalization_difference,
        "normalized_ranges": channel_ranges,
    }


def save_progress(output, state):
    temporary = output / "preparation.json.tmp"
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "preparation.json")
    temporary = output / "samples.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(state["samples"])
    temporary.replace(output / "samples.csv")


def run(manifest_dir, output, all_samples):
    dataset = ManifestFlareDataset(manifest_dir / "manifest.csv")
    teacher_path = EXPERIMENTS / "teacher_cache" / manifest_dir.name / "cache.json"
    teacher = json.loads(teacher_path.read_text())
    ids = validate_cached_rows(teacher["predictions"], dataset.rows)
    if teacher["status"] != "complete" or len(ids) != len(dataset):
        raise ValueError("A complete, matching teacher cache is required.")
    fingerprints = {"manifest_sha256": sha256(dataset.manifest_path),
                    "file_list_sha256": sha256(manifest_dir / "files.csv"),
                    "scalers_sha256": sha256(dataset.scalers_path)}
    if any(teacher["provenance"][key] != value for key, value in fingerprints.items()):
        raise ValueError("Teacher cache does not match the manifest, file list, or normalization settings.")
    if teacher["provenance"]["channels"] != dataset.channels:
        raise ValueError("Teacher/student channel order differs.")
    predictions = {row["sample_id"]: row for row in teacher["predictions"]}
    provenance = {
        **fingerprints, "teacher_cache_sha256": sha256(teacher_path),
        "script_sha256": sha256(__file__), "dataset_code_sha256": sha256(EXPERIMENTS / "flare_dataset.py"),
        "transform_code_sha256": sha256(Path(__file__).resolve().parents[3] / "surya/datasets/helio.py"),
        "channels": dataset.channels, "input_times": ["t-60min", "t"],
        "raw_size": 4096, "student_size": SIZE, "pooling_factor": FACTOR,
        "preprocessing": "raw 4x4 mean then existing sign-log normalization", "dtype": "float32",
        "numpy_version": np.__version__, "skimage_version": skimage.__version__,
    }
    state_path = output / "preparation.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["provenance"] != provenance:
            raise ValueError("Existing student preparation has different settings or sources; use --output-dir with a new folder.")
    else:
        state = {"status": "pending", "provenance": provenance, "expected_samples": len(dataset), "samples": []}
    original_rows = {r["sample_id"]: r for r in dataset.rows}
    done = {}
    for sample in state["samples"]:
        sid = sample["sample_id"]
        if sid in done or sid not in original_rows:
            raise ValueError(f"Duplicate or unknown prepared sample: {sid}")
        for key in ("split", "reference_time_utc", "previous_time_utc", "current_time_utc", "label_max"):
            if str(sample[key]) != str(original_rows[sid][key]):
                raise ValueError(f"Saved {key} does not match manifest: {sid}")
        if sample["teacher_logit"] != predictions[sid]["teacher_logit"]:
            raise ValueError(f"Saved logit does not match teacher cache: {sid}")
        expected_path = output / "arrays" / f"{sid}.npy"
        if Path(sample["array_path"]) != expected_path:
            raise ValueError(f"Unexpected saved array path: {sid}")
        verify_array(expected_path, len(dataset.channels), sample["array_sha256"])
        done[sid] = sample
    selected = dataset.rows if all_samples else dataset.rows[:1]
    pending = [row for row in selected if row["sample_id"] not in done]
    required_bytes = len(pending) * len(dataset.channels) * 2 * SIZE * SIZE * 4
    if shutil.disk_usage(output).free < required_bytes + 1024**3:
        raise RuntimeError("Insufficient free disk space for arrays plus a 1 GiB margin.")
    state.update(status="running", error=None)
    save_progress(output, state)
    try:
        for row in pending:
            sid = row["sample_id"]
            print(f"Preparing {sid}: raw 4x4 averaging, then normalization...", flush=True)
            start = time.perf_counter()
            pair, checks = prepare_pair(dataset, row)
            path = output / "arrays" / f"{sid}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".npy.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, pair, allow_pickle=False)
            reloaded = verify_array(temporary, len(dataset.channels))
            if not np.array_equal(pair, reloaded):
                raise ValueError(f"Saved/reloaded array differs for {sid}")
            del reloaded, pair
            temporary.replace(path)
            record = {key: row[key] for key in FIELDS if key in row}
            record.update(label_max=int(row["label_max"]), teacher_logit=predictions[sid]["teacher_logit"],
                          array_path=str(path), size_bytes=path.stat().st_size, array_sha256=sha256(path),
                          elapsed_seconds=time.perf_counter() - start, checks=checks)
            state["samples"].append(record)
            save_progress(output, state)
            print(f"Verified {len(state['samples'])}/{len(dataset)}: {sid}, shape={checks['shape']}, "
                  f"label={record['label_max']}, {record['elapsed_seconds']:.1f}s", flush=True)
        state.update(status="complete" if len(state["samples"]) == len(dataset) else "partial",
                     updated_at_utc=datetime.now(timezone.utc).isoformat(),
                     prepared_samples=len(state["samples"]),
                     total_array_bytes=sum(row["size_bytes"] for row in state["samples"]))
        save_progress(output, state)
        print(f"Done: {len(state['samples'])}/{len(dataset)} prepared. Results: {state_path}", flush=True)
        if state["status"] == "partial":
            print("First-example check passed. Add --all to prepare the full pilot.", flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save_progress(output, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--all", action="store_true", help="Prepare all examples; default checks the first one.")
    parser.add_argument("--output-dir", type=Path, help="Optional new subdirectory of experiments/student_data/.")
    args = parser.parse_args()
    root = (EXPERIMENTS / "student_data").resolve()
    output = (args.output_dir or root / str(SIZE) / args.manifest_dir.name).resolve()
    if not output.is_relative_to(root) or output == root:
        raise ValueError("Output must be a subdirectory of experiments/student_data/.")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is preparing this output directory.") from exc
        run(args.manifest_dir.resolve(), output, args.all)


if __name__ == "__main__":
    main()
