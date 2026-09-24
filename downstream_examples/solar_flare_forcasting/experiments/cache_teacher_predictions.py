"""Save frozen-teacher predictions for every example in a manifest.

Run from ~/Surya INSIDE a GPU allocation:
    .venv/bin/python -u downstream_examples/solar_flare_forcasting/experiments/cache_teacher_predictions.py --manifest-dir downstream_examples/solar_flare_forcasting/experiments/manifests/pilot50_20260919T220418_102479Z

Outputs under experiments/teacher_cache/<manifest-name>/:
    predictions.csv   Sample IDs, labels, raw logits, probabilities, and timings.
    cache.json        The same predictions plus provenance and completion status.

Rerun the same command to resume. Completed examples are reused only if the
manifest, image-file list, weights, config, scalers, source, and software versions
match. Each sample is saved atomically. No student training or feature matching
is performed. Raw logits are retained so distillation temperature can be chosen
later. Probabilities and predictions at 0.5 are for inspection only; these are
training examples, not a validation/test accuracy benchmark.
"""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import socket
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from flare_dataset import ManifestFlareDataset, EXPERIMENTS, TASK


FIELDS = [
    "sample_id", "split", "reference_time_utc", "label_max", "teacher_logit",
    "teacher_probability", "prediction_at_0.5", "preparation_seconds",
    "inference_seconds", "peak_allocated_gpu_gib",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def save_cache(directory, cache):
    """JSON is authoritative; the CSV can always be regenerated from it."""
    temporary = directory / "cache.json.tmp"
    temporary.write_text(json.dumps(cache, indent=2, allow_nan=False) + "\n")
    temporary.replace(directory / "cache.json")
    temporary = directory / "predictions.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(cache["predictions"])
    temporary.replace(directory / "predictions.csv")


def validate_cached_rows(rows, manifest_rows):
    expected = {row["sample_id"]: row for row in manifest_rows}
    seen = set()
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in seen or sample_id not in expected:
            raise ValueError(f"Duplicate or unknown cached sample: {sample_id}")
        original = expected[sample_id]
        for key in ("split", "reference_time_utc", "label_max"):
            if str(row[key]) != str(original[key]):
                raise ValueError(f"Cached {key} does not match manifest for {sample_id}")
        logit, probability = float(row["teacher_logit"]), float(row["teacher_probability"])
        if not math.isfinite(logit) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"Invalid cached prediction for {sample_id}")
        exp_value = math.exp(-abs(logit))
        expected_probability = 1 / (1 + exp_value) if logit >= 0 else exp_value / (1 + exp_value)
        if abs(probability - expected_probability) > 1e-7:
            raise ValueError(f"Cached probability/logit mismatch for {sample_id}")
        if row["prediction_at_0.5"] != int(probability > 0.5):
            raise ValueError(f"Cached threshold prediction mismatch for {sample_id}")
        seen.add(sample_id)
    return seen


def get_cache(directory, provenance, manifest_rows):
    path = directory / "cache.json"
    if path.exists():
        cache = json.loads(path.read_text())
        if cache["provenance"] != provenance:
            raise ValueError("Existing cache has different inputs, weights, code, or software. "
                             "Use --output-dir with a new directory under experiments/teacher_cache/.")
        validate_cached_rows(cache["predictions"], manifest_rows)
        return cache
    return {
        "status": "pending", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Teacher targets for the balanced training pilot; not an accuracy benchmark.",
        "provenance": provenance, "expected_examples": len(manifest_rows), "predictions": [],
    }


def run(args, output):
    config_path = TASK / "config_infer.yaml"
    dataset = ManifestFlareDataset(args.manifest_dir / "manifest.csv", config_path)
    if dataset.split != "train":
        raise ValueError("This pilot caching script accepts training examples only.")
    config = yaml.safe_load(config_path.read_text())
    if config["model"]["model_type"] != "spectformer" or config["model"]["learned_flow"]:
        raise ValueError("This script expects the released SpectFormer classifier without learned flow.")
    if config["model"]["img_size"] != dataset.image_size or config["model"]["in_channels"] != len(dataset.channels):
        raise ValueError("Teacher configuration and dataset input dimensions differ.")
    config["pretrained_path"] = str((TASK / config["pretrained_path"]).resolve())
    config["dtype"] = torch.float32
    checkpoint = TASK / "assets/solar_flare_weights.pth"
    # Include the dependencies that define the loaded model and preprocessing.
    repo = TASK.parents[1]
    sources = sorted(set([
        Path(__file__).resolve(), EXPERIMENTS / "flare_dataset.py",
        TASK / "infer.py", TASK / "finetune.py", TASK / "models.py",
        *list((repo / "surya/models").rglob("*.py")),
        *list((repo / "surya/datasets").glob("*.py")),
        repo / "surya/utils/data.py", repo / "surya/utils/misc.py",
    ]))
    print("Checking manifest and fingerprinting weights/configuration for safe resume...", flush=True)
    provenance = {
        "manifest_path": str(dataset.manifest_path), "manifest_sha256": sha256(dataset.manifest_path),
        "file_list_sha256": sha256(args.manifest_dir / "files.csv"),
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "base_weights_sha256": sha256(config["pretrained_path"]),
        "config_sha256": sha256(config_path), "scalers_sha256": sha256(dataset.scalers_path),
        "source_sha256": {str(p.relative_to(repo)): sha256(p) for p in sources},
        "software": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchvision", "peft", "numpy", "xarray", "h5netcdf")},
        "cuda_build": torch.version.cuda, "dtype": "float32", "channels": dataset.channels,
        "input_offsets_hours_before_reference": [1, 0], "seed": 0,
        "tf32": False, "augmentation": False,
    }
    cache = get_cache(output, provenance, dataset.rows)
    done = validate_cached_rows(cache["predictions"], dataset.rows)
    pending = [i for i, row in enumerate(dataset.rows) if row["sample_id"] not in done]
    print(f"Saved: {len(done)}/{len(dataset)}; remaining: {len(pending)}. Output: {output}", flush=True)
    if not pending:
        cache["status"] = "complete"
        save_cache(output, cache)
        print("All examples already cached. Nothing to rerun.", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this command inside a GPU allocation.")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    cache.update(status="running", last_host=socket.gethostname(), last_gpu=torch.cuda.get_device_name(0))
    cache.pop("error", None)
    save_cache(output, cache)
    try:
        # Import lazily: --help and cache-integrity tests do not load the model.
        sys.path.insert(0, str(TASK))
        from infer import load_model

        print("Loading frozen teacher using the existing strict checkpoint loader...", flush=True)
        model = load_model(config, str(checkpoint), torch.device("cuda:0"))
        model.requires_grad_(False)
        model.eval()
        cache["teacher_parameters"] = sum(p.numel() for p in model.parameters())
        cache["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if cache["trainable_parameters"] != 0:
            raise RuntimeError("Teacher parameters must all be frozen.")
        save_cache(output, cache)
        loader = iter(DataLoader(dataset, batch_size=1, sampler=pending, num_workers=0))
        for index in pending:
            expected = dataset.rows[index]
            print(f"Loading {expected['sample_id']}...", flush=True)
            start = time.perf_counter()
            batch, metadata = next(loader)
            sample_id = metadata["sample_id"][0]
            label = int(batch["label"].item())
            if sample_id != expected["sample_id"] or label != int(expected["label_max"]):
                raise ValueError("Loaded example does not match the expected manifest row.")
            inputs = {key: batch[key].to("cuda:0") for key in ("ts", "time_delta_input")}
            del batch
            torch.cuda.synchronize()
            preparation_seconds = time.perf_counter() - start
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.inference_mode():
                logits = model(inputs)
            torch.cuda.synchronize()
            inference_seconds = time.perf_counter() - start
            if logits.numel() != 1 or not torch.isfinite(logits).all().item():
                raise ValueError(f"Teacher returned invalid logits for {sample_id}")
            probability = torch.sigmoid(logits.float()).item()
            row = {
                "sample_id": sample_id, "split": dataset.split,
                "reference_time_utc": expected["reference_time_utc"], "label_max": label,
                "teacher_logit": logits.item(), "teacher_probability": probability,
                "prediction_at_0.5": int(probability > 0.5),
                "preparation_seconds": preparation_seconds, "inference_seconds": inference_seconds,
                "peak_allocated_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
            }
            validate_cached_rows([row], [expected])
            cache["predictions"].append(row)
            cache["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            save_cache(output, cache)
            print(f"Saved {len(cache['predictions'])}/{len(dataset)}: {sample_id}, label={label}, "
                  f"logit={row['teacher_logit']:.5f}, probability={probability:.5f}, "
                  f"prepare={preparation_seconds:.1f}s, infer={inference_seconds:.2f}s", flush=True)
            del inputs, logits
        covered = validate_cached_rows(cache["predictions"], dataset.rows)
        if len(covered) != len(dataset):
            raise RuntimeError("Cache does not cover every manifest example.")
        order = {row["sample_id"]: i for i, row in enumerate(dataset.rows)}
        cache["predictions"].sort(key=lambda row: order[row["sample_id"]])
        cache.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat())
        save_cache(output, cache)
        print(f"Complete: {len(covered)} predictions saved to {output / 'predictions.csv'}", flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        cache.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     error=f"{type(exc).__name__}: {exc}")
        save_cache(output, cache)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Optional new directory inside experiments/teacher_cache/.")
    args = parser.parse_args()
    args.manifest_dir = args.manifest_dir.resolve()
    root = (EXPERIMENTS / "teacher_cache").resolve()
    output = (args.output_dir or root / args.manifest_dir.name).resolve()
    if not output.is_relative_to(root) or output == root:
        raise ValueError("Output must be a subdirectory of experiments/teacher_cache/.")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is writing this cache. Wait for it to finish.") from exc
        run(args, output)


if __name__ == "__main__":
    main()
