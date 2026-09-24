"""Select 50 training examples and check remote sizes WITHOUT downloading images.

Run from the repository root (~/Surya); no GPU or AWS credentials are needed:
    .venv/bin/python downstream_examples/solar_flare_forcasting/experiments/build_pilot_manifest.py

Outputs: experiments/manifests/pilot50_<UTC time>/
    manifest.csv       One row per forecast: label, two inputs, remote/local paths.
    files.csv          Unique image files and exact remote sizes in bytes.
    summary.json       Selection rules, counts, source hashes, and total size.
    missing_files.csv  Remote files that returned 404 while selecting examples.

Rules: seed 42; five positives and five negatives in each year 2011--2015;
at least 48 hours between all selected forecast times. The entire interval from
t-1h through t+24h must remain within contiguous official training-label hours
(a conservative boundary check). Both inputs must appear as present in the
training image index and exist remotely. A missing remote input causes a
candidate to be skipped; permission/network errors stop the run.

This deliberately balanced set is for checking the pipeline, NOT evaluating
forecast skill. Images are only checked using HTTP HEAD. Local image paths
describe a future download destination; no image directories/files are created.
The current inherited flare loader must be adapted before using this two-input
manifest: it still requires an unused t+1h image, which we do not include here.
"""

import argparse
import csv
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


EXPERIMENTS = Path(__file__).resolve().parent
ASSETS = EXPERIMENTS.parent / "assets"
BUCKET = "nasa-surya-bench"
BASE_URL = f"https://{BUCKET}.s3.us-west-2.amazonaws.com"
YEARS = range(2011, 2016)
PER_CLASS_PER_YEAR = 5
HOUR = timedelta(hours=1)
MIN_GAP = timedelta(hours=48)


def read_labels(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = {}
    for row in rows:
        timestamp = datetime.fromisoformat(row["timestamp"])
        label = int(row["label_max"])
        if timestamp in labels or label not in (0, 1):
            raise ValueError(f"Duplicate timestamp or invalid label: {row}")
        labels[timestamp] = label
    return labels


def build_candidates(label_path, index_path):
    labels = read_labels(label_path)
    images = {}
    with index_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            timestamp = datetime.fromisoformat(row["timestep"])
            if timestamp.year not in YEARS or int(row["present"]) != 1:
                continue
            key = row["path"]
            if key != timestamp.strftime("%Y/%m/%Y%m%d_%H%M.nc"):
                raise ValueError(f"Unexpected image path layout: {key}")
            if timestamp in images:
                raise ValueError(f"Duplicate image timestamp: {timestamp}")
            images[timestamp] = key
    candidates = {(year, label): [] for year in YEARS for label in (0, 1)}
    for timestamp, label in sorted(labels.items()):
        if timestamp.year not in YEARS:
            continue
        if any(timestamp + offset * HOUR not in labels for offset in range(-1, 25)):
            continue
        input_times = (timestamp - HOUR, timestamp)
        if all(t in images for t in input_times):
            candidates[timestamp.year, label].append({
                "timestamp": timestamp,
                "label": label,
                "input_keys": [images[t] for t in input_times],
            })
    return candidates


def remote_metadata(key):
    """Return remote size/ETag; return None only for an explicitly missing object."""
    url = f"{BASE_URL}/{quote(key, safe='/')}"
    for attempt in range(3):
        try:
            with urlopen(Request(url, method="HEAD"), timeout=30) as response:
                size = int(response.headers["Content-Length"])
                if size <= 0:
                    raise ValueError(f"Empty remote object: {key}")
                return {
                    "size_bytes": size,
                    "etag": response.headers.get("ETag", "").strip('"'),
                    "last_modified": response.headers.get("Last-Modified", ""),
                }
        except HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {exc.code} checking {url}") from exc
            error = exc
        except (URLError, TimeoutError, OSError) as exc:
            error = exc
        if attempt < 2:
            time.sleep(attempt + 1)
    raise RuntimeError(f"Unable to check {url} after 3 attempts: {error}")


def select_examples(candidates, seed, lookup=remote_metadata):
    rng = random.Random(seed)
    selected, checked, missing = [], {}, []
    # Select the rarer positive class first, then negatives, within each year.
    for year in YEARS:
        for label in (1, 0):
            pool = list(candidates[year, label])
            rng.shuffle(pool)
            count = 0
            for example in pool:
                timestamp = example["timestamp"]
                if any(abs(timestamp - old["timestamp"]) < MIN_GAP for old in selected):
                    continue
                available = True
                for key in example["input_keys"]:
                    if key not in checked:
                        checked[key] = lookup(key)
                        if checked[key] is None:
                            missing.append({"s3_key": key, "reason": "HTTP 404"})
                    if checked[key] is None:
                        available = False
                if not available:
                    print(f"Skipping {timestamp}: missing remote input.", flush=True)
                    continue
                selected.append(example)
                count += 1
                print(f"Selected {len(selected):2d}/50: {timestamp} label={label}", flush=True)
                if count == PER_CLASS_PER_YEAR:
                    break
            if count != PER_CLASS_PER_YEAR:
                raise RuntimeError(
                    f"Found only {count}/5 examples for year={year}, label={label}. "
                    "The selection rules were not relaxed; no final manifest was written."
                )
    return sorted(selected, key=lambda x: x["timestamp"]), checked, missing


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    label_path = ASSETS / "surya-bench-flare-forecasting/train.csv"
    index_path = ASSETS / "train_index_surya_1_0.csv"
    output = EXPERIMENTS / "manifests" / ("pilot50_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    image_root = EXPERIMENTS / "data" / "core_sdo"
    summary = {"status": "running", "seed": args.seed, "purpose": "Balanced pipeline check only"}
    print(f"Output directory: {output}", flush=True)
    print("Checking metadata only; no images will be downloaded.", flush=True)
    try:
        candidates = build_candidates(label_path, index_path)
        summary.update({
            "source_files": {str(p): sha256(p) for p in (label_path, index_path)},
            "rules": {
                "years": list(YEARS), "per_class_per_year": PER_CLASS_PER_YEAR,
                "minimum_spacing_hours": 48, "input_offsets_minutes": [-60, 0],
                "label": "label_max", "label_horizon_hours": 24,
                "split": "train", "boundary_check": "Every hour t-1h through t+24h exists in train.csv",
                "remote_check": "HTTP HEAD; both objects must exist and have positive sizes",
            },
            "eligible_counts": {f"{year}_label_{label}": len(pool) for (year, label), pool in candidates.items()},
        })
        print("Eligible candidates:", summary["eligible_counts"], flush=True)
        selected, checked, missing = select_examples(candidates, args.seed)
        manifest = []
        usage = Counter(key for example in selected for key in example["input_keys"])
        files = []
        for key in sorted(usage):
            files.append({
                "s3_key": key, "s3_uri": f"s3://{BUCKET}/{key}",
                "url": f"{BASE_URL}/{quote(key, safe='/')}",
                "local_path": str(image_root / PurePosixPath(key)),
                **checked[key], "sample_references": usage[key],
            })
        for example in selected:
            timestamp = example["timestamp"]
            row = {
                "sample_id": timestamp.strftime("%Y%m%d_%H%M"), "split": "train",
                "reference_time_utc": timestamp.isoformat() + "Z", "year": timestamp.year,
                "label_max": example["label"],
                "label_window_end_utc": (timestamp + 24 * HOUR).isoformat() + "Z",
            }
            for name, offset, key in zip(("previous", "current"), (-1, 0), example["input_keys"]):
                row[f"{name}_time_utc"] = (timestamp + offset * HOUR).isoformat() + "Z"
                row[f"{name}_s3_key"] = key
                row[f"{name}_local_path"] = str(image_root / PurePosixPath(key))
                row[f"{name}_size_bytes"] = checked[key]["size_bytes"]
            manifest.append(row)
        total = sum(row["size_bytes"] for row in files)
        summary.update({
            "status": "complete", "examples": len(manifest),
            "positives": sum(row["label_max"] for row in manifest),
            "negatives": sum(1 - row["label_max"] for row in manifest),
            "unique_files": len(files), "total_remote_bytes": total,
            "total_remote_GB": total / 10**9, "total_remote_GiB": total / 1024**3,
            "remote_objects_checked": len(checked), "missing_objects": len(missing),
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "notes": [
                "Sizes cover selected remote NetCDF files only, excluding future caches/checkpoints.",
                "ETags are recorded as identifiers, not assumed to be MD5 checksums.",
                "HEAD does not verify NetCDF readability/channels; check those after downloading.",
                "48-hour spacing avoids overlapping selected label windows, but not all event dependence.",
                "The existing loader requires adaptation to avoid reading the unused t+1h image.",
            ],
        })
        write_csv(output / "manifest.csv", manifest, list(manifest[0]))
        write_csv(output / "files.csv", files, list(files[0]))
        write_csv(output / "missing_files.csv", missing, ["s3_key", "reason"])
        print(f"\n50 examples: 25 positive, 25 negative; {len(files)} unique images.")
        print(f"Exact remote total: {total:,} bytes ({total / 10**9:.2f} GB; {total / 1024**3:.2f} GiB).")
        print(f"Inspect manifest.csv, files.csv, and summary.json in {output}")
    except Exception as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
