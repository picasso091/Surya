"""Download and validate the images listed in a pilot manifest.

Run from ~/Surya (CPU only). Start with the first example, i.e. two images:
    .venv/bin/python downstream_examples/solar_flare_forcasting/experiments/download_pilot_data.py --manifest-dir downstream_examples/solar_flare_forcasting/experiments/manifests/pilot50_20260919T220418_102479Z

After reviewing that check, download all selected examples with the same command
plus --all. Already downloaded files are validated and reused. Interrupted
downloads resume from a .part file tied to the URL, size, and remote ETag.

Outputs: images at the manifest's local paths under experiments/data/core_sdo/;
a timestamped reports/<manifest-name>/download_checks/<run>/results.json under experiments/.
Validation reads all 13 channels, checks 4096x4096 float32 arrays, finite values,
the embedded data_time attribute, and exact byte size. ETags identify the remote
version; multipart ETags are NOT treated as MD5 checksums. No GPU is needed.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from http.client import IncompleteRead
import json
from pathlib import Path
import re
import shutil
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import xarray as xr


EXPERIMENTS = Path(__file__).resolve().parent
CHANNELS = [
    "aia94", "aia131", "aia171", "aia193", "aia211", "aia304", "aia335", "aia1600",
    "hmi_m", "hmi_bx", "hmi_by", "hmi_bz", "hmi_v",
]


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def partial_path(row):
    identity = "|".join(row[k] for k in ("url", "etag", "size_bytes"))
    suffix = hashlib.sha256(identity.encode()).hexdigest()[:12]
    return Path(row["local_path"] + f".{suffix}.part")


def download(row):
    final = Path(row["local_path"])
    size = int(row["size_bytes"])
    if final.exists():
        if final.stat().st_size != size:
            raise ValueError(f"Existing file has the wrong size; inspect it before retrying: {final}")
        print(f"Reusing {final.name}; validating contents again.", flush=True)
        return final, True
    final.parent.mkdir(parents=True, exist_ok=True)
    part = partial_path(row)
    for attempt in range(3):
        offset = part.stat().st_size if part.exists() else 0
        if offset > size:
            raise ValueError(f"Partial file exceeds expected size: {part}")
        if offset == size:
            return part, False
        headers = {"If-Match": f'"{row["etag"]}"'}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urlopen(Request(row["url"], headers=headers), timeout=60) as response:
                if response.headers.get("ETag", "").strip('"') != row["etag"]:
                    raise ValueError("Remote ETag differs from manifest; rebuild the manifest.")
                if response.status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or tuple(map(int, match.groups())) != (offset, size - 1, size):
                        raise ValueError("Unexpected HTTP range response; refusing to append.")
                    mode = "ab" if offset else "wb"
                elif response.status == 200:
                    offset, mode = 0, "wb"  # Server ignored Range: safely restart.
                else:
                    raise ValueError(f"Unexpected HTTP status {response.status}")
                if int(response.headers["Content-Length"]) != size - offset:
                    raise ValueError("Remote response size differs from manifest.")
                print(f"Downloading {final.name}: {offset / 10**6:.1f}/{size / 10**6:.1f} MB", flush=True)
                last_update = time.monotonic()
                with part.open(mode) as handle:
                    while True:
                        chunk = response.read(4 * 1024**2)
                        if not chunk:
                            break
                        handle.write(chunk)
                        offset += len(chunk)
                        if time.monotonic() - last_update >= 5:
                            print(f"  {final.name}: {offset / size:.0%}", flush=True)
                            last_update = time.monotonic()
                if part.stat().st_size != size:
                    raise IncompleteRead(b"", size - part.stat().st_size)
                return part, False
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {exc.code} for {row['url']}; download stopped.") from exc
            error = exc
        except (URLError, TimeoutError, ConnectionError, IncompleteRead) as exc:
            error = exc
        print(f"Transfer interrupted (attempt {attempt + 1}/3): {error}", flush=True)
        if attempt < 2:
            time.sleep(attempt + 1)
    raise RuntimeError(f"Download incomplete; rerun to resume: {part}")


def validate(path, row, expected_time):
    if path.stat().st_size != int(row["size_bytes"]):
        raise ValueError(f"Wrong size: {path}")
    expected_stamp = datetime.fromisoformat(expected_time.replace("Z", "+00:00")).strftime("%Y%m%d_%H%M")
    if Path(row["s3_key"]).stem != expected_stamp:
        raise ValueError("Manifest input timestamp does not match its image filename.")
    stats = {}
    with xr.open_dataset(path, engine="h5netcdf", chunks=None, cache=False) as dataset:
        data_time = str(dataset.attrs.get("data_time", ""))
        if data_time != expected_stamp:
            raise ValueError(f"Embedded data_time={data_time!r}, expected {expected_stamp!r}")
        for channel in CHANNELS:
            if channel not in dataset:
                raise ValueError(f"Missing channel: {channel}")
            variable = dataset[channel]
            if variable.dims != ("y", "x") or variable.shape != (4096, 4096):
                raise ValueError(f"Unexpected dimensions for {channel}: {variable.dims}, {variable.shape}")
            values = variable.to_numpy()  # Read the full array to detect corrupt/truncated data.
            if values.dtype != np.dtype("float32"):
                raise ValueError(f"Unexpected dtype for {channel}: {values.dtype}")
            nonfinite = int(values.size - np.count_nonzero(np.isfinite(values)))
            if nonfinite:
                raise ValueError(f"{channel} contains {nonfinite} non-finite values; inspect before training.")
            stats[channel] = {
                "shape": list(values.shape), "dtype": str(values.dtype),
                "nonfinite": nonfinite, "min": float(values.min()), "max": float(values.max()),
            }
            del values
    return {"data_time": data_time, "channels": stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--all", action="store_true", help="Download all examples; default is the first example only.")
    args = parser.parse_args()
    directory = args.manifest_dir.resolve()
    rows = read_csv(directory / "manifest.csv")
    if not rows:
        raise ValueError("Manifest is empty.")
    samples = rows if args.all else rows[:1]
    expected_times = {}
    for sample in samples:
        for name in ("previous", "current"):
            key, stamp = sample[f"{name}_s3_key"], sample[f"{name}_time_utc"]
            if key in expected_times and expected_times[key] != stamp:
                raise ValueError(f"Conflicting times for {key}")
            expected_times[key] = stamp
    file_index = read_csv(directory / "files.csv")
    if len({r["s3_key"] for r in file_index}) != len(file_index):
        raise ValueError("Duplicate keys in files.csv")
    files = [r for r in file_index if r["s3_key"] in expected_times]
    if len(files) != len(expected_times):
        raise ValueError("files.csv does not include every required input.")
    data_root = (EXPERIMENTS / "data" / "core_sdo").resolve()
    for row in files:
        expected_path = (data_root / row["s3_key"]).resolve()
        if not expected_path.is_relative_to(data_root) or Path(row["local_path"]).resolve() != expected_path:
            raise ValueError(f"Image destination must match experiments/data/core_sdo/<key>: {row['local_path']}")
        if row["url"] != f"https://nasa-surya-bench.s3.us-west-2.amazonaws.com/{row['s3_key']}":
            raise ValueError(f"Unexpected download URL: {row['url']}")
        if int(row["size_bytes"]) <= 0 or not row["etag"]:
            raise ValueError("Manifest requires positive sizes and ETags.")
    total = sum(int(r["size_bytes"]) for r in files)
    needed = sum(max(0, int(r["size_bytes"]) - (partial_path(r).stat().st_size if partial_path(r).exists() else 0))
                 for r in files if not Path(r["local_path"]).exists())
    if shutil.disk_usage(EXPERIMENTS).free < needed + 1024**3:
        raise RuntimeError("Insufficient filesystem free space for the selected download plus 1 GiB margin.")
    output = EXPERIMENTS / "reports" / directory.name / "download_checks" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running", "manifest_dir": str(directory),
        "samples": [{"sample_id": r["sample_id"], "label_max": int(r["label_max"])} for r in samples],
        "file_count": len(files), "total_bytes": total, "results": [],
        "scope": "File integrity and model-input compatibility; no model inference or training.",
    }

    def save():
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    print(f"Checking {len(samples)} example(s), {len(files)} files, {total / 10**9:.2f} GB total.", flush=True)
    print(f"Results: {output / 'results.json'}", flush=True)
    try:
        for row in files:
            start = time.monotonic()
            path, reused = download(row)
            print(f"Validating all channels in {Path(row['local_path']).name}...", flush=True)
            result = validate(path, row, expected_times[row["s3_key"]])
            final = Path(row["local_path"])
            if not reused:
                path.rename(final)
            report["results"].append({
                "s3_key": row["s3_key"], "local_path": str(final), "size_bytes": int(row["size_bytes"]),
                "reused": reused, "elapsed_seconds": time.monotonic() - start, **result,
            })
            save()
            print(f"Passed: {final.name}", flush=True)
        report["status"] = "passed"
        print("Download and validation passed for all selected files.", flush=True)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
