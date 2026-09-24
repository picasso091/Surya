"""
Download HMI magnetogram samples from Helioviewer at 1024x1024.

Two modes:

  --mode screenshot  (default)
      Uses /v2/takeScreenshot/ to render server-side at 1k. Returns PNG,
      ~300 KB per frame. Downloads 1/20th the bytes of the full JP2.

  --mode jp2
      Downloads the full 4096x4096 JP2 and decodes it at wavelet reduction
      level 2 (-> 1024x1024). Keeps the original file if --keep-jp2 is set.
      Same bytes over the wire, but no full-resolution decode.

Layout matches the original script:
    basedir/YYYY/MM/DD/HMI.mYYYY.MM.DD_HH.MM.SS.{png,jp2}

Usage:
    python download_samples_1k.py --n 10
    python download_samples_1k.py --n 10 --mode jp2 --keep-jp2
"""

import argparse
import datetime
import time
from pathlib import Path

import numpy as np
import requests

API = "https://api.helioviewer.org"
SOURCE_ID = 19  # SDO/HMI magnetogram

# Helioviewer's native plate scale is 0.60511 arcsec/px, which is what the
# 4096x4096 JP2 is stored at. Multiplying by 4 gives the same field of view
# in 1024 px, so the 1k frame covers exactly what the 4k one does.
NATIVE_SCALE = 0.60511
JP2_MAGIC = (b"\x00\x00\x00\x0cjP", b"\xff\x4f\xff\x51")


def resolve_timestamp(dt, timeout=60):
    """Ask for the JPIP URI and parse out the timestamp actually available."""
    r = requests.get(
        f"{API}/v2/getJP2Image/",
        params={"date": f"{dt.date()}T{dt.time()}Z", "sourceId": SOURCE_ID, "jpip": "true"},
        timeout=timeout,
    )
    r.raise_for_status()
    stamp = r.text.strip().rsplit("/", 1)[-1].rsplit("__", 1)[0][:-4]
    return datetime.datetime.strptime(stamp, "%Y_%m_%d__%H_%M_%S")


def outpath(basedir, received, ext):
    d = Path(basedir) / f"{received.year}" / f"{received.month:02d}" / f"{received.day:02d}"
    d.mkdir(parents=True, exist_ok=True)
    name = (
        f"HMI.m{received.year}.{received.month:02d}.{received.day:02d}_"
        f"{received.hour:02d}.{received.minute:02d}.{received.second:02d}.{ext}"
    )
    return d / name


def fetch_screenshot(received, basedir, size, timeout=120):
    """Server-side render at `size` px square, full disk, no watermark."""
    params = {
        "date": f"{received.date()}T{received.time()}Z",
        "imageScale": NATIVE_SCALE * (4096 / size),
        "layers": f"[{SOURCE_ID},1,100]",
        "x0": 0,
        "y0": 0,
        "width": size,
        "height": size,
        "display": "true",
        "watermark": "false",  # keep burned-in text out of training data
        "events": "",
        "eventLabels": "false",
        "scale": "false",
    }
    r = requests.get(f"{API}/v2/takeScreenshot/", params=params, timeout=timeout)
    r.raise_for_status()
    if not r.content.startswith(b"\x89PNG"):
        print(f"  skip {received} — not a PNG ({r.content[:120]!r})")
        return None
    path = outpath(basedir, received, "png")
    path.write_bytes(r.content)
    print(f"  {path}  ({len(r.content) / 1e3:.0f} KB)")
    return path


def fetch_jp2_reduced(received, basedir, size, keep_jp2, timeout=120):
    """Download the 4k JP2, decode it at a reduced resolution level."""
    import glymur

    r = requests.get(
        f"{API}/v2/getJP2Image/",
        params={"date": f"{received.date()}T{received.time()}Z", "sourceId": SOURCE_ID},
        timeout=timeout,
    )
    r.raise_for_status()
    if not r.content.startswith(JP2_MAGIC):
        print(f"  skip {received} — not a JP2 ({r.content[:120]!r})")
        return None

    jp2_path = outpath(basedir, received, "jp2")
    jp2_path.write_bytes(r.content)

    step = 4096 // size  # 4 for 1024
    j = glymur.Jp2k(str(jp2_path))
    arr = j[::step, ::step]  # decodes only the levels it needs

    png_path = jp2_path.with_suffix(".png")
    glymur_free = np.ascontiguousarray(arr)
    import cv2

    cv2.imwrite(str(png_path), glymur_free)

    if not keep_jp2:
        jp2_path.unlink()
    print(f"  {png_path}  {arr.shape}, dtype {arr.dtype}  (source {len(r.content) / 1e6:.1f} MB)")
    return png_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--start", default="2011-01-20 17:00:00")
    ap.add_argument("--cadence", type=int, default=12, help="minutes between requests")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--mode", choices=["screenshot", "jp2"], default="screenshot")
    ap.add_argument("--keep-jp2", action="store_true", help="jp2 mode: keep the 4k original")
    ap.add_argument("--basedir", default="./hmi_samples_1k")
    ap.add_argument("--sleep", type=float, default=1.0, help="pause between requests")
    args = ap.parse_args()

    dt = datetime.datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
    step = datetime.timedelta(minutes=args.cadence)
    tol = datetime.timedelta(minutes=args.cadence)

    got, attempts, seen = 0, 0, set()
    while got < args.n and attempts < args.n * 5:
        attempts += 1
        try:
            received = resolve_timestamp(dt)
            if abs(received - dt) > tol:
                print(f"  skip {dt} — nearest frame is {abs(received - dt)} away")
            elif received in seen:
                print(f"  skip {dt} — duplicate of {received}")
            else:
                seen.add(received)
                if args.mode == "screenshot":
                    ok = fetch_screenshot(received, args.basedir, args.size)
                else:
                    ok = fetch_jp2_reduced(received, args.basedir, args.size, args.keep_jp2)
                got += bool(ok)
        except requests.RequestException as e:
            print(f"  request failed at {dt}: {e}")
        dt += step
        time.sleep(args.sleep)

    print(f"\nDownloaded {got} file(s) into {args.basedir}")


if __name__ == "__main__":
    main()
