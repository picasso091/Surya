"""Measure how far the 224 zarr time labels are from the true picture times.

Run: python downstream_examples/solar_flare_forcasting/experiments/checks/check_time_offset.py

Terms used:
  true time : when a picture was really taken, in UTC. Native 4K file names are true times.
  label     : the time written next to a picture in the 224 zarr (its time array).
"""

from pathlib import Path

import hdf5plugin
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
import zarr

REPO = Path(__file__).resolve().parents[4]
ZARR_224 = REPO / "surya-bench-224.zarr"
NATIVE_4K = REPO / "downstream_examples/solar_flare_forcasting/assets/original_4k_data_samples"
CHANNEL = "aia171"          # any channel works
OFFSETS = range(-12, 13)
EXACT = 0.999               # correlation above this = same picture


def downsample(img):
    # 4096 -> 224: every output pixel is the mean of a ~19x19 block of native pixels
    x = torch.from_numpy(np.nan_to_num(img.astype(np.float32)))[None, None]
    return F.interpolate(x, size=(224, 224), mode="area")[0, 0].numpy()


def corr(a, b):
    # Pearson correlation of two images: 1.0 means the same picture
    return np.corrcoef(a.ravel(), b.ravel())[0, 1]


def pacific_offset(true_time):
    return -int(true_time.tz_localize("America/Los_Angeles").utcoffset().total_seconds() // 3600)

print("true time : when a picture was really taken (UTC). Native 4K file names are true times.")
print("label     : the time written next to a picture in the 224 zarr.\n")
summary = []

for path in sorted(NATIVE_4K.glob("*.nc")):
    # True time of the native 4K picture, from its file name (e.g. 20190123_0200 -> 02:00 UTC)
    true_time = pd.to_datetime(path.stem, format="%Y%m%d_%H%M")
    print(f"Native 4K {path.name}: taken at {true_time:%Y-%m-%d %H:%M} UTC ")

    # Read every label in that year's 224 zarr (row number = position in this list)
    group = zarr.open(str(ZARR_224 / str(true_time.year) / "dataset"), mode="r")
    labels = pd.to_datetime(np.asarray(group["time"][:]), unit="ns")
    c = list(group["images"].attrs["channel_names"]).index(CHANNEL)

    # Load the native 4K picture and shrink it to 224 so the two can be compared
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        native_img = downsample(ds[CHANNEL].to_numpy())

    # For each offset h, find the 224 row labelled (true time + h) and compare pictures
    results = []
    for h in OFFSETS:
        for row in np.flatnonzero(labels == true_time + pd.Timedelta(hours=h)):
            results.append((h, row, labels[row], corr(native_img, np.asarray(group["images"][row, c]))))
    if not results:
        print("  skipped: no 224 label within +-12 h (224 labels are all on the hour)\n")
        continue

    # Print every comparison; the highest correlation is the same picture
    best = max(results, key=lambda r: r[3])
    print(f"  {'224 row':>7}   {'label in 224 (UTC)':<18}   {'offset':>12}   {'corr':>7}")
    for h, row, label, r in results:
        mark = "   <-- same picture" if row == best[1] and r > EXACT else ""
        print(f"  {row:7d}   {label:%Y-%m-%d %H:%M}     {h:+10d} h   {r:7.5f}{mark}")

    # State the result in plain words
    h, row, label, r = best
    expected = pacific_offset(true_time)
    if r > EXACT:
        print(f"  Result: the picture taken at {true_time:%H:%M} UTC is in 224 resolution, row {row}, "
              f"labelled {label:%H:%M}.")
        print(f"          The 224 label is {h:+d} h from the true time, so true = label {-h:+d} h.\n")
    else:
        print(f"  Result: no exact match (best corr {r:.5f} at {h:+d} h).\n")
    summary.append((path.name, true_time, row, label, h, r, expected))

print("\n### Summary ###")
print(f"  {'native 4K file':<18}  {'true time (UTC)':<16}  {'224 row':>7}  {'its label':<16}  {'offset':>6}  {'corr':>7}")
for name, true_time, row, label, h, r, expected in summary:
    print(f"  {name:<18}  {true_time:%Y-%m-%d %H:%M}  {row:7d}  {label:%Y-%m-%d %H:%M}  {f'{h:+d} h':>6}  {r:7.5f}")
