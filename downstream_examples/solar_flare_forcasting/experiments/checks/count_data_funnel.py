"""Count how the 224 zarr store's rows shrink down to valid (t-1h, t) pairs.
Run: python downstream_examples/solar_flare_forcasting/experiments/checks/count_data_funnel.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import zarr
STORE = Path(__file__).resolve().parents[4] / "surya-bench-224-interarea.zarr"

# Read the time array of every year file
frames = []
for year_dir in sorted(STORE.glob("*/dataset")):
    t = np.asarray(zarr.open(str(year_dir), mode="r")["time"][:])
    frames.append(pd.DataFrame({"year": year_dir.parent.name, "row": np.arange(len(t)), "time_ns": t}))
    
df = pd.concat(frames, ignore_index=True)
declared = len(df)

# Unwritten rows read back as timestamp 0 (1970-01-01)
is_zero = df["time_ns"] == 0
zero = int(is_zero.sum())
if zero:  # the interarea store has no holes
    zero_example = df[is_zero].iloc[0]
    written_example = df[~is_zero & (df.index < zero_example.name)].iloc[-1]  # last real row before it
    next_example = df[~is_zero & (df.index > zero_example.name)].iloc[0]      # first real row after it
    hole_size = next_example.name - written_example.name - 1                   # zero rows in between
df = df[~is_zero].copy()

# Labels are already true UTC (checks/check_time_offset.py: offset +0 h), so no time-zone correction.
df["true"] = pd.to_datetime(df["time_ns"], unit="ns")
df = df.sort_values("true", kind="stable").reset_index(drop=True)
is_dup = df["true"].duplicated(keep=False)
overlap = int(df["true"].duplicated().sum())
overlap_example = df[is_dup].head(2)
df = df.drop_duplicates("true", keep="last").reset_index(drop=True)
true = df["true"]

# A pair needs a frame exactly one hour earlier
has_prev = (true - pd.Timedelta(hours=1)).isin(true)
no_prev = int((~has_prev).sum())
pairs = int(has_prev.sum())
print(f"1. Total declared rows          : {declared:,}")
print(f"2. Zero-timestamp rows          : {zero:,}")
print(f"3. Duplicate timestamps         : {overlap:,}")
print(f"4. Frames with no predecessor   : {no_prev:,}")
print(f"5. Remaining pairs written      : {pairs:,}")

if zero:
    print("\nExample zero-timestamp row:")
    print(f"  year={zero_example.year} row={zero_example.row} time_ns={zero_example.time_ns} "
          f"-> {pd.to_datetime(zero_example.time_ns, unit='ns')}")
    print("\nFor comparison, the last written row before it:")
    print(f"  year={written_example.year} row={written_example.row} time_ns={written_example.time_ns} ns "
          f"-> {pd.to_datetime(written_example.time_ns, unit='ns')}")
    print("\nAnd the first written row after it:")
    print(f"  year={next_example.year} row={next_example.row} time_ns={next_example.time_ns} ns "
          f"-> {pd.to_datetime(next_example.time_ns, unit='ns')}")
    print(f"\nSo this hole is {hole_size} zero rows long.")
if overlap:
    print("\nExample duplicate timestamp (two rows, same hour):")
    for r in overlap_example.itertuples():
        print(f"  year={r.year} row={r.row} time_ns={r.time_ns} ns -> {r.true:%Y-%m-%d %H:%M}")

# Example frame with no frame exactly 1 h earlier (skip the very first frame: nothing can precede it)
orphan = df[~has_prev & (df.index > 0)].iloc[0]
before = df.iloc[orphan.name - 1]                                   # closest earlier frame
gap = orphan.true - before.true
print("\nExample frame with no frame exactly 1 h earlier:")
print(f"  year={orphan.year} row={orphan.row} time_ns={orphan.time_ns} ns -> {orphan.true:%Y-%m-%d %H:%M}")
print("The closest earlier frame:")
print(f"  year={before.year} row={before.row} time_ns={before.time_ns} ns -> {before.true:%Y-%m-%d %H:%M}")
print(f"So the gap is {gap}: {int(gap / pd.Timedelta(hours=1)) - 1} hour(s) missing in between.")

# print(df[["year", "row", "time_ns", "label", "true"]].head(3))