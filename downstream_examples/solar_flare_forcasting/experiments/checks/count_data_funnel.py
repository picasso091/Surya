"""Count how the 224 zarr store's rows shrink down to valid (t-1h, t) pairs.
Run: python downstream_examples/solar_flare_forcasting/experiments/checks/count_data_funnel.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import zarr
STORE = Path(__file__).resolve().parents[4] / "surya-bench-224.zarr"

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
zero_example = df[is_zero].iloc[0]
written_example = df[~is_zero & (df.index < zero_example.name)].iloc[-1]  # last real row before it
next_example = df[~is_zero & (df.index > zero_example.name)].iloc[0]      # first real row after it
hole_size = next_example.name - written_example.name - 1                   # zero rows in between
df = df[~is_zero].copy()

# Labels were written as Pacific local time; convert to true UTC.
# Daylight-saving shifts make two rows map to the same true hour.
df["label"] = pd.to_datetime(df["time_ns"], unit="ns", utc=True)
df["true"] = df["label"].dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)
df = df.sort_values("true", kind="stable")
is_dup = df["true"].duplicated(keep=False)
overlap = int(df["true"].duplicated().sum())
overlap_example = df[is_dup].head(2)
true = df["true"].drop_duplicates(keep="last")

# A pair needs a frame exactly one hour earlier
has_prev = (true - pd.Timedelta(hours=1)).isin(true)
no_prev = int((~has_prev).sum())
pairs = int(has_prev.sum())
print(f"1. Total declared rows          : {declared:,}")
print(f"2. Zero-timestamp rows          : {zero:,}")
print(f"3. Daylight-saving overlaps     : {overlap:,}")
print(f"4. Frames with no predecessor   : {no_prev:,}")
print(f"5. Remaining pairs written      : {pairs:,}")

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
print("\nExample daylight-saving overlap (two stored labels, same true hour):")

for r in overlap_example.itertuples():
    print(f"  year={r.year} row={r.row} time_ns={r.time_ns} ns "
        #   stored label={r.label:%Y-%m-%d %H:%M} 
            f"-> {r.true:%Y-%m-%d %H:%M}")

# print(df[["year", "row", "time_ns", "label", "true"]].head(3))