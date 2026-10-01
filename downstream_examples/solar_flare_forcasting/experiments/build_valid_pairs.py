"""Build valid_pairs_224.csv: every (t-1h, t) teacher sample the 224 zarr store can supply.
Run: python downstream_examples/solar_flare_forcasting/experiments/build_valid_pairs.py
Pairs are matched on corrected times.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import zarr

HOUR_NS = 3600 * 10**9
REPO = Path(__file__).resolve().parents[3]
STORE = REPO / "surya-bench-224.zarr"
OUTPUT = Path(__file__).resolve().parent / "valid_pairs_224.csv"
COLUMNS = ["reference_time_utc", "sample_id", "prev_year", "prev_row", "curr_year", "curr_row"]
# Pixel-verified rows, used as a final sanity check on the time correction.
ANCHORS = {"2019-01-23 02:00": 490, "2019-01-23 03:00": 491, "2019-01-23 04:00": 492}


def true_utc(labels):
    """Stored label -> true UTC (reinterpret the label as Pacific local time)."""
    return (pd.DatetimeIndex(labels).tz_convert("America/Los_Angeles")
              .tz_localize(None).tz_localize("UTC"))


def main():
    years = sorted(p.parent.parent.name for p in STORE.glob("*/dataset/.zgroup"))
    if not years:
        raise SystemExit(f"No year groups found under {STORE}")

    # Load time arrays, drop holes (timestamp 0), correct labels to true UTC.
    frames = []
    for year in years:
        stored = np.asarray(zarr.open(str(STORE / year / "dataset"), mode="r")["time"][:])
        written = np.flatnonzero(stored != 0)
        frames.append(pd.DataFrame({
            "year": year, "row": written,
            "true": true_utc(pd.to_datetime(stored[written], unit="ns", utc=True)),
            "declared": len(stored), "holes": len(stored) - len(written),
        }))
    catalogue = pd.concat(frames, ignore_index=True)

    # Per-year counts of declared rows, holes and real frames.
    inventory = (catalogue.groupby("year")
                 .agg(declared_rows=("declared", "first"), holes=("holes", "first"),
                      real_frames=("row", "size")))
    print("Per year-file inventory:")
    print(inventory.to_string())
    print(f"{'TOTAL':<6} declared={inventory.declared_rows.sum()} "
          f"holes={inventory.holes.sum()} real={inventory.real_frames.sum()}\n")

    # Drop DST collisions, keeping the later (correct) row.
    catalogue = catalogue.sort_values(["true", "year", "row"], kind="stable")
    collisions = int(catalogue["true"].duplicated().sum())
    catalogue = catalogue.drop_duplicates("true", keep="last").reset_index(drop=True)
    print(f"frames dropped as unnameable daylight saving collisions: {collisions}")
    print(f"distinct true timestamps: {len(catalogue)}\n")

    # Pair each frame with the frame exactly 1 h earlier; unmatched frames are dropped.
    position = pd.Series(catalogue.index, index=catalogue["true"])
    previous = position.reindex(catalogue["true"] - pd.Timedelta(hours=1)).to_numpy()
    has_previous = ~pd.isna(previous)
    current = catalogue[has_previous].reset_index(drop=True)
    earlier = catalogue.iloc[previous[has_previous].astype(int)].reset_index(drop=True)

    # One row per sample: reference time + zarr addresses of both input frames.
    index = pd.DataFrame({
        "reference_time_utc": current["true"].map(lambda t: t.isoformat()),
        "sample_id": current["true"].dt.strftime("%Y%m%d_%H%M"),
        "prev_year": earlier["year"], "prev_row": earlier["row"],
        "curr_year": current["year"], "curr_row": current["row"],
    })[COLUMNS]

    # Sanity checks: unique, sorted, every pair exactly 60 min apart.
    stamps = pd.DatetimeIndex(current["true"])
    assert stamps.is_monotonic_increasing and not stamps.duplicated().any()
    gaps = (stamps - pd.DatetimeIndex(earlier["true"])).total_seconds()
    assert (gaps == 3600).all(), "a pair is not exactly 60 minutes apart"

    index.to_csv(OUTPUT, index=False)
    print(f"wrote {OUTPUT.name}: {len(index)} pairs from {len(catalogue)} frames "
          f"({100 * len(index) / len(catalogue):.1f}% paired)")
    print(f"span {stamps.min()}  ->  {stamps.max()}")
    print(f"pairs spanning two year files: {int((index.prev_year != index.curr_year).sum())}\n")
    print("pairs per calendar year:")
    print(stamps.year.value_counts().sort_index().to_string())

    print("\nend-to-end check against the pixel-verified native frames:")
    for stamp, expected in ANCHORS.items():
        hit = index[index.reference_time_utc == pd.Timestamp(stamp, tz="UTC").isoformat()]
        got = int(hit.iloc[0].curr_row) if not hit.empty else None
        print(f"  {stamp}  curr_row={got}  expected {expected}  "
              f"{'ok' if got == expected else 'MISMATCH'}")


if __name__ == "__main__":
    main()
