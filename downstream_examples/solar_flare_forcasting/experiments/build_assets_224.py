"""Build assets_224/: the flare label splits restricted to timestamps in valid_pairs_224.csv.
Run: python downstream_examples/solar_flare_forcasting/experiments/build_assets_224.py
"""

from pathlib import Path

import pandas as pd

TASK = Path(__file__).resolve().parents[1]
LABELS = TASK / "assets/surya-bench-flare-forecasting"
PAIRS = TASK / "experiments/valid_pairs_224.csv"
OUTPUT = TASK / "assets_224"
SPLITS = ["train", "validation", "leaky_validation", "test", "data"]


def main():
    pairs = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(PAIRS)["timestamp"])).tz_localize("UTC")
    OUTPUT.mkdir(exist_ok=True)

    for split in SPLITS:
        labels = pd.read_csv(LABELS / f"{split}.csv")
        stamps = pd.DatetimeIndex(pd.to_datetime(labels["timestamp"])).tz_localize("UTC")
        kept = labels[stamps.isin(pairs)]
        kept.to_csv(OUTPUT / f"{split}.csv", index=False)
        print(f"{split}: {len(kept)}/{len(labels)} rows, {kept['label_max'].sum()} positive")


if __name__ == "__main__":
    main()
