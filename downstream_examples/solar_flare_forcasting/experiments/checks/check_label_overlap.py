"""How many surya-bench-flare-forecasting timestamps are covered by valid_pairs_224.csv.
Run: python downstream_examples/solar_flare_forcasting/experiments/checks/check_label_overlap.py
"""

from pathlib import Path

import pandas as pd

TASK = Path(__file__).resolve().parents[2]
LABELS = TASK / "assets/surya-bench-flare-forecasting"
PAIRS = TASK / "experiments/valid_pairs_224.csv"
SPLITS = ["train", "validation", "leaky_validation", "test", "data"]


def main():
    pairs = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(PAIRS)["timestamp"])).tz_localize("UTC")
    print(f"valid_pairs_224: {len(pairs)} samples, {pairs.min()} -> {pairs.max()}")

    # Every valid pair should have a label in the full asset file (data.csv).
    asset = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(LABELS / "data.csv")["timestamp"])).tz_localize("UTC")
    missing = pairs[~pairs.isin(asset)]
    print(f"all valid pairs in asset data: {missing.empty} ({len(pairs) - len(missing)}/{len(pairs)})")
    if not missing.empty:
        print(f"   first missing: {list(missing[:5])}")
    print()

    for split in SPLITS:
        labels = pd.read_csv(LABELS / f"{split}.csv")
        stamps = pd.DatetimeIndex(pd.to_datetime(labels["timestamp"])).tz_localize("UTC")
        hit = stamps.isin(pairs)
        print(f"== {split}: {len(stamps)} timestamps, {stamps.min()} -> {stamps.max()}")
        print(f"   overlap: {hit.sum()} ")
        positive = labels["label_max"].to_numpy() == 1
        print(f"   pos_flare_labels: {positive.sum()}, in overlap: {(positive & hit).sum()}")

        # Per-year breakdown, plus positives kept, since class balance matters for training.
        per_year = (labels.assign(year=stamps.year, hit=hit)
                    .groupby("year").agg(labels=("hit", "size"), overlap=("hit", "sum"),
                                         pos_flare_labels=("label_max", "sum"),
                                         pos_flare_labels_overlap=("label_max", lambda s: s[hit[s.index]].sum())))
        print(per_year.to_string(), "\n")


if __name__ == "__main__":
    main()
