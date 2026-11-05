"""Count the flaring periods in the 224 training labels.

A flaring period is a run of consecutive rows (sorted by time) with label_max = 1.

Run from downstream_examples/solar_flare_forcasting:
    conda run -n suryaenv python experiments/checks/count_flare_periods.py
"""

import pandas as pd

LABELS = "assets_224/train.csv"

t = pd.read_csv(LABELS, parse_dates=["timestamp"]).sort_values("timestamp")
flare = t["label_max"] == 1

# A period starts at a flare row whose previous row is not a flare
starts = flare & ~flare.shift(fill_value=False)
period = starts.cumsum()[flare]  # period number of each flare row
rows_per_period = period.value_counts()

print(f"{len(t)} training rows, {flare.sum()} flare rows ({flare.mean():.1%})")
print(f"{starts.sum()} flaring periods, {rows_per_period.mean():.1f} flare rows each on average")
print(f"flare rows per period: median {rows_per_period.median():.0f}, max {rows_per_period.max()}")
print("\nflaring periods per year (by start):")
print(t.loc[starts, "timestamp"].dt.year.value_counts().sort_index().to_string())
