"""Plot flare vs non-flare counts for the train / validation / test splits.

Top: total counts per split. Below: monthly counts over time, one row per split.

Usage (from downstream_examples/solar_flare_forcasting):
    python experiments/plot_data_splits.py
"""

import matplotlib.pyplot as plt
import pandas as pd

SPLITS = {
    "train": "assets_224/train.csv",
    "validation": "assets_224/validation.csv",
    "test": "assets_224/test.csv",
}
LABEL = "label_max"
NONFLARE_COLOR = "#2a78d6"
FLARE_COLOR = "#eb6834"
OUT = "experiments/data_splits.png"

data = {}
for name, path in SPLITS.items():
    df = pd.read_csv(path, parse_dates=["timestamp"])
    data[name] = df
    n_flare = int(df[LABEL].sum())
    n_total = len(df)
    print(f"{name:<11} total={n_total:>6}  flare={n_flare:>5} ({100 * n_flare / n_total:.1f}%)  "
          f"non-flare={n_total - n_flare:>6}  "
          f"range={df['timestamp'].min():%Y-%m-%d} -> {df['timestamp'].max():%Y-%m-%d}")

fig = plt.figure(figsize=(12, 10))
grid = fig.add_gridspec(4, 1, height_ratios=[1.4, 1, 1, 1], hspace=0.5)

# --- Total counts per split ---
ax = fig.add_subplot(grid[0])
names = list(data)
x = range(len(names))
width = 0.38
nonflare = [int((data[n][LABEL] == 0).sum()) for n in names]
flare = [int((data[n][LABEL] == 1).sum()) for n in names]
bars_nf = ax.bar([i - width / 2 for i in x], nonflare, width, color=NONFLARE_COLOR, label="non-flare")
bars_f = ax.bar([i + width / 2 for i in x], flare, width, color=FLARE_COLOR, label="flare")
for bars, counts in ((bars_nf, nonflare), (bars_f, flare)):
    for bar, count, n in zip(bars, counts, names):
        pct = 100 * count / len(data[n])
        ax.annotate(f"{count:,}\n({pct:.1f}%)", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points")
ax.set_xticks(list(x))
ax.set_xticklabels(names)
ax.set_ylabel("samples (hours)")
ax.set_ylim(0, max(nonflare) * 1.25)
ax.set_title("Flare vs non-flare samples per split")
ax.legend(frameon=False)

# --- Monthly counts over time, one row per split (shared x axis) ---
first = None
for row, name in enumerate(names, start=1):
    ax = fig.add_subplot(grid[row], sharex=first)
    first = first or ax
    monthly = (data[name].set_index("timestamp")[LABEL]
               .resample("MS").agg(["sum", "count"]))
    monthly["nonflare"] = monthly["count"] - monthly["sum"]
    ax.bar(monthly.index, monthly["nonflare"], width=25, align="edge", color=NONFLARE_COLOR, label="non-flare")
    ax.bar(monthly.index, monthly["sum"], width=25, align="edge", bottom=monthly["nonflare"],
           color=FLARE_COLOR, label="flare")
    ax.set_title(f"{name}: samples per month", fontsize=10, loc="left")
    ax.set_ylabel("hours")

for ax in fig.axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"saved {OUT}")
