"""Plot how the 4096 flare dataset shrinks to the 224 data we train on.

Three data sets:
  4096 labels      : assets/surya-bench-flare-forecasting/*.csv, the official samples (full-resolution images)
  224 zarr frames  : every written frame in surya-bench-224-interarea.zarr (time labels are true UTC)
  224 valid pairs  : experiments/valid_pairs_224.csv, 224 frames that have a frame 1 h earlier;
                     assets_224/*.csv are the labels kept for them

Top: samples at each stage. Middle: samples and flare rate per split, 4096 vs 224.
Bottom: samples per month, 2010-2024, showing where the 224 data has gaps.

Usage (from downstream_examples/solar_flare_forcasting):
    python experiments/plot_data_4096_vs_224.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

ZARR = Path("../../surya-bench-224-interarea.zarr")        # <year>/dataset/time
LABELS_4096 = Path("assets/surya-bench-flare-forecasting")
LABELS_224 = Path("assets_224")
PAIRS = "experiments/valid_pairs_224.csv"
SPLITS = ["train", "validation", "leaky_validation", "test"]
LABEL = "label_max"
OUT = "experiments/data_4096_vs_224.png"

C4096, CFRAMES, C224 = "#2a78d6", "#eb6834", "#1baf7a"   # blue, orange, aqua
INK, INK2 = "#0b0b0b", "#52514e"


def zarr_frames():
    """True UTC hour of every written 224 frame (same conversion as checks/count_data_funnel.py)."""
    times = []
    for year_dir in sorted(ZARR.glob("*/dataset")):
        t = np.asarray(zarr.open(str(year_dir), mode="r")["time"][:])
        times.append(t[t != 0])                                       # unwritten rows read back as 0
    true = pd.to_datetime(np.concatenate(times), unit="ns")         # labels are already true UTC
    return pd.Series(true).drop_duplicates()


# 1. Data
l4096 = {s: pd.read_csv(LABELS_4096 / f"{s}.csv", parse_dates=["timestamp"]) for s in SPLITS}
l224 = {s: pd.read_csv(LABELS_224 / f"{s}.csv", parse_dates=["timestamp"]) for s in SPLITS}
all4096, all224 = pd.concat(l4096.values()), pd.concat(l224.values())
frames = zarr_frames()
pairs = pd.read_csv(PAIRS, parse_dates=["timestamp"])["timestamp"]

stages = [("4096 labelled samples", len(all4096), C4096),
          ("224 zarr frames (written, unique UTC hours)", len(frames), CFRAMES),
          ("224 valid (t − 1h, t) pairs", len(pairs), C224),
          ("224 labelled samples (assets_224)", len(all224), C224)]
for name, n, _ in stages:
    print(f"{name:<45} {n:>7,}  ({n / len(all4096):.0%} of 4096)")
for s in SPLITS:
    print(f"{s:<17} 4096 {len(l4096[s]):>6,} ({l4096[s][LABEL].mean():.1%} flares)   "
          f"224 {len(l224[s]):>6,} ({l224[s][LABEL].mean():.1%} flares)   kept {len(l224[s]) / len(l4096[s]):.0%}")

fig = plt.figure(figsize=(13, 11))
grid = fig.add_gridspec(3, 2, height_ratios=[1, 1.1, 1.2], hspace=0.55, wspace=0.25)

# 2. Top: samples at each stage, as a share of the 4096 set
ax = fig.add_subplot(grid[0, :])
y = np.arange(len(stages))[::-1]
ax.barh(y, [n for _, n, _ in stages], height=0.6, color=[c for *_, c in stages])
for yi, (name, n, _) in zip(y, stages):
    ax.annotate(f"{n:,}  ({n / len(all4096):.0%})", (n, yi), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK)
ax.set_yticks(y, [name for name, *_ in stages])
ax.set_xlim(0, len(all4096) * 1.18)
ax.set_xlabel("samples (hours)")
ax.set_title("From 4096 to 224: samples at each stage (% of the 4096 set)", loc="left")

# 3. Middle: per split, samples and flare rate, 4096 vs 224
x, w = np.arange(len(SPLITS)), 0.38
for col, (title, value, fmt) in enumerate([
        ("Samples per split", lambda d: len(d), "{:,.0f}"),
        ("Flare rate per split", lambda d: 100 * d[LABEL].mean(), "{:.1f}%")]):
    ax = fig.add_subplot(grid[1, col])
    for offset, labels, color, name in ((-w / 2, l4096, C4096, "4096"), (w / 2, l224, C224, "224 (assets_224)")):
        vals = [value(labels[s]) for s in SPLITS]
        bars = ax.bar(x + offset, vals, w, color=color, label=name)
        for bar, v in zip(bars, vals):
            ax.annotate(fmt.format(v), (bar.get_x() + w / 2, v), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, [s.replace("leaky_validation", "leaky_valid\n(unused)") for s in SPLITS])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, fontsize=9)
fig.axes[1].set_ylabel("samples (hours)")
fig.axes[2].set_ylabel("% flare samples")

# # 4. Bottom: samples per month for the three data sets
# ax = fig.add_subplot(grid[2, :])
# for times, color, name in ((all4096["timestamp"], C4096, "4096 labelled samples"),
#                            (frames, CFRAMES, "224 zarr frames"),
#                            (all224["timestamp"], C224, "224 labelled samples (assets_224)")):
#     monthly = pd.Series(1, index=pd.DatetimeIndex(times)).resample("MS").sum()
#     ax.plot(monthly.index, monthly.values, color=color, lw=2, label=name)
# ax.set_ylabel("samples per month (hours)")
# ax.set_title("Samples per month: where the 224 data has gaps", loc="left")
# ax.legend(frameon=False, fontsize=9, loc="lower left")

for a in fig.axes:
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="x" if a is fig.axes[0] else "y", color="#e4e3df", lw=0.8)
    a.set_axisbelow(True)
    a.tick_params(colors=INK2, labelsize=9)

fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"saved {OUT}")
