"""Select the best teacher trial of the 224 sweep by a rank sum of four measures.

Per trial, from history.csv, with smoothed CSS = 5-check moving average:
  smoothed_css     peak smoothed CSS                                  higher is better
  spike            best raw CSS - smoothed CSS at that check          lower is better
  plateau_checks   checks around the peak within plateau_band of it   higher is better
  fpr              FP / (FP + TN), mean over the 5 peak checks        lower is better
Rank each measure (1 = best), sum the ranks; lowest total wins, ties go to higher smoothed_css.

Run from downstream_examples/solar_flare_forcasting:
  python experiments/teacher_224/rank_trials.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SMOOTH_WINDOW = 5      # checks in the smoothing window: the check and 2 on each side
PLATEAU_BAND = 0.02 #try with 0.1 and 0.03 as well   # how far below the peak smoothed CSS still counts as the plateau
N_FLARE = 360          # flare rows (label 1) in assets_224/validation.csv
N_NONFLARE = 3006      # non-flare rows (label 0) in assets_224/validation.csv
STEPS_PER_EPOCH = 539


def false_positive_rate(n_forecast, tss):
    """False-positive rate FP / (FP + TN) at one check.

    history.csv has no FP or TN column, so they are recovered from the number of flare
    forecasts and TSS. Short names: n = n_forecast, F = N_FLARE, N = N_NONFLARE.

    Every flare row is a TP or an FN, and every non-flare row is an FP or a TN:
        TP + FN = F
        FP + TN = N

    Every flare forecast is a TP or an FP:
        TP + FP = n   ->   TP = n - FP

    TSS by definition:
        TSS = TP / (TP + FN) - FP / (FP + TN)
            = TP / F - FP / N

    Put TP = n - FP into the TSS line and solve for FP:
        TSS = (n - FP) / F - FP / N
            = n / F - FP / F - FP / N
            = n / F - FP * (1/F + 1/N)
        FP * (1/F + 1/N) = n / F - TSS
        FP = (n / F - TSS) / (1/F + 1/N) 
        
    Then TN = N - FP, and FPR = FP / (FP + TN) = FP / N.
    """
    fp = (n_forecast / N_FLARE - tss) / (1 / N_FLARE + 1 / N_NONFLARE)
    tn = N_NONFLARE - fp
    return fp / (fp + tn)


def measure_trial(history, plateau_band=PLATEAU_BAND):
    """smoothed_css, spike, plateau_checks and fpr of one trial."""
    css = history["css"].to_numpy()
    smoothed_css = history["css"].rolling(SMOOTH_WINDOW, center=True).mean().to_numpy()  # NaN at the first and last 2 checks
    half_window = SMOOTH_WINDOW // 2

    # 2. the peak of the smoothed curve
    peak_idx = int(np.nanargmax(smoothed_css))
    peak_smoothed_css = smoothed_css[peak_idx]

    # 3. spike: how far the best raw check sits above the smoothed curve
    best_raw_idx = int(np.argmax(css))
    spike = css[best_raw_idx] - smoothed_css[best_raw_idx]  # NaN if the best raw check is one of the first or last 2

    # 4. plateau: walk left and right from the peak while the smoothed curve stays within the band
    plateau_floor = peak_smoothed_css - plateau_band - 1e-9  # 1e-9: tolerance for float rounding
    plateau_start = peak_idx
    while plateau_start - 1 >= 0 and smoothed_css[plateau_start - 1] >= plateau_floor:  # NaN >= floor is False: the walk stops at the edges
        plateau_start -= 1
    plateau_end = peak_idx
    while plateau_end + 1 < len(smoothed_css) and smoothed_css[plateau_end + 1] >= plateau_floor:
        plateau_end += 1
    plateau_checks = plateau_end - plateau_start + 1

    # 5. false-positive rate, averaged over the same 5 checks that make up the peak smoothed CSS
    peak_window = history.iloc[peak_idx - half_window: peak_idx + half_window + 1]
    mean_fpr = false_positive_rate(peak_window["forecast"], peak_window["tss"]).mean()

    steps = history["step"]
    return {"smoothed_css": peak_smoothed_css, "spike": spike, "plateau_checks": plateau_checks, "fpr": mean_fpr,
            "peak_step": int(steps[peak_idx]), "peak_epoch": steps[peak_idx] / STEPS_PER_EPOCH,
            "best_raw_step": int(steps[best_raw_idx]), "best_raw_css": css[best_raw_idx]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep_dir", default="experiments/results_224/sweep_correct_css")
    parser.add_argument("--out", default="experiments/results_224/sweep_correct_css_selection.csv")
    parser.add_argument("--plateau_band", type=float, default=PLATEAU_BAND,
                        help="how far below the peak smoothed CSS still counts as the plateau (default 0.02)")
    args = parser.parse_args()

    trial_rows = []
    for trial_dir in sorted(Path(args.sweep_dir).iterdir()):
        if not trial_dir.is_dir():
            continue
        if not (trial_dir / "history.csv").exists():
            print(f"skipped, no history.csv (unfinished): {trial_dir.name}")
            continue
        trial_result = measure_trial(pd.read_csv(trial_dir / "history.csv"), args.plateau_band)
        if np.isnan(trial_result["spike"]):
            print(f"warning: {trial_dir.name}: best raw check is at the edge, so its spike is unknown (ranked last)")
        trial_rows.append({"trial": trial_dir.name, **trial_result})
    table = pd.DataFrame(trial_rows)

    # 6. rank each measure: 1 = best, equal values share the average place
    table["rank_smoothed_css"] = table["smoothed_css"].rank(ascending=False)
    table["rank_spike"] = table["spike"].rank(ascending=True, na_option="bottom")
    table["rank_plateau"] = table["plateau_checks"].rank(ascending=False)
    table["rank_fpr"] = table["fpr"].rank(ascending=True)

    # 7. add the ranks; the lowest total wins, equal totals go to the higher smoothed_css
    rank_columns = ["rank_smoothed_css", "rank_spike", "rank_plateau", "rank_fpr"]
    table["rank_total"] = table[rank_columns].sum(axis=1)
    table = table.sort_values(["rank_total", "smoothed_css"], ascending=[True, False]).reset_index(drop=True)

    table.to_csv(args.out, index=False)
    shown = table[["trial", "smoothed_css", "spike", "plateau_checks", "fpr", *rank_columns, "rank_total", "peak_epoch"]]
    shown = shown.assign(trial=shown["trial"].str[-11:])  # the start-time id, e.g. 1116-144409
    print(f"\n{len(table)} trials, best first (4 checks = 1 epoch)\n")
    print(shown.round({"smoothed_css": 4, "spike": 4, "fpr": 4, "peak_epoch": 1}).to_string(index=False))

    winner = table.iloc[0]
    print(f"\nSelected: {winner['trial']}")
    print(f"  smoothed CSS = {winner['smoothed_css']:.4f} at step {winner['peak_step']} (epoch {winner['peak_epoch']:.1f}); "
          f"highest of all trials = {table['smoothed_css'].max():.4f}")
    print(f"  spike = {winner['spike']:.4f}, plateau = {winner['plateau_checks']} checks, FPR = {winner['fpr']:.4f}")
    print(f"  best.pth holds step {winner['best_raw_step']} (CSS {winner['best_raw_css']:.4f}): "
          f"{Path(args.sweep_dir) / winner['trial'] / 'best.pth'}")
    print(f"\nFull table: {args.out}")


if __name__ == "__main__":
    main()
