"""When does shape beat results? A sample-size experiment.

The headline forecasting test said past results beat shape at predicting next
season. That is a real result and it constrains the thesis. But it was run on
pitcher-seasons with at least 250 four-seamers, which is a *generous* sample --
roughly a full season of work for a starter.

The interesting question for evaluation is what happens when you have less than
that: a prospect, a trade-deadline target, a reliever, a pitcher who just added
a pitch, the first month of a season. Outcome statistics degrade fast as the
sample shrinks because they are mostly noise. Shape does not degrade at all,
because it is measured almost exactly from the first pitch.

This script holds the *target* fixed (next season, 250+ pitches, so the thing
being predicted is equally well measured throughout) and varies only how much
data we are allowed to see about the pitcher now.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features as feat
from config import DEFAULT_SEASONS, OUTPUT_DIR, RANDOM_SEED
from evaluation import adjust_vaa_for_height, pitcher_season_table
from fetch import load_seasons

log = logging.getLogger("sample_size")

SHAPE = ["ivb_in", "hb_in", "vaa_adj", "haa_armside", "release_pos_z", "release_extension", "release_speed"]
RESULTS = ["run_value_pitcher", "whiff_rate"]
THRESHOLDS = [30, 60, 125, 250, 500, 1000]


def score(frame: pd.DataFrame, cols: list[str], target_col: str, n_splits: int = 5) -> tuple[float, float]:
    sub = frame[cols + [target_col, "pitcher"]].dropna()
    if len(sub) < 150:
        return float("nan"), float("nan")
    x = sub[cols].to_numpy(dtype=float)
    y = sub[target_col].to_numpy(dtype=float)
    groups = sub["pitcher"].to_numpy()

    scores = []
    for tr, te in GroupKFold(n_splits=n_splits).split(x, y, groups):
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=10.0, random_state=RANDOM_SEED))
        pipe.fit(x[tr], y[tr])
        pred = pipe.predict(x[te])
        base = y[tr].mean()
        ss_res = float(((y[te] - pred) ** 2).sum())
        ss_tot = float(((y[te] - base) ** 2).sum())
        scores.append(1.0 - ss_res / ss_tot)
    return float(np.mean(scores)), float(len(sub))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    raw = load_seasons(tuple(DEFAULT_SEASONS))
    data = adjust_vaa_for_height(feat.build(raw))

    # Build the table with no minimum, then apply thresholds ourselves.
    table = pitcher_season_table(data, pitch_type="FF", min_pitches=1)

    # Target: next season, always measured on a solid sample so the thing being
    # predicted has constant quality across every row of the experiment.
    stable = table[table["n_pitches"] >= 250][["pitcher", "game_year", "run_value_pitcher", "whiff_rate"]]
    stable = stable.rename(
        columns={"run_value_pitcher": "rv_next", "whiff_rate": "whiff_next"}
    )
    stable["game_year"] = stable["game_year"] - 1

    # Shuffle once, then take the first N pitches of each pitcher-season. A
    # *filter* of "at least N" would not test anything: most pitcher-seasons
    # passing a 30-pitch filter still contain a full season of pitches, so the
    # metrics would be computed on far more data than the threshold implies.
    # We need each row built from exactly N pitches.
    ff = data[data["pitch_type"].eq("FF")].sample(frac=1.0, random_state=RANDOM_SEED)
    ff["pitch_idx"] = ff.groupby(["pitcher", "game_year"], observed=True).cumcount()
    ff["season_n"] = ff.groupby(["pitcher", "game_year"], observed=True)["pitch_idx"].transform("size")

    rows = []
    for thresh in THRESHOLDS:
        window = ff[(ff["season_n"] >= thresh) & (ff["pitch_idx"] < thresh)]
        current = pitcher_season_table(window, pitch_type="FF", min_pitches=1)
        log.info("threshold %d: %d pitcher-seasons, each built from exactly %d pitches",
                 thresh, len(current), thresh)
        merged = current.merge(stable, on=["pitcher", "game_year"], how="inner")
        if len(merged) < 200:
            log.warning("threshold %d: only %d pairs, skipping", thresh, len(merged))
            continue

        for target_name, target_col in (("run value", "rv_next"), ("whiff rate", "whiff_next")):
            shape_r2, n = score(merged, SHAPE, target_col)
            results_r2, _ = score(merged, RESULTS, target_col)
            both_r2, _ = score(merged, SHAPE + RESULTS, target_col)
            rows.append(
                {
                    "min_pitches": thresh,
                    "target": target_name,
                    "n_pairs": int(n) if n == n else 0,
                    "shape_r2": shape_r2,
                    "past_results_r2": results_r2,
                    "both_r2": both_r2,
                    "shape_minus_results": shape_r2 - results_r2,
                }
            )
            log.info(
                "%4d pitches | %-10s | shape %+.4f  results %+.4f  both %+.4f",
                thresh,
                target_name,
                shape_r2,
                results_r2,
                both_r2,
            )

    out = pd.DataFrame(rows)
    path = OUTPUT_DIR / "paper" / "sample_size_crossover.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
