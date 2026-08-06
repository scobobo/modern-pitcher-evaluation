"""End-to-end run: fetch, build features, fit the ladder, test the trend.

    python run_analysis.py --seasons 2021 2022 2023 2024 2025

Writes every table to output/ as CSV and prints a verdict that follows the
numbers wherever they go.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd

import features as feat
from config import DEFAULT_SEASONS, OUTPUT_DIR, PRIMARY_PITCH_TYPE, RANDOM_SEED
from fetch import load_seasons
from model import (
    build_ladder,
    build_reverse_ladder,
    permutation_ranking,
    run_ladder,
    standardized_coefficients,
)
from plots import plot_binned_response, plot_incremental_skill, plot_season_effects
from temporal import season_effects, season_incremental_skill, trend_test

log = logging.getLogger("run_analysis")


def _save(df: pd.DataFrame, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    log.info("wrote %s (%d rows)", path.name, len(df))


def verdict(ladder: pd.DataFrame, effects: pd.DataFrame, target: str = "run_value_pitcher") -> str:
    """State what the numbers actually support, including 'neither'."""
    indexed = ladder.set_index("model")

    def block(name: str) -> tuple[float, float, float]:
        if name not in indexed.index:
            return float("nan"), float("nan"), float("nan")
        row = indexed.loc[name]
        return float(row["incremental_r2"]), float(row["incremental_se"]), float(row["incremental_t"])

    velo_gain, velo_se, velo_t = block("+velocity")
    spin_gain, spin_se, spin_t = block("+spin")
    shape_gain, shape_se, shape_t = block("+shape")

    velo_trend = trend_test(effects, "velo_coef_per_mph")
    spin_trend = trend_test(effects, "spin_coef_per_100rpm")

    # The outcome variable is not always runs, so never hard-code the unit --
    # a whiff model reports probability per mph, not runs per mph.
    unit = "runs" if target == "run_value_pitcher" else "whiff prob"

    # A block only counts as carrying real information if its gain clears its
    # own fold-to-fold noise. Point estimates alone are meaningless at these
    # effect sizes -- pitch-level run value is mostly irreducible.
    def is_real(gain: float, t_stat: float) -> bool:
        return gain > 0 and t_stat == t_stat and t_stat > 2.0

    lines = [
        "",
        "=" * 72,
        "VERDICT",
        "=" * 72,
        "",
        "H1 -- spin adds value beyond velocity:",
        f"  velocity block: {velo_gain:+.5f} +/- {velo_se:.5f} CV R^2  (t = {velo_t:+.2f})",
        f"  spin block:     {spin_gain:+.5f} +/- {spin_se:.5f}          (t = {spin_t:+.2f})",
        f"  shape block:    {shape_gain:+.5f} +/- {shape_se:.5f}          (t = {shape_t:+.2f})",
        "  (gains are paired across identical CV folds; t > 2 is the bar for 'real')",
    ]

    spin_real, velo_real = is_real(spin_gain, spin_t), is_real(velo_gain, velo_t)

    if not spin_real and not velo_real:
        lines.append(
            "  -> Neither block clears its own noise at the pitch level. No support for the "
            "hypothesis, and no support for the opposite either."
        )
    elif spin_real and not velo_real:
        lines.append("  -> Spin carries unique signal where velocity does not. Hypothesis supported.")
    elif velo_real and not spin_real:
        lines.append("  -> Velocity carries unique signal where spin does not. Hypothesis contradicted.")
    elif spin_gain > velo_gain:
        lines.append("  -> Both are real; spin contributes more unique signal. Hypothesis supported.")
    else:
        lines.append(
            "  -> Both are real, but velocity contributes more unique signal. "
            "Partial support at best."
        )

    if is_real(shape_gain, shape_t) and shape_gain > max(velo_gain, spin_gain):
        lines.append(
            "  -> Note: movement and approach angle beat both raw inputs. That is the more "
            "defensible framing -- spin matters through the shape it creates, not on its own."
        )

    lines += [
        "",
        "H2 -- velocity's value is declining over time:",
        f"  velocity effect: {velo_trend['slope_per_year']:+.6f} {unit}/mph per year "
        f"(t = {velo_trend['t_stat']:+.2f}, {velo_trend['n_seasons']} seasons)",
        f"  spin effect:     {spin_trend['slope_per_year']:+.6f} {unit}/100rpm per year "
        f"(t = {spin_trend['t_stat']:+.2f})",
    ]

    if velo_trend["t_stat"] < -2:
        lines.append("  -> Velocity's per-mph value is measurably falling. Hypothesis supported.")
    elif velo_trend["t_stat"] > 2:
        lines.append("  -> Velocity's per-mph value is rising. Hypothesis contradicted.")
    else:
        lines.append(
            "  -> No detectable trend either way. With ~10 seasons this is the most "
            "likely outcome, and 'no evidence of decline' is the correct thing to report."
        )

    lines += ["", "=" * 72, ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Velocity vs spin: does spin add value beyond velo?")
    parser.add_argument("--seasons", type=int, nargs="+", default=list(DEFAULT_SEASONS))
    parser.add_argument(
        "--pitch-type",
        default=PRIMARY_PITCH_TYPE,
        help="Pitch type for the primary analysis (default FF). Use ALL to pool.",
    )
    parser.add_argument("--target", default="run_value_pitcher", choices=["run_value_pitcher", "is_whiff"])
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Subsample whole pitchers down to roughly this many pitches for the ladder, "
        "to keep runtime sane. Sampling is by pitcher so folds stay clean.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    raw = load_seasons(tuple(args.seasons))
    data = feat.build(raw)

    if args.pitch_type != "ALL":
        data = data[data["pitch_type"].eq(args.pitch_type)]
        log.info("primary analysis on %s: %s pitches", args.pitch_type, f"{len(data):,}")

    if args.target == "is_whiff":
        data = data[data["is_swing"]]
        data["is_whiff"] = data["is_whiff"].astype(float)
        log.info("whiff model restricted to swings: %s", f"{len(data):,}")

    ladder_data = data
    if args.max_rows and len(data) > args.max_rows:
        # Sample whole pitchers, never individual pitches -- dropping random
        # pitches would break the grouped-CV guarantee we rely on.
        rng = np.random.default_rng(RANDOM_SEED)
        pitchers = np.asarray(data["pitcher"].dropna().unique(), dtype="int64")
        rng.shuffle(pitchers)
        counts = data["pitcher"].value_counts()
        keep, running = [], 0
        for p in pitchers:
            keep.append(p)
            running += int(counts[p])
            if running >= args.max_rows:
                break
        ladder_data = data[data["pitcher"].isin(keep)]
        log.info(
            "ladder subsampled to %s pitches from %d pitchers", f"{len(ladder_data):,}", len(keep)
        )

    forward = run_ladder(ladder_data, build_ladder(), target=args.target)
    _save(forward, "ladder_velocity_first")

    reverse = run_ladder(ladder_data, build_reverse_ladder(), target=args.target)
    _save(reverse, "ladder_spin_first")

    full_features = build_ladder()[-1].features
    _save(permutation_ranking(ladder_data, full_features, target=args.target), "permutation_importance")
    _save(standardized_coefficients(data, full_features, target=args.target), "standardized_coefficients")

    effects = season_effects(data, target=args.target)
    _save(effects, "season_effects")
    _save(season_incremental_skill(data, target=args.target), "season_incremental_skill")

    if not args.no_plots:
        plot_incremental_skill(forward, reverse)
        plot_binned_response(data, target=args.target)
        if len(effects) >= 3:
            plot_season_effects(effects)
        else:
            log.info("skipping season-trend figure: needs at least 3 seasons")

    print(verdict(forward, effects, target=args.target))


if __name__ == "__main__":
    main()
