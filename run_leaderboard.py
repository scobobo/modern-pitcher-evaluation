"""Full shape leaderboard, validated by walk-forward backtest.

    python run_leaderboard.py                 # current season, default whiff rate
    python run_leaderboard.py --season 2025
    python run_leaderboard.py --pitch-type SL

Runs the backtest first. If shape does not beat plain mean reversion
out-of-sample, the leaderboard is labelled unvalidated rather than presented as
a forecast.
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
from config import COMPLETE_SEASONS, CURRENT_PARTIAL_SEASON, OUTPUT_DIR
from evaluation import adjust_vaa_for_height, pitcher_season_table
from fetch import load_seasons
from leaderboard import build_leaderboard, decile_table, walk_forward_backtest

log = logging.getLogger("leaderboard")
OUT = OUTPUT_DIR / "leaderboard"


def main() -> None:
    ap = argparse.ArgumentParser(description="Validated pitch-shape leaderboard.")
    ap.add_argument("--season", type=int, default=CURRENT_PARTIAL_SEASON)
    ap.add_argument("--target", default="whiff_rate", choices=["whiff_rate", "run_value_pitcher"])
    ap.add_argument("--pitch-type", default="FF")
    ap.add_argument("--min-pitches", type=int, default=250)
    ap.add_argument("--min-swings", type=int, default=100)
    ap.add_argument("--show", type=int, default=15)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    seasons = tuple(sorted(set(COMPLETE_SEASONS) | {args.season}))
    data = adjust_vaa_for_height(feat.build(load_seasons(seasons)))

    table = pitcher_season_table(data, pitch_type=args.pitch_type, min_pitches=1)
    names = (
        data[data["pitch_type"].eq(args.pitch_type)]
        .groupby(["pitcher", "game_year"], observed=True)["player_name"]
        .first()
        .reset_index()
    )
    table = table.merge(names, on=["pitcher", "game_year"], how="left")

    # ---------- validation ----------
    log.info("=== walk-forward backtest (each season scored using only earlier seasons) ===")
    test_seasons = tuple(s for s in COMPLETE_SEASONS if s + 1 in COMPLETE_SEASONS)
    per_season, pooled = walk_forward_backtest(
        table,
        target=args.target,
        seasons=test_seasons,
        min_pitches=args.min_pitches,
        min_swings=args.min_swings,
    )

    if pooled.empty:
        log.error("backtest produced no usable seasons — aborting")
        return

    edge = pooled["shape_edge"].to_numpy(float)
    resid = pooled["naive_residual"].to_numpy(float)
    r = float(np.corrcoef(edge, resid)[0, 1])
    n = len(pooled)
    t_stat = r * np.sqrt((n - 2) / max(1e-12, 1 - r**2))
    mean_lift = float(per_season["shape_lift"].mean())
    seasons_positive = int((per_season["shape_lift"] > 0).sum())

    print()
    print("=" * 82)
    print(f"WALK-FORWARD VALIDATION   target = {args.target}   pitch = {args.pitch_type}")
    print("=" * 82)
    print(per_season.round(4).to_string(index=False))
    print()
    print(f"  pooled out-of-sample pitcher-seasons : {n:,}")
    print(f"  mean shape lift over mean reversion  : {mean_lift:+.4f} R²")
    print(f"  seasons where shape helped           : {seasons_positive} of {len(per_season)}")
    print(f"  corr(predicted edge, what mean       : {r:+.3f}  (t = {t_stat:+.1f})")
    print( "       reversion missed)")

    validated = t_stat > 2 and mean_lift > 0
    print()
    if validated:
        print("  -> VALIDATED. Shape explains part of what mean reversion misses,")
        print("     out-of-sample, without using the future to predict the past.")
    else:
        print("  -> NOT VALIDATED. Shape adds nothing beyond mean reversion here.")
        print("     The leaderboard below is descriptive only.")
    print("=" * 82)

    deciles = decile_table(pooled, target=args.target)
    print()
    print("EDGE DECILES — does a higher predicted edge actually beat mean reversion?")
    print("-" * 82)
    show = deciles.copy()
    show.columns = ["decile", "n", "mean_edge", "realised_vs_meanrev", "actual_next", "this_season", "beat_%"]
    print(show.round(4).to_string(index=False))
    spread = float(deciles["mean_residual"].iloc[-1] - deciles["mean_residual"].iloc[0])
    print()
    print(f"  top decile minus bottom decile, realised: {spread:+.4f} {args.target}")

    per_season.to_csv(OUT / f"backtest_{args.pitch_type}_{args.target}.csv", index=False)
    deciles.to_csv(OUT / f"deciles_{args.pitch_type}_{args.target}.csv", index=False)

    # ---------- leaderboard ----------
    train_seasons = tuple(s for s in COMPLETE_SEASONS if s < args.season)
    board = build_leaderboard(
        table,
        season=args.season,
        target=args.target,
        train_seasons=train_seasons,
        min_pitches=args.min_pitches,
        min_swings=args.min_swings,
    )
    if board.empty:
        log.error("no qualifying pitchers for %s", args.season)
        return

    path = OUT / f"leaderboard_{args.season}_{args.pitch_type}_{args.target}.csv"
    board.to_csv(path, index=False)

    disp = board.copy()
    disp["player_name"] = disp["player_name"].str.slice(0, 22)
    keep = ["player_name", "n_pitches", "n_swings", "release_speed", "ivb_in", "vaa_adj",
            args.target, "shape_exp", "proj_full", "shape_edge", "verdict"]
    disp = disp[[c for c in keep if c in disp.columns]]
    disp.columns = ["pitcher", "n", "sw", "velo", "IVB", "VAAadj", "actual", "shape_exp",
                    "proj", "edge", "verdict"][: len(disp.columns)]
    for c in ("actual", "shape_exp", "proj", "edge"):
        disp[c] = disp[c].map(lambda v: f"{v:+.4f}")
    for c in ("velo", "IVB", "VAAadj"):
        disp[c] = disp[c].map(lambda v: f"{v:.1f}")

    status = "validated" if validated else "UNVALIDATED — descriptive only"
    print()
    print("=" * 82)
    print(f"LEADERBOARD — {args.season} {args.pitch_type}, {len(board)} qualified pitchers ({status})")
    print(f"trained on {min(train_seasons)}–{max(train_seasons)}; "
          f"min {args.min_pitches} pitches / {args.min_swings} swings")
    print("=" * 82)
    print(f"\nTOP {args.show} — shape argues for MORE than mean reversion")
    print(disp.head(args.show).to_string(index=False))
    print(f"\nBOTTOM {args.show} — shape argues for LESS")
    print(disp.tail(args.show).to_string(index=False))
    print()
    log.info("full leaderboard (%d rows) -> %s", len(board), path)


if __name__ == "__main__":
    main()
