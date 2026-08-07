"""Who is over- or under-performing what their pitch shape supports?

    python run_projection.py                    # latest complete season
    python run_projection.py --season 2026      # in-progress season
    python run_projection.py --target run_value_pitcher

Validates the signal before ranking anyone: if the shape-results gap does not
actually reverse in the following season, the script says so and declines to
present the leaderboard as predictive.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd

import features as feat
from config import COMPLETE_SEASONS, OUTPUT_DIR
from evaluation import adjust_vaa_for_height, pitcher_season_table
from fetch import load_seasons
from projection import fit_shape_expectation, measure_gap_reversal, rank_candidates

log = logging.getLogger("projection")


def _fmt(df: pd.DataFrame, target: str) -> str:
    d = df.copy()
    d["player_name"] = d["player_name"].str.slice(0, 22)
    rename = {
        "player_name": "pitcher",
        "n_pitches": "n",
        "release_speed": "velo",
        "ivb_in": "IVB",
        "vaa_adj": "VAAadj",
        target: "actual",
        f"{target}_expected": "shape_exp",
        f"{target}_gap": "gap",
        "projected_change": "proj_chg",
        "projected_next": "proj_next",
    }
    d = d[[c for c in rename if c in d.columns]].rename(columns=rename)
    for c in ("actual", "shape_exp", "gap", "proj_chg", "proj_next"):
        if c in d.columns:
            d[c] = d[c].map(lambda v: f"{v:+.4f}" if pd.notna(v) else "")
    for c in ("velo", "IVB", "VAAadj"):
        if c in d.columns:
            d[c] = d[c].map(lambda v: f"{v:.1f}" if pd.notna(v) else "")
    return d.to_string(index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Regression / progression candidates from pitch shape.")
    ap.add_argument("--season", type=int, default=max(COMPLETE_SEASONS))
    ap.add_argument("--target", default="whiff_rate", choices=["whiff_rate", "run_value_pitcher"])
    ap.add_argument("--pitch-type", default="FF")
    ap.add_argument("--min-pitches", type=int, default=150)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

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

    table = fit_shape_expectation(table, target=args.target)

    log.info("=== validating the signal ===")
    # Validate on complete seasons only. Pairing a full season against the
    # in-progress one would compare a 600-pitch sample to a 250-pitch sample and
    # attribute the resulting shrinkage to reversal.
    complete = table[table["game_year"].isin(COMPLETE_SEASONS)]
    stats = measure_gap_reversal(complete, target=args.target)
    if not stats.get("usable"):
        log.error("only %d consecutive-season pairs — cannot validate", stats["n_pairs"])
        return

    print()
    print("=" * 78)
    print(f"DOES THE SHAPE-RESULTS GAP ACTUALLY REVERSE?   target = {args.target}")
    print("=" * 78)
    print(f"  consecutive-season pairs analysed : {stats['n_pairs']:,}")
    print(f"  weight on past results            : {stats['beta_actual_per_sd']:+.4f} per SD")
    print(f"  weight on shape expectation       : {stats['beta_expected_per_sd']:+.4f} per SD")
    print(f"  share of forecast carried by shape: {stats['shape_weight']:.1%}")
    print(f"  fraction of the gap that reverses : {stats['reversal_fraction']:.1%} "
          f"(t = {stats['reversal_t']:+.1f})")

    if abs(stats["reversal_t"]) < 2:
        print()
        print("  -> The gap does NOT reliably reverse. These rankings are descriptive")
        print("     only and should not be presented as predictions.")
        predictive = False
    else:
        print()
        print(f"  -> The gap reverses reliably. About {stats['reversal_fraction']:.0%} of it is")
        print("     transient; the rest is persistent (command, sequencing, defence).")
        predictive = True
    print("=" * 78)

    reversal = max(0.0, min(1.0, stats["reversal_fraction"]))
    regression, progression = rank_candidates(
        table,
        season=args.season,
        target=args.target,
        reversal_fraction=reversal,
        min_pitches=args.min_pitches,
        top_n=args.top,
    )

    label = "predicted" if predictive else "descriptive only"
    print()
    print(f"REGRESSION CANDIDATES — {args.season} {args.pitch_type}, results above what shape supports ({label})")
    print("-" * 78)
    print(_fmt(regression, args.target))
    print()
    print(f"PROGRESSION CANDIDATES — {args.season} {args.pitch_type}, shape better than results ({label})")
    print("-" * 78)
    print(_fmt(progression, args.target))
    print()

    out = OUTPUT_DIR / "projection"
    out.mkdir(parents=True, exist_ok=True)
    regression.to_csv(out / f"regression_{args.season}_{args.pitch_type}_{args.target}.csv", index=False)
    progression.to_csv(out / f"progression_{args.season}_{args.pitch_type}_{args.target}.csv", index=False)
    pd.DataFrame([stats]).to_csv(out / f"validation_{args.target}.csv", index=False)
    log.info("wrote rankings and validation to %s", out)


if __name__ == "__main__":
    main()
