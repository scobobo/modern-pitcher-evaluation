"""Analyses supporting the pitch-shape paper.

Four things the pitch-level study did not establish, and which a public paper
would be attacked for omitting:

  1. Is the shape result an artifact of location? (VAA is confounded with pitch
     height -- the single most serious objection to a shape thesis.)
  2. Does it generalise beyond four-seamers?
  3. Are shape metrics actually more reliable than outcome metrics?
  4. Does shape forecast next season better than past results do?
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd

import features as feat
from config import DEFAULT_SEASONS, OUTPUT_DIR
from evaluation import (
    adjust_vaa_for_height,
    forecast_next_season,
    pitcher_season_table,
    split_half_reliability,
    year_over_year,
)
from fetch import load_seasons
from model import CONTROL_FEATURES, ModelSpec, run_ladder
from features import SPIN_RESID_FEATURES, VELOCITY_FEATURES

log = logging.getLogger("paper")

PAPER_DIR = OUTPUT_DIR / "paper"

# Shape block using the height-adjusted approach angle instead of the raw one.
SHAPE_ADJ = ["ivb_in", "hb_in", "vaa_adj", "haa_armside", "release_pos_z", "release_pos_x_armside"]
SHAPE_RAW = ["ivb_in", "hb_in", "vaa_deg", "haa_armside", "release_pos_z", "release_pos_x_armside"]


def _save(df: pd.DataFrame, name: str) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    path = PAPER_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    log.info("wrote %s (%d rows)", path.name, len(df))


def ladder_for(data: pd.DataFrame, shape_block: list[str], *, target: str, n_splits: int = 5):
    controls = list(CONTROL_FEATURES)
    ladder = [
        ModelSpec("controls", controls, "count, location, platoon"),
        ModelSpec("+velocity", controls + VELOCITY_FEATURES, "adds velocity"),
        ModelSpec("+spin", controls + VELOCITY_FEATURES + SPIN_RESID_FEATURES, "adds residual spin"),
        ModelSpec("+shape", controls + VELOCITY_FEATURES + SPIN_RESID_FEATURES + shape_block, "adds shape"),
    ]
    return run_ladder(data, ladder, target=target, n_splits=n_splits)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    raw = load_seasons(tuple(DEFAULT_SEASONS))
    data = feat.build(raw)
    data = adjust_vaa_for_height(data)
    log.info("feature table ready: %s pitches", f"{len(data):,}")

    # --- 1. Is shape just location in disguise? ------------------------------
    ff = data[data["pitch_type"].eq("FF")]
    sample = ff.sample(n=min(600_000, len(ff)), random_state=17)

    log.info("=== robustness: raw VAA vs height-adjusted VAA (four-seamers) ===")
    raw_vaa = ladder_for(sample, SHAPE_RAW, target="run_value_pitcher")
    raw_vaa["shape_block"] = "raw VAA"
    adj_vaa = ladder_for(sample, SHAPE_ADJ, target="run_value_pitcher")
    adj_vaa["shape_block"] = "height-adjusted VAA"
    _save(pd.concat([raw_vaa, adj_vaa], ignore_index=True), "robustness_vaa")

    # --- 2. Does it hold for every pitch type? -------------------------------
    log.info("=== generality across pitch types ===")
    per_type = []
    for pt in ["FF", "SI", "FC", "SL", "CU", "CH", "ST"]:
        block = data[data["pitch_type"].eq(pt)]
        if len(block) < 60_000:
            log.warning("pitch type %s has only %s pitches, skipping", pt, f"{len(block):,}")
            continue
        block = block.sample(n=min(400_000, len(block)), random_state=17)
        for target in ("run_value_pitcher", "is_whiff"):
            frame = block[block["is_swing"]].copy() if target == "is_whiff" else block
            if target == "is_whiff":
                frame["is_whiff"] = frame["is_whiff"].astype(float)
            if len(frame) < 40_000:
                continue
            res = ladder_for(frame, SHAPE_ADJ, target=target)
            res["pitch_type"] = pt
            res["target"] = target
            res["n_pitches"] = len(frame)
            per_type.append(res)
            log.info("  %s / %s done", pt, target)
    if per_type:
        _save(pd.concat(per_type, ignore_index=True), "generality_by_pitch_type")

    # --- 3. Reliability: signal vs noise -------------------------------------
    log.info("=== split-half reliability (four-seamers) ===")
    _save(split_half_reliability(data, pitch_type="FF"), "reliability_split_half")

    table = pitcher_season_table(data, pitch_type="FF")
    _save(table, "pitcher_season_FF")
    _save(year_over_year(table), "reliability_year_over_year")

    # --- 4. Forecasting next season ------------------------------------------
    log.info("=== forecasting next season ===")
    for target in ("run_value_pitcher", "whiff_rate"):
        res = forecast_next_season(table, target=target)
        res["target"] = target
        _save(res, f"forecast_{target}")

    log.info("paper analyses complete -> %s", PAPER_DIR)


if __name__ == "__main__":
    main()
