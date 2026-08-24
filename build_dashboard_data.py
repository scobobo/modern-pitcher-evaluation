"""Export the pitcher-season dataset the dashboard filters over.

One row per pitcher, season, and pitch type, carrying the shape metrics, the
outcomes, and the model's verdict. The models are fit here, once, on complete
seasons only -- the dashboard applies them, it does not refit. Filtering to a
subset in the browser must not silently change what the model was trained on,
or every filtered view would be reporting a different, unvalidated model.

Output is a compact column-oriented JSON so the whole thing can be embedded in
a single self-contained page.
"""

from __future__ import annotations

import json
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
from leaderboard import _pairs, _shape_model
from players import attach_age, fetch_bios
from projection import SHAPE_FEATURES
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from config import RANDOM_SEED

log = logging.getLogger("dashboard_data")

PITCH_TYPES = ["FF", "SI", "FC", "SL", "ST", "CU", "CH"]
MIN_PITCHES = 150
MIN_SWINGS = 60
TARGET = "whiff_rate"

OUT = OUTPUT_DIR / "dashboard"

# Columns exported, in order. Short keys keep the embedded payload small.
COLUMNS = [
    ("name", "player_name"),
    ("yr", "game_year"),
    ("pt", "pitch_type"),
    ("age", "age"),
    ("hand", "throws"),
    ("n", "n_pitches"),
    ("sw", "n_swings"),
    ("velo", "release_speed"),
    ("ivb", "ivb_in"),
    ("hb", "hb_in"),
    ("vaa", "vaa_adj"),
    ("ext", "release_extension"),
    ("whiff", "whiff_rate"),
    ("rv", "run_value_pitcher"),
    ("exp", "shape_exp"),
    ("edge", "shape_edge"),
]
ROUND = {"age": 0, "velo": 1, "ivb": 1, "hb": 1, "vaa": 2, "ext": 1,
         "whiff": 4, "rv": 5, "exp": 4, "edge": 5}


def oof_shape_expectation(block: pd.DataFrame, target: str) -> pd.Series:
    """Shape-only expectation, out-of-fold and grouped by pitcher."""
    frame = block[SHAPE_FEATURES + [target, "pitcher"]].dropna()
    if len(frame) < 60:
        return pd.Series(np.nan, index=block.index)
    x = frame[SHAPE_FEATURES].to_numpy(float)
    y = frame[target].to_numpy(float)
    g = frame["pitcher"].to_numpy()
    oof = np.full(len(frame), np.nan)
    for tr, te in GroupKFold(n_splits=min(5, len(np.unique(g)))).split(x, y, g):
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=10.0, random_state=RANDOM_SEED))
        pipe.fit(x[tr], y[tr])
        oof[te] = pipe.predict(x[te])
    return pd.Series(oof, index=frame.index).reindex(block.index)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    seasons = tuple(sorted(set(COMPLETE_SEASONS) | {CURRENT_PARTIAL_SEASON}))
    data = adjust_vaa_for_height(feat.build(load_seasons(seasons)))
    log.info("feature table: %s pitches", f"{len(data):,}")

    bios = fetch_bios(data["pitcher"].dropna().unique())
    log.info("biographical rows: %d", len(bios))

    frames = []
    for pt in PITCH_TYPES:
        block = data[data["pitch_type"].eq(pt)]
        if len(block) < 50_000:
            log.warning("%s: only %s pitches, skipping", pt, f"{len(block):,}")
            continue

        table = pitcher_season_table(block, pitch_type=pt, min_pitches=MIN_PITCHES)
        names = (
            block.groupby(["pitcher", "game_year"], observed=True)["player_name"]
            .first().reset_index()
        )
        table = table.merge(names, on=["pitcher", "game_year"], how="left")
        table["pitch_type"] = pt
        table = table[table["n_swings"].fillna(0) >= MIN_SWINGS]
        if table.empty:
            continue

        # Shape expectation, out-of-fold within each season so a pitcher never
        # informs his own benchmark.
        table["shape_exp"] = np.nan
        for yr, idx in table.groupby("game_year").groups.items():
            table.loc[idx, "shape_exp"] = oof_shape_expectation(table.loc[idx], TARGET)

        # A ridge extrapolates linearly, so a pitcher whose shape sits far
        # outside the normal range gets an expectation no pitcher has ever
        # posted -- a knuckleballer's rare four-seamer drew 1.1%, against a
        # league floor near 6%. Only a handful of rows are affected, but they
        # are by construction the extremes, so they dominate any "most
        # over/under-performing" ranking. Clip to the observed range.
        lo, hi = table[TARGET].quantile([0.01, 0.99])
        clipped = table["shape_exp"].clip(lo, hi)
        n_clipped = int((clipped != table["shape_exp"]).sum())
        if n_clipped:
            log.info("%s: clipped %d shape expectations to [%.3f, %.3f]", pt, n_clipped, lo, hi)
        table["shape_exp"] = clipped

        # The edge model is fit on complete seasons only, then applied.
        train = _pairs(table[table["game_year"].isin(COMPLETE_SEASONS)],
                       TARGET, MIN_PITCHES, MIN_SWINGS)
        if len(train) >= 150:
            pipe = _shape_model(train, TARGET)
            train = train.assign(shape_exp=pipe.predict(train[SHAPE_FEATURES].to_numpy(float)))
            y = train["next_actual"].to_numpy(float)
            naive = Ridge(alpha=1.0, random_state=RANDOM_SEED).fit(train[[TARGET]].to_numpy(float), y)
            full = Ridge(alpha=1.0, random_state=RANDOM_SEED).fit(
                train[[TARGET, "shape_exp"]].to_numpy(float), y)

            usable = table[[TARGET, "shape_exp"]].notna().all(axis=1)
            edge = np.full(len(table), np.nan)
            if usable.any():
                sub = table.loc[usable, [TARGET, "shape_exp"]].to_numpy(float)
                edge[usable.to_numpy()] = full.predict(sub) - naive.predict(sub[:, :1])
            table["shape_edge"] = edge
        else:
            log.warning("%s: only %d training pairs, no edge computed", pt, len(train))
            table["shape_edge"] = np.nan

        frames.append(table)
        log.info("%s: %d pitcher-seasons", pt, len(table))

    combined = pd.concat(frames, ignore_index=True)
    combined = attach_age(combined, bios)

    # Prefer the StatsAPI name: Statcast's "Last, First" reads badly in a table.
    combined["player_name"] = combined["full_name"].fillna(combined["player_name"])

    payload_cols, rows = [], []
    for key, src in COLUMNS:
        payload_cols.append(key)
    for _, r in combined.iterrows():
        row = []
        for key, src in COLUMNS:
            v = r.get(src)
            if pd.isna(v):
                row.append(None)
            elif key in ROUND:
                row.append(round(float(v), ROUND[key]) if ROUND[key] else int(v))
            else:
                row.append(v if isinstance(v, str) else (int(v) if float(v).is_integer() else v))
        rows.append(row)

    payload = {
        "columns": payload_cols,
        "rows": rows,
        "meta": {
            "seasons": [int(min(seasons)), int(max(seasons))],
            "partial_season": int(CURRENT_PARTIAL_SEASON),
            "pitch_types": sorted(combined["pitch_type"].unique().tolist()),
            "min_pitches": MIN_PITCHES,
            "min_swings": MIN_SWINGS,
            "target": TARGET,
            "n_rows": len(rows),
        },
    }

    path = OUT / "pitcher_seasons.json"
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    log.info("wrote %s (%d rows, %.1f MB)", path.name, len(rows), path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
