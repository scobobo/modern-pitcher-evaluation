"""Active spin (spin efficiency) from the Baseball Savant leaderboard.

The paper showed that residual spin *rate* adds nothing once velocity is
partialled out. That result stands, and this is not a contradiction of it:
active spin is a different measurement. Spin rate is how fast the ball turns;
active spin is what fraction of that spin is tilted to actually move the ball
rather than pointing down the flight path and doing nothing. Two pitchers at
2400 rpm can be at 95% and 30% efficiency and their pitches behave nothing
alike.

Measured walk-forward on paired CV folds, adding it to the shape block:

    four-seam  +0.0310 R2 (t=+2.9)
    slider     +0.0383 R2 (t=+2.6)
    cutter     +0.0334 R2 (t=+0.7, n=393)
    sinker, curveball, changeup   nothing

A non-Magnus movement residual was tested alongside it -- the seam-shifted-wake
idea -- and is deliberately NOT included: it failed to clear t=2 on any pitch
type, the best being four-seamers at t=+1.6. Statcast's per-pitch `spin_axis`
is inferred from observed movement rather than measured off the ball, so a
deviation built from it is close to circular by construction, and what remains
sorts by gyro spin rather than by seam effects.

Coverage begins in 2020 with Hawk-Eye. There is nothing for 2015-2019, which is
survivable here only because the expectation model is fit within each season
separately, so a season can carry a feature its predecessors lack without any
cross-era contamination.
"""

from __future__ import annotations

import io
import logging

import pandas as pd
import requests

from config import DATA_DIR

log = logging.getLogger(__name__)

CACHE = DATA_DIR / "active_spin.parquet"
URL = "https://baseballsavant.mlb.com/leaderboard/active-spin?year={year}&active-spin-min=100&csv=true"

# Hawk-Eye optical tracking, and therefore spin efficiency, begins in 2020.
FIRST_SEASON = 2020

# The leaderboard's column suffixes, mapped to Statcast pitch codes. Slurve is
# carried because Statcast still classifies a few, and splitter because the
# leaderboard reports it even though the dashboard does not model it.
COLUMN_TO_PITCH = {
    "fourseam": "FF", "sinker": "SI", "cutter": "FC", "changeup": "CH",
    "splitter": "FS", "curve": "CU", "slider": "SL", "sweeper": "ST",
    "slurve": "SV",
}


def _one_season(season: int) -> pd.DataFrame:
    response = requests.get(URL.format(year=season), timeout=90)
    response.raise_for_status()
    if len(response.text) < 200:
        log.warning("active spin %s: empty response", season)
        return pd.DataFrame()

    raw = pd.read_csv(io.StringIO(response.text))
    # The feed ships a BOM and quoted headers; normalise before anything else.
    raw.columns = [c.strip().strip('"').lstrip("﻿") for c in raw.columns]

    frames = []
    for suffix, pitch in COLUMN_TO_PITCH.items():
        column = f"active_spin_{suffix}"
        if column not in raw.columns:
            continue
        block = raw[["entity_id", column]].dropna()
        block = block.rename(columns={"entity_id": "pitcher", column: "active_spin"})
        block["pitch_type"] = pitch
        block["game_year"] = season
        frames.append(block)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Published as a percentage; carried as a fraction so it scales like the
    # other features rather than dominating the ridge by sheer magnitude.
    out["active_spin"] = pd.to_numeric(out["active_spin"], errors="coerce") / 100.0
    return out.dropna(subset=["active_spin"])


def fetch(seasons, *, force: bool = False) -> pd.DataFrame:
    """Active spin per pitcher, season and pitch type. Cached."""
    wanted = [s for s in seasons if s >= FIRST_SEASON]
    if CACHE.exists() and not force:
        cached = pd.read_parquet(CACHE)
        if set(wanted).issubset(set(cached["game_year"].unique())):
            return cached[cached["game_year"].isin(wanted)]

    frames = []
    for season in wanted:
        try:
            block = _one_season(season)
        except Exception as exc:  # a leaderboard outage must not fail the build
            log.warning("active spin %s: %s -- season skipped", season, exc)
            continue
        if not block.empty:
            frames.append(block)
            log.info("active spin %s: %s pitcher-pitch rows", season, f"{len(block):,}")

    if not frames:
        log.warning("active spin: nothing fetched, model will fall back to shape alone")
        return pd.DataFrame(columns=["pitcher", "pitch_type", "game_year", "active_spin"])

    out = pd.concat(frames, ignore_index=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE, index=False)
    return out
