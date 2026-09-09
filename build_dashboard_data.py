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
from projection import (
    SEPARATION_FEATURES,
    SHAPE_FEATURES,
    add_separation,
    primary_fastball,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from config import RANDOM_SEED

log = logging.getLogger("dashboard_data")

PITCH_TYPES = ["FF", "SI", "FC", "SL", "ST", "CU", "CH"]
MIN_PITCHES = 150
MIN_SWINGS = 60
# The dashboard will not put a name on a board below this many pitches, so any
# statistic describing the boards has to be measured on the same population.
# Measured at 150 the gap looks far more temporary than it is, because thin
# samples carry more noise and noise always reverses.
NAMEABLE_PITCHES = 300
# The edge model trains on a rolling window of recent seasons rather than all
# history. The relationship between shape and results drifts -- league velocity
# is up 1.7 mph since 2015 and whiff rate is up 2.4 points -- so seasons from a
# decade ago describe a game that is no longer being played, and including them
# measurably degrades recent predictions.
#
# Measured walk-forward on four-seamers, comparing training windows:
#
#   window        edge/residual r (2022+)   named candidates beat naive
#   all history            0.205                   +1.86 pp (t=+5.7)
#   5 seasons              0.246                   +2.01 pp (t=+5.9)
#   4 seasons              0.263                   +2.38 pp (t=+6.8)
#   3 seasons              0.287                   +2.48 pp (t=+7.2)
#
# It also removes the decay that made the old boards untrustworthy: the trend in
# per-season edge quality goes from -0.0199/yr (t=-2.36, p=0.05) on all history
# to -0.0112/yr (t=-1.46, p=0.19) at four seasons, which is no longer
# distinguishable from flat.
#
# Three seasons scored best but falls to 183 training pairs in one year, and
# picking the top window on the same data used to evaluate it is a way to
# overfit a hyperparameter. Four keeps 391-595 pairs and gets most of the gain.
TRAIN_WINDOW = 4
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


def _verdict(t: float) -> str:
    """Four states, not two.

    Collapsing everything that fails the significance bar into one bucket reads
    as "this pitch does not matter", which is both discouraging and wrong. A
    slider whose edge points the right way at t = 1.9 is in a completely
    different position from a pitch where the effect is flat or backwards, and
    the reader deserves to be told which one they are looking at.

    The bar for "validated" stays t > 2, the same standard the paper uses.
    """
    if t > 2:
        return "validated"
    if t > 1:
        return "promising"
    if t < -1.5:
        return "inverted"
    return "no signal"


def _persistence(block: pd.DataFrame, target: str) -> dict:
    """How much of a shape-results gap is temporary, and how much is the pitcher?

    The boards rest on the idea that a disagreement between results and shape
    is partly luck and will partly correct. That is true, but only about half
    true, and the half that does not correct is the part a reader most needs
    warned about: it is why the same names come back year after year.

    Measured two ways. The reversal fraction is one minus the year-over-year
    slope of the gap on itself. The recurrence rate is how often a pitcher in
    the top or bottom twenty by edge is there again the following season,
    against the rate you would get by drawing twenty names at random from the
    qualifying pool.
    """
    g = block.dropna(subset=["shape_exp", target, "shape_edge"])
    g = g[g["n_pitches"] >= NAMEABLE_PITCHES]
    if "n_swings" in g.columns:
        g = g[g["n_swings"] >= MIN_SWINGS]
    g = g[g["game_year"] <= max(COMPLETE_SEASONS)].copy()
    g["gap"] = g[target] - g["shape_exp"]

    nxt = g[["pitcher", "game_year", "gap"]].copy()
    nxt["game_year"] -= 1
    nxt = nxt.rename(columns={"gap": "gap_next"})
    m = g.merge(nxt, on=["pitcher", "game_year"]).dropna(subset=["gap", "gap_next"])
    if len(m) < 120:
        return {"n": int(len(m)), "reversal_pct": None, "recurrence_pct": None, "chance_pct": None}

    slope = float(np.polyfit(m["gap"], m["gap_next"], 1)[0])
    reversal = max(0.0, min(1.0, 1.0 - slope))

    repeats = total = 0
    pool_sizes = []
    for season in sorted(g["game_year"].unique()):
        cur = g[g["game_year"] == season]
        nxt_season = g[g["game_year"] == season + 1]
        if len(cur) < 40 or nxt_season.empty:
            continue
        pool_sizes.append(len(nxt_season))
        for frame in (cur.nlargest(20, "shape_edge"), cur.nsmallest(20, "shape_edge")):
            names = set(frame["pitcher"])
            board_next = set(nxt_season.nlargest(20, "shape_edge")["pitcher"]) | set(
                nxt_season.nsmallest(20, "shape_edge")["pitcher"])
            total += len(names)
            repeats += len(names & board_next)

    if not total or not pool_sizes:
        return {"n": int(len(m)), "reversal_pct": round(reversal * 100),
                "recurrence_pct": None, "chance_pct": None}

    # Recurrence only means something when the board is actually selective. For
    # a thin pitch type the two boards name forty pitchers out of a qualifying
    # pool of fifty, so "they came back" is arithmetic rather than signal, and
    # the chance rate can exceed the observed rate. Report nothing rather than
    # something that reads as a finding.
    chance = min(1.0, 40.0 / (sum(pool_sizes) / len(pool_sizes)))
    observed = repeats / total
    selective = chance <= 0.5 * observed
    return {
        "n": int(len(m)),
        "reversal_pct": round(reversal * 100),
        "recurrence_pct": round(100 * observed) if selective else None,
        "chance_pct": round(100 * chance) if selective else None,
    }


def _walk_forward_verdict(block: pd.DataFrame, target: str) -> dict:
    """Does the edge predict next season, using only prior seasons to build it?

    For each season the naive and full models are fit on pairs that closed
    before it, the edge is computed for that season's pitchers, and the result
    is scored against what they actually did the following year. Seasons with
    too little history to train on contribute nothing, which is why the sample
    here is smaller than the season count suggests.
    """
    g = block.dropna(subset=["shape_exp", target])
    g = g[g["n_pitches"] >= MIN_PITCHES]
    if "n_swings" in g.columns:
        g = g[g["n_swings"] >= MIN_SWINGS]

    scored = []
    for season in sorted(g["game_year"].unique()):
        if season > max(COMPLETE_SEASONS):
            continue
        history = g[(g["game_year"] < season) & (g["game_year"] >= season - TRAIN_WINDOW)]
        train = _pairs(history, target, MIN_PITCHES, MIN_SWINGS)
        train = train.dropna(subset=[target, "shape_exp", "next_actual"])
        if len(train) < 120:
            continue

        y = train["next_actual"].to_numpy(float)
        naive = Ridge(alpha=1.0, random_state=RANDOM_SEED).fit(train[[target]].to_numpy(float), y)
        full = Ridge(alpha=1.0, random_state=RANDOM_SEED).fit(
            train[[target, "shape_exp"]].to_numpy(float), y)

        board = g[g["game_year"] == season].copy()
        if board.empty:
            continue
        x_full = board[[target, "shape_exp"]].to_numpy(float)
        board["wf_edge"] = full.predict(x_full) - naive.predict(x_full[:, :1])
        board["wf_naive"] = naive.predict(x_full[:, :1])

        future = g[g["game_year"] == season + 1][["pitcher", target]].rename(
            columns={target: "next_actual"})
        scored.append(board.merge(future, on="pitcher", how="inner"))

    if not scored:
        return {"n": 0, "r": None, "t": None, "verdict": "untested"}

    m = pd.concat(scored, ignore_index=True).dropna(subset=["next_actual", "wf_edge"])
    if len(m) < 120:
        return {"n": int(len(m)), "r": None, "t": None, "verdict": "untested"}

    # What mean reversion alone failed to predict, and whether the edge knew.
    resid = m["next_actual"] - m["wf_naive"]
    r = float(np.corrcoef(m["wf_edge"], resid)[0, 1])
    t = r * np.sqrt((len(m) - 2) / max(1e-12, 1 - r**2))
    return {"n": int(len(m)), "r": round(r, 3), "t": round(float(t), 1),
            "verdict": _verdict(float(t))}


def oof_shape_expectation(block: pd.DataFrame, target: str,
                          features: list[str] | None = None) -> pd.Series:
    """Shape expectation, out-of-fold and grouped by pitcher.

    `features` carries the separation columns for secondary pitches. They are
    zero for whichever fastball the pitcher leads with, so the same feature list
    can be used for every pitch type without special-casing fastballs.
    """
    features = features or SHAPE_FEATURES
    frame = block[features + [target, "pitcher"]].dropna()
    if len(frame) < 60:
        return pd.Series(np.nan, index=block.index)
    x = frame[features].to_numpy(float)
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

    # One reference fastball per pitcher-season, so every secondary pitch can be
    # measured against the pitch hitters are timing.
    recent_seasons = [y for y in COMPLETE_SEASONS if y > max(COMPLETE_SEASONS) - TRAIN_WINDOW]
    reference = primary_fastball(data)
    log.info("reference fastballs: %s pitcher-seasons", f"{len(reference):,}")

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

        rel_x = (block.groupby(["pitcher", "game_year"], observed=True)["release_pos_x"]
                 .mean().rename("release_pos_x").reset_index())
        table = table.merge(rel_x, on=["pitcher", "game_year"], how="left")
        table = add_separation(table, reference)
        model_features = SHAPE_FEATURES + SEPARATION_FEATURES

        # Shape expectation, out-of-fold within each season so a pitcher never
        # informs his own benchmark.
        table["shape_exp"] = np.nan
        for yr, idx in table.groupby("game_year").groups.items():
            table.loc[idx, "shape_exp"] = oof_shape_expectation(
                table.loc[idx], TARGET, model_features)

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

        # The edge model is fit on the most recent complete seasons, then
        # applied to every row. See TRAIN_WINDOW for why it is not all history.
        recent = [y for y in COMPLETE_SEASONS if y > max(COMPLETE_SEASONS) - TRAIN_WINDOW]
        train = _pairs(table[table["game_year"].isin(recent)],
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

    # Validate the edge separately for every pitch type, and ship the verdict
    # with the data. The model is fit per pitch type but it does not work
    # equally well on all of them, and a dashboard that presents a sweeper
    # candidate with the same confidence as a four-seam one is lying by
    # omission.
    #
    # The validation must be walk-forward. An earlier version scored the
    # shipped `shape_edge`, which is fit on every complete season, against
    # pairs drawn from those same seasons; each pair was therefore judged by a
    # model that had seen its own future. That is generous by roughly three
    # t-points, enough to turn "no signal" into a validated badge on cutters,
    # sliders and curveballs. Here the edge is refit for each season using only
    # the seasons before it, which is the information a user actually has when
    # reading the board.
    validation = {}
    for pt, g in combined.groupby("pitch_type"):
        validation[pt] = _walk_forward_verdict(g, TARGET)
        v = validation[pt]
        log.info("%s edge validation (walk-forward): r=%s t=%s (%s, n=%d)",
                 pt, v["r"], v["t"], v["verdict"], v["n"])

    # How much of a gap actually corrects. Measured per pitch type, because the
    # answer differs a lot between them -- a four-seam gap is about half
    # permanent, a breaking-ball gap much less so -- and the panel quotes this
    # number next to whichever pitch the reader has selected.
    persistence = {}
    for pt, g in combined.groupby("pitch_type"):
        persistence[pt] = _persistence(g, TARGET)

    # The all-pitches figure is the weighted average of the per-type ones, not a
    # pooled fit. Pooling types with different whiff baselines manufactures
    # reversion that is really just between-type variance: it reported 78% where
    # every individual pitch type sits between 35% and 47%.
    parts = [(v["n"], v["reversal_pct"]) for v in persistence.values() if v["reversal_pct"] is not None]
    if parts:
        weight = sum(n for n, _ in parts)
        persistence["_all"] = {
            "n": weight,
            "reversal_pct": round(sum(n * r for n, r in parts) / weight),
            "recurrence_pct": None,
            "chance_pct": None,
        }
    for pt, v in sorted(persistence.items()):
        if v["reversal_pct"] is not None:
            log.info("%s gap persistence: %s%% reverses; repeats %s%% vs %s%% by chance (n=%d)",
                     pt, v["reversal_pct"], v["recurrence_pct"], v["chance_pct"], v["n"])

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
            "train_window": TRAIN_WINDOW,
            "train_seasons": [int(min(recent_seasons)), int(max(recent_seasons))],
            "tracked_pitches": int(len(data)),
            "n_rows": len(rows),
            "validation": validation,
            "persistence": persistence,
        },
    }

    path = OUT / "pitcher_seasons.json"
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    log.info("wrote %s (%d rows, %.1f MB)", path.name, len(rows), path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
