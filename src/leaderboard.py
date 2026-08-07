"""Full shape leaderboard, plus the walk-forward backtest that validates it.

The ranking in `projection.py` is only worth publishing if it survives an
honest out-of-sample test. Two forms of leakage have to be excluded:

  1. **Temporal.** Fitting the shape model on 2015-2025 and then "predicting"
     2018 uses the future to forecast the past. Every model here is fit on
     seasons strictly earlier than the one being scored.
  2. **Pitcher identity.** The shape expectation must never be informed by a
     model that has already seen that pitcher, or the gap shrinks toward zero
     and the tool looks better calibrated than it is.

The question the backtest answers is deliberately narrow, because it is the
only one that matters: *after mean reversion has taken its cut, does shape
explain any of what is left?* Anything that fails to beat the naive
mean-reversion model out-of-sample is not a finding.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import RANDOM_SEED
from projection import SHAPE_FEATURES

log = logging.getLogger(__name__)


def _shape_model(train: pd.DataFrame, target: str):
    frame = train[SHAPE_FEATURES + [target]].dropna()
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=10.0, random_state=RANDOM_SEED))
    pipe.fit(frame[SHAPE_FEATURES].to_numpy(float), frame[target].to_numpy(float))
    return pipe


def _oof_shape_expectation(block: pd.DataFrame, target: str, n_splits: int = 5) -> pd.Series:
    """Out-of-fold shape expectation within one season, grouped by pitcher."""
    frame = block[SHAPE_FEATURES + [target, "pitcher"]].dropna()
    if len(frame) < 50:
        return pd.Series(np.nan, index=block.index)
    x = frame[SHAPE_FEATURES].to_numpy(float)
    y = frame[target].to_numpy(float)
    groups = frame["pitcher"].to_numpy()
    oof = np.full(len(frame), np.nan)
    splits = min(n_splits, len(np.unique(groups)))
    for tr, te in GroupKFold(n_splits=splits).split(x, y, groups):
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=10.0, random_state=RANDOM_SEED))
        pipe.fit(x[tr], y[tr])
        oof[te] = pipe.predict(x[te])
    return pd.Series(oof, index=frame.index).reindex(block.index)


def _pairs(table: pd.DataFrame, target: str, min_pitches: int, min_swings: int) -> pd.DataFrame:
    """Pitcher-seasons joined to the following season's outcome."""
    cur = table[table["n_pitches"] >= min_pitches].copy()
    if "n_swings" in cur.columns and target == "whiff_rate":
        cur = cur[cur["n_swings"].fillna(0) >= min_swings]
    nxt = cur[["pitcher", "game_year", target]].copy()
    nxt["game_year"] -= 1
    nxt = nxt.rename(columns={target: "next_actual"})
    return cur.merge(nxt, on=["pitcher", "game_year"], how="inner")


def walk_forward_backtest(
    table: pd.DataFrame,
    *,
    target: str = "whiff_rate",
    seasons: tuple[int, ...],
    min_pitches: int = 250,
    min_swings: int = 100,
    min_train: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score each season using only the seasons before it.

    Returns (per-season results, pooled predictions) so the caller can compute
    aggregate statistics over genuinely out-of-sample rows.
    """
    rows, pooled = [], []

    for season in sorted(seasons):
        train_tbl = table[table["game_year"] < season]
        train = _pairs(train_tbl, target, min_pitches, min_swings)
        if len(train) < min_train:
            log.info("season %s: only %d training pairs, skipping", season, len(train))
            continue

        # Shape expectation: fit on prior seasons, applied to this season.
        shape_pipe = _shape_model(train, target)
        train = train.assign(shape_exp=shape_pipe.predict(train[SHAPE_FEATURES].to_numpy(float)))

        test_tbl = table[table["game_year"] == season]
        test = _pairs(pd.concat([test_tbl, table[table["game_year"] == season + 1]]),
                      target, min_pitches, min_swings)
        test = test[test["game_year"] == season]
        if test.empty:
            continue
        test = test.dropna(subset=SHAPE_FEATURES + [target, "next_actual"])
        if len(test) < 30:
            continue
        test = test.assign(shape_exp=shape_pipe.predict(test[SHAPE_FEATURES].to_numpy(float)))

        y_tr = train["next_actual"].to_numpy(float)
        naive = Ridge(alpha=1.0, random_state=RANDOM_SEED).fit(train[[target]].to_numpy(float), y_tr)
        full = Ridge(alpha=1.0, random_state=RANDOM_SEED).fit(
            train[[target, "shape_exp"]].to_numpy(float), y_tr
        )

        y_te = test["next_actual"].to_numpy(float)
        p_naive = naive.predict(test[[target]].to_numpy(float))
        p_full = full.predict(test[[target, "shape_exp"]].to_numpy(float))
        base = y_tr.mean()

        def r2(pred):
            ss_res = float(((y_te - pred) ** 2).sum())
            ss_tot = float(((y_te - base) ** 2).sum())
            return 1 - ss_res / ss_tot

        test = test.assign(
            proj_naive=p_naive,
            proj_full=p_full,
            shape_edge=p_full - p_naive,
            naive_residual=y_te - p_naive,
        )
        pooled.append(test)

        rows.append(
            {
                "season": season,
                "n_train_pairs": len(train),
                "n_test": len(test),
                "naive_r2": r2(p_naive),
                "full_r2": r2(p_full),
                "shape_lift": r2(p_full) - r2(p_naive),
                "edge_vs_residual_r": float(np.corrcoef(test["shape_edge"], test["naive_residual"])[0, 1]),
            }
        )
        log.info(
            "season %s -> %s: naive R2 %+.4f, +shape %+.4f (lift %+.4f), edge/residual r %+.3f",
            season, season + 1, rows[-1]["naive_r2"], rows[-1]["full_r2"],
            rows[-1]["shape_lift"], rows[-1]["edge_vs_residual_r"],
        )

    return pd.DataFrame(rows), (pd.concat(pooled, ignore_index=True) if pooled else pd.DataFrame())


def decile_table(pooled: pd.DataFrame, *, target: str, n_bins: int = 10) -> pd.DataFrame:
    """Do high-edge pitchers actually beat mean reversion, in order?

    Bucketing by predicted edge and reading the realised residual is the test a
    practitioner cares about: if the buckets are not monotone, the ranking is
    not usable even when the pooled correlation is significant.
    """
    d = pooled.dropna(subset=["shape_edge", "naive_residual"]).copy()
    d["bucket"] = pd.qcut(d["shape_edge"], n_bins, labels=False, duplicates="drop")
    g = d.groupby("bucket").agg(
        n=("shape_edge", "size"),
        mean_edge=("shape_edge", "mean"),
        mean_residual=("naive_residual", "mean"),
        actual_next=("next_actual", "mean"),
        this_season=(target, "mean"),
    )
    g["beat_mean_reversion_pct"] = d.groupby("bucket")["naive_residual"].apply(lambda s: (s > 0).mean() * 100)
    return g.reset_index()


def build_leaderboard(
    table: pd.DataFrame,
    *,
    season: int,
    target: str,
    train_seasons: tuple[int, ...],
    min_pitches: int = 250,
    min_swings: int = 100,
) -> pd.DataFrame:
    """Full ranked leaderboard for one season, fit only on earlier seasons."""
    train = _pairs(table[table["game_year"].isin(train_seasons)], target, min_pitches, min_swings)
    shape_pipe = _shape_model(train, target)
    train = train.assign(shape_exp=shape_pipe.predict(train[SHAPE_FEATURES].to_numpy(float)))

    y = train["next_actual"].to_numpy(float)
    naive = Ridge(alpha=1.0, random_state=RANDOM_SEED).fit(train[[target]].to_numpy(float), y)
    full = Ridge(alpha=1.0, random_state=RANDOM_SEED).fit(
        train[[target, "shape_exp"]].to_numpy(float), y
    )

    block = table[table["game_year"] == season].copy()
    block = block[block["n_pitches"] >= min_pitches]
    if "n_swings" in block.columns and target == "whiff_rate":
        block = block[block["n_swings"].fillna(0) >= min_swings]
    block = block.dropna(subset=SHAPE_FEATURES + [target])
    if block.empty:
        return block

    # Within-season out-of-fold expectation, so a pitcher's own season does not
    # inform his own benchmark.
    block["shape_exp"] = _oof_shape_expectation(block, target)
    block["shape_exp"] = block["shape_exp"].fillna(
        pd.Series(shape_pipe.predict(block[SHAPE_FEATURES].to_numpy(float)), index=block.index)
    )

    block["proj_naive"] = naive.predict(block[[target]].to_numpy(float))
    block["proj_full"] = full.predict(block[[target, "shape_exp"]].to_numpy(float))
    block["shape_edge"] = block["proj_full"] - block["proj_naive"]
    block["projected_change"] = block["proj_full"] - block[target]
    block["edge_percentile"] = block["shape_edge"].rank(pct=True) * 100
    block["verdict"] = np.where(
        block["edge_percentile"] >= 80, "progression",
        np.where(block["edge_percentile"] <= 20, "regression", "neutral"),
    )

    cols = [
        "player_name", "pitcher", "game_year", "n_pitches", "n_swings",
        "release_speed", "ivb_in", "hb_in", "vaa_adj", "release_extension",
        target, "shape_exp", "proj_naive", "proj_full", "shape_edge",
        "projected_change", "edge_percentile", "verdict",
    ]
    cols = [c for c in cols if c in block.columns]
    return block[cols].sort_values("shape_edge", ascending=False).reset_index(drop=True)
