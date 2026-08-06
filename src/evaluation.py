"""Pitcher-level evaluation analyses: reliability, stability, and forecasting.

The pitch-level study answers "what predicts the outcome of a pitch". That is
not the same question a front office asks, which is "what tells me how good this
pitcher will be *next* year". Those come apart badly, and the reason is
reliability.

An outcome statistic like run value per pitch is mostly noise at the sample
sizes a single season provides. A shape statistic like induced vertical break is
a near-deterministic property of how the pitcher throws. If shape carries even a
modest true signal, it can beat a noisy outcome measure at forecasting, because
it is measured almost without error.

This module quantifies that directly:

  * `split_half_reliability` -- how much of a metric is signal within a season
  * `year_over_year` -- how much persists across seasons
  * `forecast_next_season` -- what actually predicts next year's results
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import MIN_PITCHES_PER_PITCHER_SEASON, RANDOM_SEED

log = logging.getLogger(__name__)

# Physical descriptors of a pitch, averaged over a pitcher-season.
SHAPE_METRICS = [
    "release_speed",
    "release_spin_rate",
    "spin_resid",
    "ivb_in",
    "hb_in",
    "vaa_deg",
    "vaa_adj",
    "haa_armside",
    "release_extension",
    "release_pos_z",
]
OUTCOME_METRICS = ["run_value_pitcher", "whiff_rate"]


def adjust_vaa_for_height(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the part of vertical approach angle that is just pitch location.

    This is the most serious threat to a shape-based thesis. VAA is mechanically
    tied to where the pitch crosses the plate: any pitch at the top of the zone
    arrives flatter than the same pitch at the knees, regardless of how it was
    thrown. A model that credits "shape" for VAA may simply be re-discovering
    that high fastballs play well.

    We therefore regress VAA on plate height (quadratic, within pitch type and
    season) and keep the residual: how flat the pitch arrived *relative to other
    pitches at the same height*. That residual is a property of the pitcher's
    delivery and the ball's flight, not of where he aimed.
    """
    out = df.reset_index(drop=True).copy()
    vaa = out["vaa_deg"].to_numpy(dtype=float)
    height = out["plate_z"].to_numpy(dtype=float)
    adj = np.full(len(out), np.nan)

    for _, pos in out.groupby(["game_year", "pitch_type"], observed=True).indices.items():
        y, z = vaa[pos], height[pos]
        ok = np.isfinite(y) & np.isfinite(z)
        if ok.sum() < 500:
            continue
        coef = np.polyfit(z[ok], y[ok], 2)
        adj[pos[ok]] = y[ok] - np.polyval(coef, z[ok])

    out["vaa_adj"] = adj
    return out


def pitcher_season_table(
    df: pd.DataFrame,
    *,
    pitch_type: str | None = "FF",
    min_pitches: int = MIN_PITCHES_PER_PITCHER_SEASON,
) -> pd.DataFrame:
    """Collapse pitch-level data to one row per pitcher-season."""
    d = df if pitch_type is None else df[df["pitch_type"].eq(pitch_type)]

    present = [m for m in SHAPE_METRICS if m in d.columns]
    grouped = d.groupby(["pitcher", "game_year"], observed=True)
    table = grouped.agg(
        n_pitches=("release_speed", "size"),
        **{m: (m, "mean") for m in present},
        run_value_pitcher=("run_value_pitcher", "mean"),
    ).reset_index()

    swings = (
        d[d["is_swing"]]
        .groupby(["pitcher", "game_year"], observed=True)["is_whiff"]
        .agg(whiff_rate="mean", n_swings="size")
        .reset_index()
    )
    table = table.merge(swings, on=["pitcher", "game_year"], how="left")

    before = len(table)
    table = table[table["n_pitches"] >= min_pitches]
    log.info(
        "pitcher-seasons: %d of %d kept at >=%d pitches", len(table), before, min_pitches
    )
    return table


def split_half_reliability(
    df: pd.DataFrame,
    *,
    pitch_type: str = "FF",
    min_pitches: int = MIN_PITCHES_PER_PITCHER_SEASON,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Correlate each metric between two random halves of the same pitcher-season.

    This separates signal from noise without needing a second season. A metric
    that does not even agree with itself inside one season cannot be used to
    judge a pitcher.

    The Spearman-Brown correction rescales the half-sample correlation up to
    what the full sample would give, which is the number that matters in use.
    """
    d = df[df["pitch_type"].eq(pitch_type)].reset_index(drop=True).copy()
    rng = np.random.default_rng(seed)
    d["half"] = rng.integers(0, 2, len(d))

    metrics = [m for m in SHAPE_METRICS if m in d.columns]
    rows = []

    for half in (0, 1):
        block = d[d["half"] == half]
        agg = block.groupby(["pitcher", "game_year"], observed=True).agg(
            n=("release_speed", "size"),
            **{m: (m, "mean") for m in metrics},
            run_value_pitcher=("run_value_pitcher", "mean"),
        )
        sw = (
            block[block["is_swing"]]
            .groupby(["pitcher", "game_year"], observed=True)["is_whiff"]
            .mean()
            .rename("whiff_rate")
        )
        agg = agg.join(sw)
        agg.columns = [f"{c}_h{half}" for c in agg.columns]
        rows.append(agg)

    joined = rows[0].join(rows[1], how="inner")
    joined = joined[(joined["n_h0"] >= min_pitches / 2) & (joined["n_h1"] >= min_pitches / 2)]

    results = []
    for m in metrics + OUTCOME_METRICS:
        a, b = f"{m}_h0", f"{m}_h1"
        if a not in joined or b not in joined:
            continue
        pair = joined[[a, b]].dropna()
        if len(pair) < 30:
            continue
        r = float(pair[a].corr(pair[b]))
        # Spearman-Brown: reliability of the full (double-length) sample.
        sb = 2 * r / (1 + r) if r > -1 else float("nan")
        results.append(
            {
                "metric": m,
                "kind": "shape" if m in SHAPE_METRICS else "outcome",
                "split_half_r": r,
                "spearman_brown_r": sb,
                "n_pitcher_seasons": len(pair),
            }
        )

    return pd.DataFrame(results).sort_values("spearman_brown_r", ascending=False).reset_index(drop=True)


def year_over_year(table: pd.DataFrame) -> pd.DataFrame:
    """Correlate each metric for the same pitcher in consecutive seasons.

    Year-over-year correlation is the practical question: if I saw this number
    last year, how much does it tell me about this year?
    """
    nxt = table.copy()
    nxt["game_year"] = nxt["game_year"] - 1
    merged = table.merge(nxt, on=["pitcher", "game_year"], suffixes=("_t", "_t1"))

    metrics = [m for m in SHAPE_METRICS + OUTCOME_METRICS if f"{m}_t" in merged.columns]
    rows = []
    for m in metrics:
        pair = merged[[f"{m}_t", f"{m}_t1"]].dropna()
        if len(pair) < 30:
            continue
        rows.append(
            {
                "metric": m,
                "kind": "shape" if m in SHAPE_METRICS else "outcome",
                "yoy_r": float(pair[f"{m}_t"].corr(pair[f"{m}_t1"])),
                "n_pairs": len(pair),
            }
        )
    return pd.DataFrame(rows).sort_values("yoy_r", ascending=False).reset_index(drop=True)


FEATURE_SETS: dict[str, list[str]] = {
    "past results only": ["run_value_pitcher", "whiff_rate"],
    "velocity only": ["release_speed", "release_extension"],
    "velocity + spin": ["release_speed", "release_extension", "release_spin_rate"],
    "shape only": ["ivb_in", "hb_in", "vaa_adj", "haa_armside", "release_pos_z", "release_extension"],
    "shape + velocity": [
        "ivb_in",
        "hb_in",
        "vaa_adj",
        "haa_armside",
        "release_pos_z",
        "release_extension",
        "release_speed",
    ],
    "everything": [
        "ivb_in",
        "hb_in",
        "vaa_adj",
        "haa_armside",
        "release_pos_z",
        "release_extension",
        "release_speed",
        "release_spin_rate",
        "run_value_pitcher",
        "whiff_rate",
    ],
}


def forecast_next_season(
    table: pd.DataFrame,
    *,
    target: str = "run_value_pitcher",
    n_splits: int = 5,
) -> pd.DataFrame:
    """Predict a pitcher's NEXT season from his current one.

    This is the evaluation question in its honest form. If shape beats past
    results here, then shape is the better scouting input -- not because shape
    explains more of what already happened, but because it is measured with less
    error and therefore carries forward.

    Folds are grouped by pitcher so the same arm never appears in train and test.
    """
    nxt = table.copy()
    nxt["game_year"] = nxt["game_year"] - 1
    merged = table.merge(nxt, on=["pitcher", "game_year"], suffixes=("", "_next"))
    target_col = f"{target}_next"

    rows = []
    for name, feats in FEATURE_SETS.items():
        cols = [c for c in feats if c in merged.columns]
        frame = merged[cols + [target_col, "pitcher"]].dropna()
        if len(frame) < 200:
            log.warning("feature set '%s' has only %d rows, skipping", name, len(frame))
            continue

        x = frame[cols].to_numpy(dtype=float)
        y = frame[target_col].to_numpy(dtype=float)
        groups = frame["pitcher"].to_numpy()

        scores = []
        for tr, te in GroupKFold(n_splits=n_splits).split(x, y, groups):
            pipe = make_pipeline(StandardScaler(), Ridge(alpha=10.0, random_state=RANDOM_SEED))
            pipe.fit(x[tr], y[tr])
            pred = pipe.predict(x[te])
            baseline = y[tr].mean()
            ss_res = float(((y[te] - pred) ** 2).sum())
            ss_tot = float(((y[te] - baseline) ** 2).sum())
            scores.append(1.0 - ss_res / ss_tot)

        rows.append(
            {
                "feature_set": name,
                "n_features": len(cols),
                "n_pitcher_seasons": len(frame),
                "cv_r2": float(np.mean(scores)),
                "cv_r2_se": float(np.std(scores, ddof=1) / np.sqrt(len(scores))),
                "corr": float(np.sqrt(max(np.mean(scores), 0.0))),
            }
        )
        log.info("%-20s next-season CV R2 = %+.4f", name, rows[-1]["cv_r2"])

    return pd.DataFrame(rows).sort_values("cv_r2", ascending=False).reset_index(drop=True)
