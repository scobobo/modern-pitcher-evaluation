"""Season-by-season test of the actual claim: is velocity's edge shrinking?

"Velocity is not as important anymore" is a statement about a trend, so a
single pooled model cannot test it. This module refits the same model
separately in every season and tracks how the velocity and spin effects move.

One trap worth naming: as the league throws harder, the *spread* of velocity
can change, and a standardized (per-standard-deviation) coefficient will drift
purely because the denominator moved. So we report both.

  * `coef_per_unit` -- runs saved per extra mph, or per extra 100 rpm. This is
    the physical effect. If hitters have genuinely adapted to velocity, this is
    the number that falls.
  * `coef_per_sd` -- runs saved per standard deviation. This is the practical
    effect for a front office choosing between available pitchers, and it can
    fall simply because everyone now throws hard, with no adaptation at all.

Those two telling different stories is itself the finding, and it is the kind
of distinction that separates a real analysis from a chart.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import RANDOM_SEED
from features import CONTROL_FEATURES, SHAPE_FEATURES, SPIN_RESID_FEATURES, VELOCITY_FEATURES
from model import ModelSpec, cross_validate_spec

log = logging.getLogger(__name__)

# Reported in units a scout would recognise.
SPIN_UNIT = 100.0  # rpm
VELO_UNIT = 1.0  # mph


def season_effects(
    df: pd.DataFrame,
    *,
    target: str = "run_value_pitcher",
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Per-season ridge effects for velocity and spin, in both unit systems."""
    features = features or (
        list(CONTROL_FEATURES) + VELOCITY_FEATURES + SPIN_RESID_FEATURES + SHAPE_FEATURES
    )

    rows = []
    for season, block in df.groupby("game_year", observed=True):
        frame = block[features + [target]].dropna()
        if len(frame) < 5_000:
            log.warning("season %s: only %d usable pitches, skipping", season, len(frame))
            continue

        x = frame[features].astype(float)
        y = frame[target].to_numpy()

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=RANDOM_SEED))
        pipe.fit(x.to_numpy(), y)
        per_sd = dict(zip(features, pipe.named_steps["ridge"].coef_))
        sds = x.std(ddof=0)

        row = {
            "season": int(season),
            "n_pitches": len(frame),
            "mean_velo": float(frame["release_speed"].mean()),
            "sd_velo": float(sds["release_speed"]),
            "mean_spin": float(block["release_spin_rate"].mean()),
            "velo_coef_per_sd": per_sd["release_speed"],
            "spin_coef_per_sd": per_sd["spin_resid"],
            # Convert back out of standardized space: coef_per_unit =
            # coef_per_sd / sd_of_feature, then scale to the reporting unit.
            "velo_coef_per_mph": per_sd["release_speed"] / sds["release_speed"] * VELO_UNIT,
            "spin_coef_per_100rpm": per_sd["spin_resid"] / sds["spin_resid"] * SPIN_UNIT,
        }
        rows.append(row)
        log.info(
            "season %s: velo %+.5f/mph, spin %+.5f/100rpm",
            season,
            row["velo_coef_per_mph"],
            row["spin_coef_per_100rpm"],
        )

    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)


def season_incremental_skill(
    df: pd.DataFrame, *, target: str = "run_value_pitcher", n_splits: int = 4
) -> pd.DataFrame:
    """Per-season: how much unique cross-validated skill does each block add?

    Complements `season_effects` -- a coefficient can stay flat while the
    feature's share of total explained variance falls, and vice versa.
    """
    controls = list(CONTROL_FEATURES)
    rows = []
    for season, block in df.groupby("game_year", observed=True):
        if len(block) < 20_000:
            continue
        try:
            base = cross_validate_spec(
                block, ModelSpec("controls", controls), target=target, n_splits=n_splits
            )
            velo = cross_validate_spec(
                block,
                ModelSpec("+velo", controls + VELOCITY_FEATURES),
                target=target,
                n_splits=n_splits,
            )
            spin = cross_validate_spec(
                block,
                ModelSpec("+spin", controls + SPIN_RESID_FEATURES),
                target=target,
                n_splits=n_splits,
            )
        except ValueError as exc:
            log.warning("season %s skipped: %s", season, exc)
            continue

        rows.append(
            {
                "season": int(season),
                "controls_r2": base.r2_mean,
                "velo_gain": velo.r2_mean - base.r2_mean,
                "spin_gain": spin.r2_mean - base.r2_mean,
                "velo_over_spin": (velo.r2_mean - base.r2_mean) - (spin.r2_mean - base.r2_mean),
            }
        )
        log.info(
            "season %s: velo gain %+.5f vs spin gain %+.5f", season, rows[-1]["velo_gain"], rows[-1]["spin_gain"]
        )

    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)


def trend_test(effects: pd.DataFrame, column: str) -> dict[str, float]:
    """Least-squares slope of an effect over seasons, with a rough t-statistic.

    Deliberately simple: with ~11 seasons there is no power for anything
    fancier, and quoting a p-value from 11 points invites more confidence than
    the data supports. The slope's sign and magnitude are the honest output.
    """
    frame = effects[["season", column]].dropna()
    if len(frame) < 4:
        return {"slope_per_year": float("nan"), "t_stat": float("nan"), "n_seasons": len(frame)}

    x = frame["season"].to_numpy(dtype=float)
    y = frame[column].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    dof = len(x) - 2
    se_slope = np.sqrt((resid**2).sum() / dof / ((x - x.mean()) ** 2).sum())

    return {
        "slope_per_year": float(slope),
        "t_stat": float(slope / se_slope) if se_slope > 0 else float("nan"),
        "n_seasons": len(frame),
        "first_season_value": float(y[0]),
        "last_season_value": float(y[-1]),
    }
