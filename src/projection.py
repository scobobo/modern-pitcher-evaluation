"""Identify regression and progression candidates from the shape-results gap.

The logic follows directly from the paper. A pitcher's shape is measured almost
exactly (split-half r ~ 0.99); his results are mostly noise (r = 0.20). So when
the two disagree, at least part of the disagreement is transient, and the
results should drift back toward what the shape supports.

That gives a usable signal:

    gap = actual outcome - outcome predicted from shape alone

A pitcher with results much better than his shape supports (positive gap on a
"good outcome" measure) is a regression candidate. A pitcher whose shape is
better than his results is a progression candidate.

The important word is *part*. The gap is not pure luck -- it also contains
command, sequencing, and defence, which are real and partly persistent. This
module therefore does not assert that the gap predicts change; it measures how
much of the gap actually reverses, using every consecutive pair of seasons in
the data, and reports the answer including "not much".
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

log = logging.getLogger(__name__)

# Shape and delivery only. No outcome variable may appear here, or the
# "expected" value would be contaminated by the very results it is meant to
# provide an independent benchmark for.
SHAPE_FEATURES = [
    "ivb_in",
    "hb_in",
    "vaa_adj",
    "haa_armside",
    "release_pos_z",
    "release_extension",
    "release_speed",
]


def fit_shape_expectation(
    table: pd.DataFrame,
    *,
    target: str,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Predict an outcome from shape alone, out-of-fold for every pitcher-season.

    Predictions are generated out-of-fold with folds grouped by pitcher, so no
    pitcher-season's expectation is informed by a model that has seen that
    pitcher. An in-sample fit would shrink the gaps toward zero and make the
    whole exercise look better calibrated than it is.
    """
    frame = table[SHAPE_FEATURES + [target, "pitcher"]].dropna().copy()
    x = frame[SHAPE_FEATURES].to_numpy(dtype=float)
    y = frame[target].to_numpy(dtype=float)
    groups = frame["pitcher"].to_numpy()

    oof = np.full(len(frame), np.nan)
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(x, y, groups):
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=10.0, random_state=RANDOM_SEED))
        pipe.fit(x[train_idx], y[train_idx])
        oof[test_idx] = pipe.predict(x[test_idx])

    frame[f"{target}_expected"] = oof
    frame[f"{target}_gap"] = frame[target] - oof

    out = table.merge(
        frame[["pitcher", f"{target}_expected", f"{target}_gap"]].assign(
            _row=frame.index
        ).set_index("_row")[[f"{target}_expected", f"{target}_gap"]],
        left_index=True,
        right_index=True,
        how="left",
    )
    return out


def measure_gap_reversal(table: pd.DataFrame, *, target: str, min_pitches: int = 250) -> dict:
    """How much of this year's shape-results gap disappears next year?

    This is the validation step, and the one that decides whether any of this is
    usable. For every pitcher who appears in consecutive seasons we regress next
    season's outcome on this season's outcome and this season's shape-based
    expectation. The coefficients say how the two should be weighted.

    A `reversal_fraction` near 1 means the gap is essentially all transient and
    fully reverses. Near 0 means the gap is a real, persistent property of the
    pitcher and carries no regression signal at all.
    """
    gap_col, exp_col = f"{target}_gap", f"{target}_expected"
    cur = table[table["n_pitches"] >= min_pitches]
    nxt = cur[["pitcher", "game_year", target]].copy()
    nxt["game_year"] -= 1
    nxt = nxt.rename(columns={target: "next_actual"})

    merged = cur.merge(nxt, on=["pitcher", "game_year"], how="inner")
    merged = merged[[target, exp_col, gap_col, "next_actual"]].dropna()
    if len(merged) < 100:
        return {"n_pairs": len(merged), "usable": False}

    # next ~ actual + expected. If shape carries independent information about
    # next season, its coefficient is non-zero even with actual in the model.
    x = merged[[target, exp_col]].to_numpy(dtype=float)
    y = merged["next_actual"].to_numpy(dtype=float)
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=RANDOM_SEED))
    pipe.fit(x, y)
    beta_actual, beta_expected = pipe.named_steps["ridge"].coef_

    # How much of the gap reverses: regress the change on the gap. A slope of
    # -1 means the gap vanishes entirely next season.
    change = merged["next_actual"] - merged[target]
    gap = merged[gap_col]
    slope = float(np.polyfit(gap, change, 1)[0])

    resid = change - np.polyval(np.polyfit(gap, change, 1), gap)
    dof = len(gap) - 2
    se = float(np.sqrt((resid**2).sum() / dof / ((gap - gap.mean()) ** 2).sum()))

    return {
        "n_pairs": len(merged),
        "usable": True,
        "beta_actual_per_sd": float(beta_actual),
        "beta_expected_per_sd": float(beta_expected),
        "shape_weight": float(abs(beta_expected) / (abs(beta_actual) + abs(beta_expected))),
        "reversal_slope": slope,
        "reversal_se": se,
        "reversal_t": slope / se if se > 0 else float("nan"),
        "reversal_fraction": -slope,
        "gap_sd": float(gap.std()),
    }


def rank_candidates(
    table: pd.DataFrame,
    *,
    season: int,
    target: str,
    reversal_fraction: float,
    min_pitches: int = 150,
    top_n: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank the largest over- and under-performers relative to their shape.

    The projected change is the gap scaled by the fraction of it that
    historically reverses -- not the raw gap, which would overstate the move by
    treating command and defence as if they were luck.
    """
    gap_col, exp_col = f"{target}_gap", f"{target}_expected"
    block = table[(table["game_year"] == season) & (table["n_pitches"] >= min_pitches)].copy()
    block = block.dropna(subset=[gap_col, target])

    block["projected_change"] = -block[gap_col] * reversal_fraction
    block["projected_next"] = block[target] + block["projected_change"]

    cols = [
        "player_name",
        "pitcher",
        "n_pitches",
        "release_speed",
        "ivb_in",
        "vaa_adj",
        target,
        exp_col,
        gap_col,
        "projected_change",
        "projected_next",
    ]
    cols = [c for c in cols if c in block.columns]

    regression = block.nlargest(top_n, gap_col)[cols].reset_index(drop=True)
    progression = block.nsmallest(top_n, gap_col)[cols].reset_index(drop=True)
    return regression, progression
