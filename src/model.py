"""Nested models that attribute pitch value to velocity, spin, and shape.

The design is deliberately built so the hypothesis can lose.

We fit a sequence of models that differ only in which block of features they
are allowed to see, and score each by cross-validated skill on held-out
*pitchers*. The gain from adding a block is that block's unique contribution,
because every earlier block is already in the model. If spin genuinely carries
information that velocity does not, adding the spin block to a model that
already knows velocity must improve out-of-sample skill. If it does not, the
hypothesis is wrong, and the code will say so.

Grouping the folds by pitcher matters. Pitch-level random splits leak: the same
pitcher's fastballs appear in train and test, and the model can memorise the
pitcher instead of learning the physics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import RANDOM_SEED
from features import (
    CONTROL_FEATURES,
    SHAPE_FEATURES,
    SPIN_RAW_FEATURES,
    SPIN_RESID_FEATURES,
    VELOCITY_FEATURES,
)

log = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    """One rung of the nested ladder."""

    name: str
    features: list[str]
    note: str = ""


def build_ladder(*, use_resid_spin: bool = True) -> list[ModelSpec]:
    """Feature blocks added one at a time, each building on the last.

    Order is chosen to be maximally unkind to the hypothesis: velocity gets
    first claim on any shared variance, and spin has to earn its keep on top of
    it. Reversing the order (see `build_reverse_ladder`) is the fair
    counterpart, and the honest write-up reports both.
    """
    spin_block = SPIN_RESID_FEATURES if use_resid_spin else SPIN_RAW_FEATURES
    controls = list(CONTROL_FEATURES)
    return [
        ModelSpec("controls", controls, "count, location, platoon only"),
        ModelSpec("+velocity", controls + VELOCITY_FEATURES, "adds release speed and extension"),
        ModelSpec("+spin", controls + VELOCITY_FEATURES + spin_block, "adds spin on top of velocity"),
        ModelSpec(
            "+shape",
            controls + VELOCITY_FEATURES + spin_block + SHAPE_FEATURES,
            "adds movement and approach angle",
        ),
    ]


def build_reverse_ladder(*, use_resid_spin: bool = True) -> list[ModelSpec]:
    """The same blocks with spin given first claim on shared variance."""
    spin_block = SPIN_RESID_FEATURES if use_resid_spin else SPIN_RAW_FEATURES
    controls = list(CONTROL_FEATURES)
    return [
        ModelSpec("controls", controls, "count, location, platoon only"),
        ModelSpec("+spin", controls + spin_block, "adds spin before velocity"),
        ModelSpec("+velocity", controls + spin_block + VELOCITY_FEATURES, "adds velocity on top of spin"),
        ModelSpec(
            "+shape",
            controls + spin_block + VELOCITY_FEATURES + SHAPE_FEATURES,
            "adds movement and approach angle",
        ),
    ]


@dataclass
class CVResult:
    name: str
    features: list[str]
    r2_mean: float
    r2_std: float
    fold_scores: list[float] = field(default_factory=list)


def _prepare(df: pd.DataFrame, features: list[str], target: str):
    frame = df[features + [target, "pitcher"]].dropna()
    x = frame[features].astype(float).to_numpy()
    y = frame[target].to_numpy()
    groups = frame["pitcher"].to_numpy()
    return x, y, groups


def cross_validate_spec(
    df: pd.DataFrame,
    spec: ModelSpec,
    *,
    target: str = "run_value_pitcher",
    n_splits: int = 5,
) -> CVResult:
    """Score one model spec with pitcher-grouped cross-validation."""
    x, y, groups = _prepare(df, spec.features, target)
    if len(np.unique(groups)) < n_splits:
        raise ValueError(f"only {len(np.unique(groups))} pitchers -- cannot make {n_splits} folds")

    scores: list[float] = []
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(x, y, groups):
        model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.06,
            max_depth=6,
            min_samples_leaf=200,
            l2_regularization=1.0,
            random_state=RANDOM_SEED,
        )
        model.fit(x[train_idx], y[train_idx])
        pred = model.predict(x[test_idx])
        # R^2 against the training mean, so a useless model scores ~0 rather
        # than being flattered by the test set's own mean.
        baseline = y[train_idx].mean()
        ss_res = float(((y[test_idx] - pred) ** 2).sum())
        ss_tot = float(((y[test_idx] - baseline) ** 2).sum())
        scores.append(1.0 - ss_res / ss_tot)

    return CVResult(
        name=spec.name,
        features=spec.features,
        r2_mean=float(np.mean(scores)),
        r2_std=float(np.std(scores)),
        fold_scores=scores,
    )


def run_ladder(
    df: pd.DataFrame,
    ladder: list[ModelSpec],
    *,
    target: str = "run_value_pitcher",
    n_splits: int = 5,
) -> pd.DataFrame:
    """Fit every rung and report the incremental skill of each block."""
    # Every rung must be scored on the SAME rows, or the comparison is
    # meaningless. Features have different missingness -- release_extension in
    # particular is null for a couple of percent of pitches -- so a per-rung
    # dropna would quietly change the sample between rungs, and the "gain" from
    # a block would partly be the gain from switching to an easier subset.
    # Restrict once to complete cases across every feature any rung uses.
    all_features = sorted({f for spec in ladder for f in spec.features})
    before = len(df)
    frame = df[all_features + [target, "pitcher"]].dropna()
    log.info(
        "ladder common sample: %s of %s rows complete across all %d features",
        f"{len(frame):,}",
        f"{before:,}",
        len(all_features),
    )

    rows = []
    previous: CVResult | None = None
    for spec in ladder:
        result = cross_validate_spec(frame, spec, target=target, n_splits=n_splits)

        # GroupKFold is deterministic, so rung k and rung k-1 are scored on the
        # identical folds. That makes the incremental gain a *paired* quantity:
        # we can difference fold by fold and get a standard error, instead of
        # comparing two point estimates and hoping the gap is real. Without
        # this, a gain of +0.0004 against fold noise of 0.005 reads as a
        # finding when it is nothing at all.
        if previous is None:
            gain, gain_se, t_stat = result.r2_mean, float("nan"), float("nan")
        else:
            paired = np.array(result.fold_scores) - np.array(previous.fold_scores)
            gain = float(paired.mean())
            gain_se = float(paired.std(ddof=1) / np.sqrt(len(paired))) if len(paired) > 1 else float("nan")
            t_stat = gain / gain_se if gain_se and gain_se > 0 else float("nan")

        rows.append(
            {
                "model": result.name,
                "note": spec.note,
                "n_features": len(spec.features),
                "cv_r2": result.r2_mean,
                "cv_r2_sd": result.r2_std,
                "incremental_r2": gain,
                "incremental_se": gain_se,
                "incremental_t": t_stat,
            }
        )
        log.info(
            "%-12s cv R2 = %+.5f | gain %+.5f +/- %.5f (t = %s)",
            result.name,
            result.r2_mean,
            gain,
            gain_se if gain_se == gain_se else 0.0,
            f"{t_stat:+.2f}" if t_stat == t_stat else "n/a",
        )
        previous = result
    return pd.DataFrame(rows)


def permutation_ranking(
    df: pd.DataFrame,
    features: list[str],
    *,
    target: str = "run_value_pitcher",
    n_repeats: int = 5,
    sample: int | None = 400_000,
) -> pd.DataFrame:
    """Rank features by how much held-out skill dies when they are shuffled.

    Permutation importance on a *held-out* split, not the training data --
    training-set importance rewards features the model overfit to.
    """
    x, y, groups = _prepare(df, features, target)

    if sample is not None and len(y) > sample:
        rng = np.random.default_rng(RANDOM_SEED)
        pick = rng.choice(len(y), size=sample, replace=False)
        x, y, groups = x[pick], y[pick], groups[pick]

    train_idx, test_idx = next(GroupKFold(n_splits=5).split(x, y, groups))
    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.06,
        max_depth=6,
        min_samples_leaf=200,
        l2_regularization=1.0,
        random_state=RANDOM_SEED,
    )
    model.fit(x[train_idx], y[train_idx])

    imp = permutation_importance(
        model, x[test_idx], y[test_idx], n_repeats=n_repeats, random_state=RANDOM_SEED, n_jobs=-1
    )
    return (
        pd.DataFrame(
            {"feature": features, "importance": imp.importances_mean, "sd": imp.importances_std}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def standardized_coefficients(
    df: pd.DataFrame,
    features: list[str],
    *,
    target: str = "run_value_pitcher",
) -> pd.DataFrame:
    """Ridge coefficients on standardized inputs, for a readable effect size.

    The boosted model is the better predictor; this exists so the result can be
    stated in a sentence -- "one standard deviation more velocity is worth X
    runs per pitch, one standard deviation more spin is worth Y".
    """
    x, y, _ = _prepare(df, features, target)
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=RANDOM_SEED))
    pipe.fit(x, y)
    coefs = pipe.named_steps["ridge"].coef_
    return (
        pd.DataFrame({"feature": features, "coef_per_sd": coefs})
        .assign(abs_coef=lambda d: d["coef_per_sd"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
        .reset_index(drop=True)
    )
