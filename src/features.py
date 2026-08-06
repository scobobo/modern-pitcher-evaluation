"""Turn raw Statcast rows into modelling features and outcome variables.

Three ideas drive this module:

1. Raw spin rate is not a clean treatment. Within a pitcher, spin and velocity
   rise and fall together (harder throws spin more), so a model given both will
   split credit between them arbitrarily. We therefore also build a spin
   measure that is residualised on velocity, which is the only version that can
   answer "does spin matter *beyond* velocity".

2. Spin only matters through the movement it produces. Gyroscopic spin moves
   the ball not at all. Public Statcast does not publish true spin efficiency
   (Hawk-Eye's measured axis is not in the feed -- `spin_axis` is inferred from
   observed movement), so we use observed movement and approach angle as the
   causal channel and treat spin rate as the upstream proxy it really is.

3. The outcome has to be a run-value outcome, not a whiff outcome alone. A
   pitch that misses bats but gets hit hard when it does not is not a good
   pitch.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import (
    MODELLED_PITCH_TYPES,
    PLATE_FRONT_Y_FT,
    TRACKING_START_Y_FT,
)

log = logging.getLogger(__name__)

SWING_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "hit_into_play",
    "foul_bunt",
    "missed_bunt",
}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "swinging_pitch", "missed_bunt"}


def compute_approach_angles(df: pd.DataFrame) -> pd.DataFrame:
    """Add vertical and horizontal approach angle at the front of the plate.

    Statcast reports velocity and acceleration at y = 50 ft. Ball flight is
    well approximated as constant acceleration over the remaining distance, so
    we solve the quadratic for the time the ball reaches the plate and take the
    velocity direction there.

    Vertical approach angle (VAA) is the single best-understood reason a
    high-spin four-seamer plays well: backspin keeps the ball on a flatter
    descent than the hitter's swing plane expects. It is the mechanism that
    "spin rate" is usually standing in for.
    """
    out = df.copy()

    y0 = TRACKING_START_Y_FT
    yf = PLATE_FRONT_Y_FT
    vy0, ay = out["vy0"], out["ay"]

    # y(t) = y0 + vy0*t + 0.5*ay*t^2  ->  solve for y(t) = yf.
    # vy0 is negative (ball travels toward the plate), so the physical root is
    # the one that subtracts the discriminant.
    disc = vy0**2 - 2.0 * ay * (y0 - yf)
    disc = disc.where(disc >= 0)
    t_plate = (-vy0 - np.sqrt(disc)) / ay

    vz_f = out["vz0"] + out["az"] * t_plate
    vx_f = out["vx0"] + out["ax"] * t_plate
    vy_f = vy0 + ay * t_plate

    out["vaa_deg"] = np.degrees(np.arctan2(vz_f, np.abs(vy_f)))
    out["haa_deg"] = np.degrees(np.arctan2(vx_f, np.abs(vy_f)))
    out["flight_time_s"] = t_plate
    return out


def compute_pitch_shape(df: pd.DataFrame) -> pd.DataFrame:
    """Add movement, Bauer units, and handedness-neutral horizontal features."""
    out = df.copy()

    # Statcast reports pfx in feet from the catcher's view. Mirror lefties so
    # "positive horizontal break" means arm-side for every pitcher; otherwise
    # the two handedness groups cancel each other out in any pooled model.
    hand_sign = np.where(out["p_throws"].eq("R"), 1.0, -1.0)
    out["pfx_x_armside"] = out["pfx_x"] * hand_sign
    out["release_pos_x_armside"] = out["release_pos_x"] * hand_sign
    out["haa_armside"] = out["haa_deg"] * hand_sign

    out["ivb_in"] = out["pfx_z"] * 12.0  # induced vertical break, inches
    out["hb_in"] = out["pfx_x_armside"] * 12.0
    out["movement_total_in"] = np.hypot(out["ivb_in"], out["hb_in"])

    # Bauer units: spin normalised by velocity. The classic "is this pitcher
    # spinning more than his velocity would predict" heuristic. It is a crude
    # ratio -- the residualised version below is the defensible one -- but it
    # is what the industry conversation uses, so we report both.
    out["bauer_units"] = out["release_spin_rate"] / out["release_speed"]

    # Perceived velocity: extension buys effective speed. A 95 mph fastball
    # released 7.0 ft out plays faster than one released 6.0 ft out, which is
    # itself an argument that raw velocity is the wrong variable -- not that
    # velocity stopped mattering.
    league_ext = out["release_extension"].median()
    out["perceived_velo"] = out["release_speed"] * (
        (TRACKING_START_Y_FT + out["release_extension"] - league_ext) / TRACKING_START_Y_FT
    ).clip(0.9, 1.1)

    return out


def residualise_spin_on_velocity(
    df: pd.DataFrame, *, by: tuple[str, ...] = ("game_year", "pitch_type")
) -> pd.DataFrame:
    """Add spin residualised on velocity within each season and pitch type.

    This is the variable that answers the actual research question. A positive
    coefficient on `spin_resid` means: holding velocity fixed, extra spin has
    independent value. If spin only looked valuable because hard throwers spin
    the ball more, this term goes to zero.
    """
    # Statcast pulls are concatenated day by day, so the incoming index has
    # duplicate labels. Work positionally and never touch label-based .loc.
    out = df.reset_index(drop=True).copy()

    speed = out["release_speed"].to_numpy(dtype=float)
    spin = out["release_spin_rate"].to_numpy(dtype=float)
    resid = np.full(len(out), np.nan)

    for key, pos in out.groupby(list(by), observed=True).indices.items():
        x, y = speed[pos], spin[pos]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 100:
            log.debug("group %s has only %d usable pitches, leaving spin_resid null", key, int(ok.sum()))
            continue
        slope, intercept = np.polyfit(x[ok], y[ok], 1)
        resid[pos[ok]] = y[ok] - (slope * x[ok] + intercept)

    out["spin_resid"] = resid
    return out


def add_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the outcome variables the models predict.

    `delta_run_exp` is Statcast's change in run expectancy attributable to the
    pitch, signed from the offense's perspective. We flip it so that positive
    means "good for the pitcher", which keeps every coefficient sign readable
    in one direction.
    """
    out = df.copy()

    out["is_swing"] = out["description"].isin(SWING_DESCRIPTIONS)
    out["is_whiff"] = out["description"].isin(WHIFF_DESCRIPTIONS)

    if "delta_run_exp" in out.columns:
        out["run_value_pitcher"] = -out["delta_run_exp"]
    else:
        out["run_value_pitcher"] = np.nan
        log.warning("delta_run_exp absent -- run-value models will be unavailable")

    # Count state matters enormously and is not a property of the pitch, so it
    # enters every model as a control rather than being ignored.
    out["count"] = out["balls"].astype("Int64").astype(str) + "-" + out["strikes"].astype("Int64").astype(str)
    out["is_platoon_advantage"] = out["p_throws"].eq(out["stand"])

    # Location quality has to be controlled for, or a model will credit spin
    # for command. Distance from the middle of the zone is a blunt but honest
    # control.
    zone_mid = (out["sz_top"] + out["sz_bot"]) / 2.0
    out["dist_from_zone_center"] = np.hypot(out["plate_x"], out["plate_z"] - zone_mid)

    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that cannot support the analysis, and log what was lost."""
    before = len(df)
    out = df[df["pitch_type"].isin(MODELLED_PITCH_TYPES)].copy()

    required = [
        "release_speed",
        "release_spin_rate",
        "pfx_x",
        "pfx_z",
        "plate_x",
        "plate_z",
        "vy0",
        "ay",
    ]
    out = out.dropna(subset=required)

    # Tracking glitches: physically impossible readings that would otherwise
    # dominate a squared-error loss.
    out = out[out["release_speed"].between(60, 108)]
    out = out[out["release_spin_rate"].between(500, 3600)]

    log.info("cleaning kept %s of %s pitches (%.1f%%)", f"{len(out):,}", f"{before:,}", 100 * len(out) / before)
    return out.reset_index(drop=True)


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature pipeline."""
    out = clean(df)
    out = compute_approach_angles(out)
    out = compute_pitch_shape(out)
    out = residualise_spin_on_velocity(out)
    out = add_outcomes(out)
    return out


# Feature blocks. The nested-model comparison in model.py adds these one block
# at a time, so each block's incremental predictive power is attributable.
CONTROL_FEATURES = [
    "dist_from_zone_center",
    "plate_x",
    "plate_z",
    "is_platoon_advantage",
    "strikes",
    "balls",
]
VELOCITY_FEATURES = ["release_speed", "release_extension"]
SPIN_RAW_FEATURES = ["release_spin_rate"]
SPIN_RESID_FEATURES = ["spin_resid"]
SHAPE_FEATURES = ["ivb_in", "hb_in", "vaa_deg", "haa_armside", "release_pos_z", "release_pos_x_armside"]
