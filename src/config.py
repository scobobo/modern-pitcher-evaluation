"""Shared configuration for the velocity-vs-spin study."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

# Statcast is complete and reliable from 2015 onward. Spin-rate measurement
# switched from Trackman to Hawk-Eye before the 2020 season, so spin values
# before and after that boundary are not perfectly comparable -- we fit
# per-season models rather than pooling across the discontinuity.
HAWKEYE_TRANSITION_SEASON = 2020

DEFAULT_SEASONS = tuple(range(2015, 2026))

# Regular season only. Spring training and the postseason have different
# selection effects (roster churn, leverage) that would bias season comparisons.
SEASON_WINDOWS: dict[int, tuple[str, str]] = {
    2015: ("2015-04-05", "2015-10-04"),
    2016: ("2016-04-03", "2016-10-02"),
    2017: ("2017-04-02", "2017-10-01"),
    2018: ("2018-03-29", "2018-10-01"),
    2019: ("2019-03-28", "2019-09-29"),
    2020: ("2020-07-23", "2020-09-27"),  # pandemic-shortened, 60 games
    2021: ("2021-04-01", "2021-10-03"),
    2022: ("2022-04-07", "2022-10-05"),
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-20", "2024-09-29"),
    2025: ("2025-03-18", "2025-09-28"),
}

# Pitch types worth modelling. Anything rarer than this is too sparse to fit
# per-season without the estimates becoming noise.
FASTBALLS = ("FF", "SI", "FC")
BREAKING = ("SL", "CU", "KC", "ST", "SV")
OFFSPEED = ("CH", "FS")
MODELLED_PITCH_TYPES = FASTBALLS + BREAKING + OFFSPEED

# The primary analysis runs on four-seamers. It is the pitch where the
# "spin matters more than velocity" claim is most often made, and where
# backspin-driven ride is the dominant movement mechanism.
PRIMARY_PITCH_TYPE = "FF"

# Physical constants for approach-angle geometry.
PLATE_FRONT_Y_FT = 17.0 / 12.0  # front edge of home plate, in feet from apex
TRACKING_START_Y_FT = 50.0  # Statcast reports v0/a0 at y = 50 ft

RANDOM_SEED = 17

# Minimum pitches for a pitcher-season to enter pitcher-level aggregations.
MIN_PITCHES_PER_PITCHER_SEASON = 250
