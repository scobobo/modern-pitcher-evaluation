"""Pull season-level Statcast pitch data and cache it locally.

Baseball Savant caps each query at 25k rows, so pybaseball chunks the request
by date internally. A full season is roughly 700k pitches and takes a while --
each season is cached as its own parquet file so an interrupted run resumes
instead of starting over.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from config import DATA_DIR, DEFAULT_SEASONS, SEASON_WINDOWS

log = logging.getLogger(__name__)

# Only the columns the study actually uses. Statcast returns ~90; keeping the
# subset cuts the cached footprint by roughly 5x.
KEEP_COLUMNS = [
    "game_date",
    "game_year",
    "game_pk",
    "pitcher",
    "batter",
    "player_name",
    "pitch_type",
    "release_speed",
    "release_spin_rate",
    "release_extension",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "spin_axis",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "p_throws",
    "stand",
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "description",
    "events",
    "type",
    "zone",
    "sz_top",
    "sz_bot",
    "delta_run_exp",
    "estimated_woba_using_speedangle",
    "launch_speed",
    "launch_angle",
]


def season_path(season: int) -> Path:
    return DATA_DIR / f"statcast_{season}.parquet"


CHUNK_DAYS = 14
MAX_ATTEMPTS = 4


def _fetch_window(start: str, end: str) -> pd.DataFrame:
    """Fetch one date window, retrying transient Savant failures.

    Savant intermittently answers with an HTML error page instead of CSV, which
    surfaces as a pandas ParserError. A single such response should not cost us
    an entire season, so we back off and retry rather than propagating.
    """
    from pybaseball import statcast

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            frame = statcast(start_dt=start, end_dt=end, verbose=False)
            if frame is not None and not frame.empty:
                return frame
            log.warning("window %s..%s returned no rows (attempt %d)", start, end, attempt)
        except Exception as exc:  # noqa: BLE001 - any parse/network failure is retryable here
            log.warning("window %s..%s failed (attempt %d): %s", start, end, attempt, type(exc).__name__)

        if attempt < MAX_ATTEMPTS:
            time.sleep(5 * attempt)

    log.error("window %s..%s permanently failed, skipping", start, end)
    return pd.DataFrame()


def fetch_season(season: int, *, force: bool = False) -> pd.DataFrame:
    """Fetch one regular season of pitch-level Statcast data, using the cache.

    The season is pulled in two-week windows so a transient failure costs one
    window instead of five months of downloading.
    """
    path = season_path(season)
    if path.exists() and not force:
        log.info("season %s: reading cache %s", season, path.name)
        return pd.read_parquet(path)

    start, end = SEASON_WINDOWS[season]
    log.info("season %s: downloading %s to %s (this takes several minutes)", season, start, end)

    windows: list[tuple[str, str]] = []
    cursor = pd.Timestamp(start)
    final = pd.Timestamp(end)
    while cursor <= final:
        stop = min(cursor + pd.Timedelta(days=CHUNK_DAYS - 1), final)
        windows.append((cursor.strftime("%Y-%m-%d"), stop.strftime("%Y-%m-%d")))
        cursor = stop + pd.Timedelta(days=1)

    parts = []
    for i, (w_start, w_end) in enumerate(windows, 1):
        part = _fetch_window(w_start, w_end)
        if not part.empty:
            parts.append(part)
        log.info("season %s: window %d/%d done (%s rows)", season, i, len(windows), f"{len(part):,}")

    if not parts:
        raise RuntimeError(f"Statcast returned no rows for {season}")

    raw = pd.concat(parts, ignore_index=True)

    # game_year is occasionally null in the raw feed; derive it from the date
    # so downstream season grouping never silently drops pitches.
    raw["game_date"] = pd.to_datetime(raw["game_date"])
    raw["game_year"] = raw["game_date"].dt.year

    available = [c for c in KEEP_COLUMNS if c in raw.columns]
    missing = sorted(set(KEEP_COLUMNS) - set(available))
    if missing:
        log.warning("season %s: columns absent from feed: %s", season, ", ".join(missing))

    df = raw[available].copy()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("season %s: cached %s pitches -> %s", season, f"{len(df):,}", path.name)
    return df


def load_seasons(seasons: tuple[int, ...] = DEFAULT_SEASONS, *, force: bool = False) -> pd.DataFrame:
    """Fetch (or load) several seasons and concatenate them."""
    frames = [fetch_season(s, force=force) for s in seasons]
    combined = pd.concat(frames, ignore_index=True)
    log.info("loaded %s pitches across %d seasons", f"{len(combined):,}", len(seasons))
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and cache Statcast seasons.")
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEASONS),
        help="Seasons to fetch (default: 2015-2025).",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if cached.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for season in args.seasons:
        fetch_season(season, force=args.force)


if __name__ == "__main__":
    main()
