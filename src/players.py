"""Pitcher biographical data: birth date and throwing hand.

Statcast carries neither, and pybaseball's Chadwick register has identifiers
but no birth dates. MLB's public StatsAPI has both and accepts batched person
IDs, so one pass covers every pitcher in the sample. The result is cached to
parquet because it changes about once per career.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from datetime import date

import pandas as pd

from config import DATA_DIR

log = logging.getLogger(__name__)

CACHE = DATA_DIR / "pitcher_bio.parquet"
ENDPOINT = "https://statsapi.mlb.com/api/v1/people?personIds={ids}"
BATCH = 120

# Baseball's convention is age as of 30 June of the season, so a player who
# turns 30 in September is 29 for that year. Using 1 January instead would
# shift roughly half the population by a year.
AGE_REFERENCE = (6, 30)


def _fetch_batch(ids: list[int]) -> list[dict]:
    url = ENDPOINT.format(ids=",".join(str(i) for i in ids))
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8")).get("people", [])
        except Exception as exc:  # noqa: BLE001 - network hiccups are retryable
            log.warning("bio batch failed (attempt %d): %s", attempt, type(exc).__name__)
            time.sleep(3 * attempt)
    return []


def fetch_bios(pitcher_ids, *, force: bool = False) -> pd.DataFrame:
    """Return one row per pitcher: id, full name, birth date, throwing hand."""
    wanted = sorted({int(p) for p in pitcher_ids if pd.notna(p)})

    cached = pd.read_parquet(CACHE) if (CACHE.exists() and not force) else pd.DataFrame()
    known = set(cached["pitcher"].astype(int)) if len(cached) else set()
    missing = [p for p in wanted if p not in known]

    if missing:
        log.info("fetching biographical data for %d pitchers", len(missing))
        rows = []
        for i in range(0, len(missing), BATCH):
            for person in _fetch_batch(missing[i : i + BATCH]):
                rows.append(
                    {
                        "pitcher": int(person["id"]),
                        "full_name": person.get("fullName"),
                        "birth_date": person.get("birthDate"),
                        "throws": (person.get("pitchHand") or {}).get("code"),
                    }
                )
            log.info("  %d / %d", min(i + BATCH, len(missing)), len(missing))
        fresh = pd.DataFrame(rows)
        cached = pd.concat([cached, fresh], ignore_index=True) if len(cached) else fresh
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        cached.to_parquet(CACHE, index=False)

    return cached[cached["pitcher"].isin(wanted)].reset_index(drop=True)


def age_in_season(birth_date: str | None, season: int) -> float | None:
    """Age as of 30 June of the given season."""
    if not birth_date:
        return None
    try:
        y, m, d = (int(x) for x in str(birth_date)[:10].split("-"))
    except ValueError:
        return None
    ref = date(season, *AGE_REFERENCE)
    born = date(y, m, d)
    return ref.year - born.year - ((ref.month, ref.day) < (born.month, born.day))


def attach_age(table: pd.DataFrame, bios: pd.DataFrame) -> pd.DataFrame:
    """Join bios onto a pitcher-season table and compute age for each season."""
    out = table.merge(bios, on="pitcher", how="left")
    out["age"] = [
        age_in_season(bd, int(yr))
        for bd, yr in zip(out["birth_date"], out["game_year"])
    ]
    missing = out["age"].isna().mean()
    if missing:
        log.warning("age unavailable for %.1f%% of pitcher-seasons", 100 * missing)
    return out
