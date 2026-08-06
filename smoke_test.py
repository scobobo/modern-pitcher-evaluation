"""Fast end-to-end validation on a two-week slice, before committing to full seasons."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from pybaseball import statcast

import features as feat
from model import build_ladder, run_ladder, standardized_coefficients

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("smoke")

CACHE = Path(__file__).parent / "data" / "smoke_slice.parquet"

if CACHE.exists():
    raw = pd.read_parquet(CACHE)
    log.info("loaded cached slice: %s pitches", f"{len(raw):,}")
else:
    raw = statcast(start_dt="2025-05-01", end_dt="2025-05-14", verbose=False)
    raw["game_date"] = pd.to_datetime(raw["game_date"])
    raw["game_year"] = raw["game_date"].dt.year
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(CACHE, index=False)
    log.info("downloaded slice: %s pitches", f"{len(raw):,}")

data = feat.build(raw)
log.info("post-feature rows: %s", f"{len(data):,}")

print("\n--- sanity checks on derived physics ---")
ff = data[data["pitch_type"].eq("FF")]
print(f"four-seamers: {len(ff):,}")
print(f"VAA (deg):        mean {ff['vaa_deg'].mean():.2f}  sd {ff['vaa_deg'].std():.2f}")
print(f"IVB (in):         mean {ff['ivb_in'].mean():.2f}  sd {ff['ivb_in'].std():.2f}")
print(f"velocity (mph):   mean {ff['release_speed'].mean():.2f}  sd {ff['release_speed'].std():.2f}")
print(f"spin (rpm):       mean {ff['release_spin_rate'].mean():.0f}  sd {ff['release_spin_rate'].std():.0f}")
print(f"spin_resid (rpm): mean {ff['spin_resid'].mean():.2f}  sd {ff['spin_resid'].std():.0f}")
print(f"corr(velo, spin)        = {ff['release_speed'].corr(ff['release_spin_rate']):.3f}")
print(f"corr(velo, spin_resid)  = {ff['release_speed'].corr(ff['spin_resid']):.3f}  <- must be ~0")
print(f"run_value_pitcher: mean {ff['run_value_pitcher'].mean():.5f}  sd {ff['run_value_pitcher'].std():.4f}")
print(f"whiff rate on swings: {ff[ff['is_swing']]['is_whiff'].mean():.3f}")

print("\n--- nested ladder (small sample, indicative only) ---")
ladder = run_ladder(ff, build_ladder(), n_splits=4)
print(ladder.to_string(index=False))

print("\n--- standardized coefficients ---")
print(standardized_coefficients(ff, build_ladder()[-1].features).to_string(index=False))
