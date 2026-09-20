"""Phase H10 -- daily rainfall / temperature / humidity at each division's
population-weighted centroid, from NASA POWER (free, no key). Snapshotted to a
dated CSV: POWER revises recent days, so an unrecorded fetch is not
reproducible. Starts in early October 2025 so that even the first 2026 weeks
have several weeks of weather history to lag against.

Needs data/raw/bgd_adm1.geojson (fetch_boundaries.py --level ADM1) and the
WorldPop raster (fetch_population.py).

Usage:
    python scripts/fetch_weather_power.py
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.weather import parse_power_response, population_weighted_centroid  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
START = "20251006"  # a Monday
ADM1 = ROOT / "data" / "raw" / "bgd_adm1.geojson"
POPULATION = ROOT / "data" / "raw" / "bgd_worldpop_2020_constrained.tif"
SNAPSHOT_DIR = ROOT / "data" / "snapshots"


def fetch_power_daily(lon: float, lat: float, start: str, end: str, timeout: int = 90) -> pd.DataFrame:
    query = urllib.parse.urlencode(
        {"parameters": "PRECTOTCORR,T2M,RH2M", "community": "AG", "longitude": f"{lon:.4f}",
         "latitude": f"{lat:.4f}", "start": start, "end": end, "format": "JSON"}
    )
    with urllib.request.urlopen(f"{POWER_URL}?{query}", timeout=timeout) as response:
        return parse_power_response(json.load(response))


def main() -> None:
    today = date.today()
    divisions = gpd.read_file(ADM1)[["shapeName", "geometry"]]
    frames = []
    for name, geometry in zip(divisions["shapeName"], divisions.geometry):
        lon, lat = population_weighted_centroid(str(POPULATION), geometry)
        daily = fetch_power_daily(lon, lat, START, today.strftime("%Y%m%d"))
        daily.insert(0, "division", name)
        daily["lon"], daily["lat"] = lon, lat
        frames.append(daily.reset_index())
        print(f"{name:<11} centroid ({lon:.3f}, {lat:.3f}), {len(daily)} days, "
              f"{int(daily['rain_mm'].isna().sum())} missing rain days")
    out = pd.concat(frames, ignore_index=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SNAPSHOT_DIR / f"power_daily_divisions_{today.isoformat()}.csv"
    out.to_csv(out_path, index=False)
    print(f"saved {out_path} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
