"""Phase H5 -- downloads real health-facility points from OpenStreetMap
(via the Overpass API) for an AOI -- free, public, no account needed.
Used to identify which facilities fall inside a mapped flood extent.

Usage:
    python scripts/fetch_health_facilities.py
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
HEALTH_AMENITIES = ["hospital", "clinic", "doctors", "pharmacy"]
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "health_facilities_sunamganj.geojson"


def build_overpass_query(bbox: tuple[float, float, float, float], amenities: list[str]) -> str:
    """bbox: (min_lon, min_lat, max_lon, max_lat). Overpass itself wants
    (south, west, north, east) -- easy to swap by hand, so this function
    exists specifically to get that order right in one tested place."""
    min_lon, min_lat, max_lon, max_lat = bbox
    amenity_pattern = "|".join(amenities)
    overpass_bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"
    return (
        f'[out:json][timeout:60];'
        f'(node["amenity"~"{amenity_pattern}"]({overpass_bbox});'
        f'way["amenity"~"{amenity_pattern}"]({overpass_bbox}););'
        f"out center;"
    )


def overpass_elements_to_geojson(elements: list[dict]) -> dict:
    """Overpass returns nodes (lat/lon directly) and ways (a 'center'
    point instead) -- both turned into the same Point-geometry GeoJSON
    Feature shape here."""
    features = []
    for el in elements:
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        if lat is None or lon is None:
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "osm_id": el.get("id"),
                "osm_type": el.get("type"),
                "amenity": el.get("tags", {}).get("amenity"),
                "name": el.get("tags", {}).get("name"),
            },
        })
    return {"type": "FeatureCollection", "features": features}


def fetch_health_facilities(bbox: tuple[float, float, float, float], retries: int = 3) -> dict:
    query = build_overpass_query(bbox, HEALTH_AMENITIES)
    data = urllib.parse.urlencode({"data": query}).encode()
    # Overpass returns 406 Not Acceptable without a real User-Agent -- it
    # blocks the default Python urllib one to discourage anonymous scraping.
    request = urllib.request.Request(
        OVERPASS_URL, data=data,
        headers={"User-Agent": "geohealth-risk-mapping/0.1 (github.com/khalilurrrahmanridoykhan/geohealth-risk-mapping)"},
    )
    # Real failure seen running this from a cloud GPU notebook (not from a home
    # connection): a transient 504 Gateway Timeout from Overpass's public instance
    # under load. A plain retry with backoff is the honest fix -- no second mirror
    # endpoint is used here because none could actually be verified reachable.
    last_error: urllib.error.HTTPError | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                result = json.load(response)
            return overpass_elements_to_geojson(result["elements"])
        except urllib.error.HTTPError as e:
            last_error = e
            if attempt < retries - 1:
                wait = 10 * (attempt + 1)
                print(f"Overpass request failed ({e.code} {e.reason}), retrying in {wait}s...")
                time.sleep(wait)
    raise last_error


def main() -> None:
    bbox = (91.15, 24.5, 91.5, 24.8)  # Sunamganj/Dirai flood AOI, same as notebooks/06
    geojson = fetch_health_facilities(bbox)
    print(f"{len(geojson['features'])} health facilities found")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(geojson, f)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
