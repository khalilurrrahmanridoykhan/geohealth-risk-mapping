"""Downloads Bangladesh administrative boundaries from geoBoundaries
(https://www.geoboundaries.org) -- free, public, CC BY 3.0 IGO. ADM2 is
district level (64 units); this is the smallest admin unit geoBoundaries
publishes for Bangladesh (no ward/union-level release), so district is the
zonal-stats unit for the Phase H0 smoke tile. Finer units (ward/union) come
from a different source in a later phase if needed.

Usage:
    python scripts/fetch_boundaries.py
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

API_URL = "https://www.geoboundaries.org/api/current/gbOpen/BGD/ADM2/"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "bgd_adm2.geojson"


def fetch_boundary_metadata(api_url: str = API_URL) -> dict:
    with urllib.request.urlopen(api_url, timeout=30) as response:
        return json.load(response)


def download_geojson(download_url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(download_url, out_path)


def main() -> None:
    meta = fetch_boundary_metadata()
    print(f"{meta['boundaryName']} -- {meta['admUnitCount']} units, "
          f"license: {meta['boundaryLicense']}")
    download_geojson(meta["gjDownloadURL"], OUT_PATH)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
