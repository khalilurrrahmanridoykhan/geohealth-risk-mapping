"""Downloads Bangladesh administrative boundaries from geoBoundaries
(https://www.geoboundaries.org) -- free, public, CC BY 3.0 IGO. ADM2 is
district level (64 units); this is the smallest admin unit geoBoundaries
publishes for Bangladesh (no ward/union-level release), so district is the
zonal-stats unit for the Phase H0 smoke tile. Finer units (ward/union) come
from a different source in a later phase if needed. ADM1 (8 divisions) is
what Phase H9 needs: the DGHS dengue dashboard's finest geography is
division level.

Usage:
    python scripts/fetch_boundaries.py            # ADM2, districts (default)
    python scripts/fetch_boundaries.py --level ADM1   # divisions
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

API_URL = "https://www.geoboundaries.org/api/current/gbOpen/BGD/{level}/"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def fetch_boundary_metadata(api_url: str) -> dict:
    with urllib.request.urlopen(api_url, timeout=30) as response:
        return json.load(response)


def download_geojson(download_url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(download_url, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level", choices=["ADM1", "ADM2"], default="ADM2")
    level = parser.parse_args().level
    out_path = OUT_DIR / f"bgd_{level.lower()}.geojson"

    meta = fetch_boundary_metadata(API_URL.format(level=level))
    print(f"{meta['boundaryName']} -- {meta['admUnitCount']} units, "
          f"license: {meta['boundaryLicense']}")
    download_geojson(meta["gjDownloadURL"], out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
