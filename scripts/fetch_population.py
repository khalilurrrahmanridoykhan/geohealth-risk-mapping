"""Phase H5 -- downloads WorldPop's constrained 2020 Bangladesh population
count raster (free, public, no account) -- building-footprint-constrained,
so population is allocated to where people actually live rather than
spread evenly across a grid cell.

Real pitfall hit while writing this: the server advertises
`Accept-Ranges: bytes` but GDAL's actual range-request probe fails
("Range downloading not supported by this server!"), so a windowed
/vsicurl/ read (the usual free-tier no-bulk-download approach used
elsewhere in this repo, e.g. Planetary Computer COGs) doesn't work here --
the whole ~14MB file has to be downloaded. A first attempt with a 60s curl
timeout also silently produced a truncated, corrupt-looking-but-openable
file (rasterio could read metadata but failed on actual pixel reads) --
fixed with a longer timeout and --retry.

Usage:
    python scripts/fetch_population.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

POPULATION_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/"
    "2020/BSGM/BGD/bgd_ppp_2020_constrained.tif"
)
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "bgd_worldpop_2020_constrained.tif"


def download_population_raster(url: str = POPULATION_URL, out_path: Path = OUT_PATH, timeout: int = 180) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout) as response, open(out_path, "wb") as f:
        f.write(response.read())
    return out_path


def main() -> None:
    path = download_population_raster()
    size_mb = path.stat().st_size / 1e6
    print(f"saved {path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
