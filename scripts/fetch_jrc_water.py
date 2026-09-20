"""Phase H9 -- national JRC Global Surface Water "occurrence" layer for
Bangladesh, from the Planetary Computer's `jrc-gsw` collection (free, no
account).

occurrence = % of valid observations (1984-2020) in which a pixel was water,
0-100. **255 is nodata** (never observed), not a value: averaging when
decimating mixes 255 into real pixels and produces impossible values above
100, which is a real bug hit while writing this. The layer is read with
nearest-neighbour sampling and 255 declared as nodata, so zonal statistics
downstream skip unobserved pixels instead of averaging them in.

Output is ~0.001 deg (~110 m) -- the native 30 m layer is 4x finer and far
more than a division-level (n=8) model needs.

Usage:
    python scripts/fetch_jrc_water.py
"""

from __future__ import annotations

from pathlib import Path

import planetary_computer
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
BANGLADESH_BBOX = (88.0, 20.5, 92.7, 26.7)  # min_lon, min_lat, max_lon, max_lat
RESOLUTION_DEG = 0.001
NODATA = 255
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "bgd_jrc_water_occurrence.tif"


def fetch_occurrence(bbox: tuple[float, float, float, float] = BANGLADESH_BBOX, out_path: Path = OUT_PATH) -> Path:
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    items = list(catalog.search(collections=["jrc-gsw"], bbox=list(bbox)).items())
    if not items:
        raise RuntimeError("no jrc-gsw tiles found for the bbox")

    sources = [rasterio.open(item.assets["occurrence"].href) for item in items]
    try:
        mosaic, transform = merge(
            sources, bounds=bbox, res=RESOLUTION_DEG, nodata=NODATA, resampling=Resampling.nearest
        )
    finally:
        for src in sources:
            src.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w", driver="GTiff", height=mosaic.shape[1], width=mosaic.shape[2], count=1,
        dtype=mosaic.dtype, crs="EPSG:4326", transform=transform, nodata=NODATA, compress="deflate",
    ) as dst:
        dst.write(mosaic)
    return out_path


def main() -> None:
    path = fetch_occurrence()
    with rasterio.open(path) as src:
        data = src.read(1)
    valid = data[data != NODATA]
    print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB), shape {data.shape}")
    print(f"valid pixels {valid.size:,} ({valid.size / data.size:.1%}), value range {valid.min()}-{valid.max()}, "
          f"pixels >0%: {(valid > 0).mean():.1%}")


if __name__ == "__main__":
    main()
