"""Phase H6 -- real very-high-resolution drone imagery from OpenAerialMap
(free, public, no account) and matching real building-footprint labels
from Microsoft Building Footprints (via Planetary Computer), for one real
Dhaka neighborhood (Rupnagar Khal, a canal-side settlement).

Real pitfall hit and fixed: OpenAerialMap's own catalog metadata points at
an 's3://oin-hotosm/...' URL that returns 403 Forbidden -- traced the real
redirect chain from the tile service instead of assuming the catalog URL
is correct, and found the actual public asset lives in a *different*
bucket, 'oin-hotosm-temp'.

Usage:
    python scripts/fetch_drone_imagery.py
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import adlfs
import geopandas as gpd
import numpy as np
import planetary_computer
import pystac_client
import rasterio
from planetary_computer import sas
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from shapely.geometry import box

OAM_API = "https://api.openaerialmap.org/meta"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def oam_search(bbox: tuple[float, float, float, float], limit: int = 50) -> list[dict]:
    """bbox: (min_lon, min_lat, max_lon, max_lat) -- OpenAerialMap's own API
    takes bbox in this same order, unlike Overpass."""
    url = f"{OAM_API}?bbox={','.join(str(b) for b in bbox)}&limit={limit}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)["results"]


def pick_scene_by_title_substring(results: list[dict], substring: str) -> dict:
    for r in results:
        if substring.lower() in (r.get("title") or "").lower():
            return r
    raise ValueError(f"no OpenAerialMap scene with {substring!r} in its title among {len(results)} results")


def real_asset_url(scene_meta: dict) -> str:
    """OpenAerialMap's metadata 'url' field ('s3://oin-hotosm/...') 404s/403s
    directly -- confirmed by opening it, not assumed. The real, publicly
    readable asset lives in the 'oin-hotosm-temp' bucket instead (found by
    tracing the tile service's actual redirect chain), at the same
    path/filename as the thumbnail but with a .tif extension."""
    thumbnail = scene_meta["properties"]["thumbnail"]
    if not thumbnail.endswith(".png"):
        raise ValueError(f"expected a .png thumbnail url, got {thumbnail!r}")
    return thumbnail[: -len(".png")] + ".tif"


def fetch_crop(
    asset_url: str, center_xy: tuple[float, float], half_size_m: float, target_resolution_m: float,
) -> tuple[np.ndarray, rasterio.Affine, str]:
    """A real windowed, decimated read (using the COG's own overview
    levels) centered on center_xy (in the asset's own CRS -- typically a
    local UTM zone), half_size_m in each direction. Returns (bands, HWC=3,
    transform, crs)."""
    cx, cy = center_xy
    with rasterio.open(asset_url) as src:
        window = from_bounds(cx - half_size_m, cy - half_size_m, cx + half_size_m, cy + half_size_m, transform=src.transform)
        native_res = src.transform.a
        out_h = max(1, int(window.height * native_res / target_resolution_m))
        out_w = max(1, int(window.width * native_res / target_resolution_m))
        data = src.read([1, 2, 3], window=window, out_shape=(3, out_h, out_w), resampling=Resampling.average)
        out_transform = src.window_transform(window) * rasterio.Affine.scale(window.width / out_w, window.height / out_h)
        crs = src.crs
    return data, out_transform, crs


def fetch_building_footprints(bbox_lonlat: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """Real Microsoft Building Footprints for a bbox, via Planetary
    Computer's ms-buildings collection. Real pitfall: these are stored as
    quadkey-partitioned GeoParquet on Azure Blob Storage behind an abfs://
    URL that pystac_client's normal sign_inplace modifier doesn't sign --
    needs planetary_computer.sas.get_token() + adlfs directly instead.
    bbox-pushdown filtering isn't supported for this file's Parquet
    metadata either, so the whole quadkey partition is read and clipped
    with geopandas afterward.
    """
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(collections=["ms-buildings"], bbox=bbox_lonlat)
    items = list(search.items())
    quadkey_items = [i for i in items if "quadkey" in i.assets["data"].href]
    if not quadkey_items:
        raise RuntimeError(f"no quadkey-partitioned ms-buildings item found for bbox={bbox_lonlat}")
    item = quadkey_items[0]

    token = sas.get_token("bingmlbuildings", "footprints")
    fs = adlfs.AzureBlobFileSystem(account_name="bingmlbuildings", sas_token=token.token)
    path = "footprints/" + item.assets["data"].href.replace("abfs://footprints/", "")
    gdf = gpd.read_parquet(path, filesystem=fs)

    aoi = box(*bbox_lonlat)
    return gdf[gdf.intersects(aoi)].reset_index(drop=True)


def main() -> None:
    results = oam_search(bbox=(90.0, 23.0, 91.0, 24.0), limit=50)
    scene = pick_scene_by_title_substring(results, "Rupnagar_Khal_3")
    asset_url = real_asset_url(scene)
    print(f"using scene: {scene['title']} ({scene['gsd']:.4f} m/px native)")

    with rasterio.open(asset_url) as src:
        center = ((src.bounds.left + src.bounds.right) / 2, (src.bounds.top + src.bounds.bottom) / 2)

    data, transform, crs = fetch_crop(asset_url, center, half_size_m=300, target_resolution_m=0.10)
    print(f"crop: {data.shape}, crs={crs}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        OUT_DIR / "dncc_rupnagar_khal3_crop.tif", "w", driver="GTiff",
        height=data.shape[1], width=data.shape[2], count=3, dtype=data.dtype,
        crs=crs, transform=transform,
    ) as dst:
        dst.write(data)
    print(f"saved {OUT_DIR / 'dncc_rupnagar_khal3_crop.tif'}")

    from pyproj import Transformer
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon1, lat1 = transformer.transform(transform.c, transform.f)
    lon2, lat2 = transformer.transform(transform.c + data.shape[2] * transform.a, transform.f + data.shape[1] * transform.e)
    bbox_lonlat = (min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2))

    buildings = fetch_building_footprints(bbox_lonlat)
    print(f"{len(buildings)} real buildings found in the crop's extent")
    buildings.to_file(OUT_DIR / "dncc_rupnagar_khal3_buildings.geojson", driver="GeoJSON")
    print(f"saved {OUT_DIR / 'dncc_rupnagar_khal3_buildings.geojson'}")


if __name__ == "__main__":
    main()
