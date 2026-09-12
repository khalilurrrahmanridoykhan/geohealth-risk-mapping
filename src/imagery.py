"""Phase H1 -- a reusable get_imagery(aoi, date_range, sensor) that returns an
analysis-ready, cloud-masked composite from either Microsoft Planetary
Computer (free, no account -- the default and the only backend actually
verified against live data in this repo so far) or Google Earth Engine (real
API calls, but requires a one-time interactive `earthengine authenticate`
login this environment cannot perform -- see the H1 notebook for what
happens when it isn't set up).

The cloud-masking math (`scl_cloud_mask`, `composite_cloud_masked`) is pure
numpy and unit-tested with synthetic arrays in tests/test_imagery.py. The
STAC search / stackstac loading is I/O and is instead verified by actually
running notebooks/02_cloud_data_access.ipynb against live data.
"""

from __future__ import annotations

import re
import warnings

import numpy as np

# Sentinel-2 L2A Scene Classification Layer (SCL) class codes to KEEP in a
# cloud-free composite. Excluded: 0 no-data, 1 saturated/defective,
# 2 dark-area pixel, 3 cloud shadow, 8/9 cloud medium/high probability,
# 10 thin cirrus. Kept: 4 vegetation, 5 bare soil, 6 water,
# 7 unclassified/low-probability cloud, 11 snow/ice.
SENTINEL2_CLEAR_SCL_CLASSES = frozenset({4, 5, 6, 7, 11})


def scl_cloud_mask(scl: np.ndarray) -> np.ndarray:
    """True where a Sentinel-2 SCL pixel is clear enough to use in a
    composite, False where it's cloud/shadow/cirrus/nodata."""
    return np.isin(scl, list(SENTINEL2_CLEAR_SCL_CLASSES))


def always_cloudy_fraction(scl_stack: np.ndarray) -> float:
    """Fraction of pixels that have zero clear observations across the
    whole time stack -- i.e. the fraction of the composite that
    composite_cloud_masked() had to fall back to a cloud-contaminated plain
    median for. Real value found during Dhaka monsoon-season testing: ~0.49
    even with 46 scenes across 4 months -- some AOI/seasons have so few
    clear passes that no amount of compositing over the STAC search window
    fully removes cloud contamination. High fallback fraction is a reason to
    widen the date range, add more scenes, or switch to Sentinel-1 (H5) --
    not something to silently paper over."""
    keep = scl_cloud_mask(scl_stack)
    return float(np.all(~keep, axis=0).mean())


def composite_cloud_masked(band_stack: np.ndarray, scl_stack: np.ndarray) -> np.ndarray:
    """Per-pixel median across the time axis (axis 0), using only
    observations SCL marks as clear. A pixel that's cloudy in every single
    date range falls back to the plain (unmasked) median rather than
    returning NaN -- some real signal beats none for a pixel with zero clear
    observations in the requested window."""
    keep = scl_cloud_mask(scl_stack)
    masked = np.where(keep, band_stack, np.nan).astype("float32")
    always_cloudy = np.all(~keep, axis=0)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        # "All-NaN slice" is expected here -- it's exactly the always_cloudy
        # case, which the fallback below handles on purpose.
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        composite = np.nanmedian(masked, axis=0)
        if np.any(always_cloudy):
            fallback = np.nanmedian(band_stack.astype("float32"), axis=0)
            composite = np.where(always_cloudy, fallback, composite)
    return composite.astype("float32")


def utm_epsg_from_sentinel2_id(item_id: str) -> int:
    """Sentinel-2 item/tile ids encode their MGRS UTM zone
    (e.g. '..._T46QBM_...' -> zone 46). Bangladesh sits entirely in the
    northern hemisphere (zones 45N/46N), so this always adds to 32600 --
    a southern-hemisphere AOI would need 32700 instead."""
    match = re.search(r"_T(\d{2})[A-Z]{3}_", item_id)
    if not match:
        raise ValueError(f"could not find an MGRS tile zone in id {item_id!r}")
    return 32600 + int(match.group(1))


def get_imagery(
    aoi: tuple[float, float, float, float],
    date_range: str,
    sensor: str = "sentinel-2",
    source: str = "planetary_computer",
    cloud_cover_max: int = 60,
    resolution: int = 20,
    max_scenes: int = 10,
):
    """Returns (composite, meta) -- composite is an xarray.DataArray
    (band, y, x) in the scene's native UTM CRS; meta is a dict documenting
    what went into it (item ids used, date range, per-scene cloud cover).

    aoi: (min_lon, min_lat, max_lon, max_lat).
    date_range: STAC-style 'YYYY-MM-DD/YYYY-MM-DD'.
    sensor: 'sentinel-2' (cloud-masked optical composite). 'sentinel-1' is
        recognized but raises NotImplementedError -- deferred to Phase H5,
        which needs GCP-aware raster reading this function doesn't have.
    source: 'planetary_computer' (default, free, no account, verified
        against live data) or 'earth_engine' (real GEE API calls, but
        requires a one-time `earthengine authenticate` login).
    """
    # Validated before any network call, so an invalid sensor/source fails
    # fast and offline -- exercised directly by tests/test_imagery.py without
    # needing real network access.
    if sensor == "sentinel-1":
        raise NotImplementedError(
            "sensor='sentinel-1' needs GCP-aware raster reading that stackstac's "
            "plain .stack() doesn't provide (Planetary Computer's sentinel-1-grd "
            "assets carry GCPs in EPSG:4326 instead of a direct affine CRS -- "
            "confirmed by opening one directly, not assumed). Deferred to Phase "
            "H5 (SAR flood mapping), which implements this properly."
        )
    if sensor != "sentinel-2":
        raise ValueError(f"unknown sensor {sensor!r}, expected 'sentinel-2' (sentinel-1 deferred to H5)")

    if source == "planetary_computer":
        return _get_imagery_planetary_computer(aoi, date_range, sensor, cloud_cover_max, resolution, max_scenes)
    if source == "earth_engine":
        return _get_imagery_earth_engine(aoi, date_range, sensor, cloud_cover_max, resolution)
    raise ValueError(f"unknown source {source!r}, expected 'planetary_computer' or 'earth_engine'")


def _get_imagery_planetary_computer(aoi, date_range, sensor, cloud_cover_max, resolution, max_scenes):
    # sensor is always 'sentinel-2' here -- get_imagery() validates it before
    # dispatching to this backend.
    import planetary_computer
    import pystac_client
    import rioxarray  # noqa: F401 -- registers the .rio accessor
    import stackstac

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=aoi,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": cloud_cover_max}},
    )
    items = sorted(search.items(), key=lambda i: i.properties["eo:cloud_cover"])[:max_scenes]
    if not items:
        raise RuntimeError(
            f"no sentinel-2 scenes under {cloud_cover_max}% cloud cover found for "
            f"aoi={aoi}, date_range={date_range!r}"
        )
    epsg = utm_epsg_from_sentinel2_id(items[0].id)
    stack = stackstac.stack(
        items,
        assets=["B03", "B04", "B08", "B11", "SCL"],
        bounds_latlon=list(aoi),
        resolution=resolution,
        epsg=epsg,
    )
    arr = stack.compute()
    bands = arr.sel(band=["B03", "B04", "B08", "B11"]).values  # (time, band, y, x)
    scl = arr.sel(band="SCL").values  # (time, y, x)
    scl_per_band = np.broadcast_to(scl[:, None, :, :], bands.shape)
    composite = composite_cloud_masked(bands, scl_per_band)  # (band, y, x)

    meta = {
        "sensor": sensor,
        "source": "planetary_computer",
        "date_range": date_range,
        "n_scenes_used": len(items),
        "item_ids": [i.id for i in items],
        "cloud_cover_pct": [round(i.properties["eo:cloud_cover"], 2) for i in items],
        "epsg": epsg,
        "bands": ["B03_green", "B04_red", "B08_nir", "B11_swir"],
        # Fraction of pixels with zero clear observation across every scene
        # used -- these fell back to a cloud-contaminated plain median. High
        # values mean this composite is NOT reliable; widen date_range, raise
        # max_scenes, or use Sentinel-1 (H5) instead. Not swept under the rug.
        "always_cloudy_fraction": always_cloudy_fraction(scl),
    }
    composite_da = stack.sel(band=["B03", "B04", "B08", "B11"]).isel(time=0).copy(data=composite)
    return composite_da, meta


def _get_imagery_earth_engine(aoi, date_range, sensor, cloud_cover_max, resolution):
    """Real Earth Engine API calls -- correct against the documented ee
    Python API, but NOT verified end-to-end in this repo: it requires a
    one-time interactive `earthengine authenticate` login (opens a browser
    for Google OAuth) that this environment cannot perform on the user's
    behalf. Run that once locally, then this function should work; if it
    doesn't, that's a real gap to fix, not an assumption to trust blindly.
    """
    import ee

    ee.Initialize()

    start, end = date_range.split("/")
    region = ee.Geometry.Rectangle(list(aoi))

    if sensor == "sentinel-2":
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_cover_max))
        )

        def mask_clouds(image):
            scl = image.select("SCL")
            clear = scl.remap(list(SENTINEL2_CLEAR_SCL_CLASSES), [1] * len(SENTINEL2_CLEAR_SCL_CLASSES), 0)
            return image.updateMask(clear)

        composite = collection.map(mask_clouds).select(["B3", "B4", "B8", "B11"]).median().clip(region)
        n_scenes = collection.size().getInfo()
        meta = {
            "sensor": sensor,
            "source": "earth_engine",
            "date_range": date_range,
            "n_scenes_used": n_scenes,
            "bands": ["B3_green", "B4_red", "B8_nir", "B11_swir"],
        }
        return composite, meta

    raise NotImplementedError(f"earth_engine backend for sensor={sensor!r} not implemented yet")
