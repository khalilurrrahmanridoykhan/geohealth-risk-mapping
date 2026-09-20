"""Phase H10 -- weekly weather per division from NASA POWER daily point data.

NASA POWER (https://power.larc.nasa.gov) serves daily, gridded, satellite- and
reanalysis-based meteorology (MERRA-2 with corrected precipitation) for any
point, free and without a key. One point per division is a crude summary of
areas of 10,000-30,000 km2, so the point is the *population-weighted* centroid
(where people, and therefore admissions, actually are) rather than the
geometric middle -- and that remains a MAUP-style limitation, stated in the
notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rasterio_mask

POWER_FILL_VALUE = -999.0
_PARAMETERS = {"PRECTOTCORR": "rain_mm", "T2M": "temp_c", "RH2M": "rh_pct"}


def population_weighted_centroid(raster_path: str, geometry) -> tuple[float, float]:
    """(lon, lat) of the population-weighted centre of `geometry` (EPSG:4326)
    from a population-count raster. Falls back to the geometric representative
    point if the polygon contains no population at all."""
    with rasterio.open(raster_path) as src:
        data, transform = rasterio_mask(src, [geometry], crop=True, filled=False)
        nodata = src.nodata
    counts = np.ma.filled(data[0].astype(float), 0.0)
    if nodata is not None:
        counts[counts == nodata] = 0.0
    counts[counts < 0] = 0.0
    total = counts.sum()
    if total <= 0:
        point = geometry.representative_point()
        return float(point.x), float(point.y)
    rows, cols = np.indices(counts.shape)
    xs, ys = rasterio.transform.xy(transform, rows.ravel(), cols.ravel(), offset="center")
    weights = counts.ravel() / total
    return float(np.dot(weights, xs)), float(np.dot(weights, ys))


def parse_power_response(payload: dict) -> pd.DataFrame:
    """POWER daily JSON -> DataFrame indexed by date with rain_mm, temp_c,
    rh_pct. POWER marks missing values with -999, which is turned into NaN
    (leaving it in would look like a -999 mm rainfall)."""
    parameters = payload["properties"]["parameter"]
    frame = pd.DataFrame({new: pd.Series(parameters[old]) for old, new in _PARAMETERS.items()})
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d")
    frame.index.name = "date"
    return frame.where(frame != POWER_FILL_VALUE)


def to_iso_weeks(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily weather -> weekly (ISO weeks, Monday-Sunday): rainfall summed,
    temperature and humidity averaged. Only complete 7-day weeks are kept, so a
    partial trailing week can't masquerade as a dry one. Index = (iso_year, week)."""
    iso = daily.index.isocalendar()
    grouped = daily.groupby([iso["year"].values, iso["week"].values])
    weekly = pd.DataFrame(
        {
            "rain_mm": grouped["rain_mm"].sum(min_count=7),
            "temp_c": grouped["temp_c"].mean(),
            "rh_pct": grouped["rh_pct"].mean(),
            "n_days": grouped["rain_mm"].size(),
        }
    )
    weekly = weekly[weekly["n_days"] == 7].drop(columns="n_days")
    weekly.index.names = ["iso_year", "week"]
    return weekly
