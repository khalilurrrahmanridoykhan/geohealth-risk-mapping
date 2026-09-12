"""Phase H4 -- the actual deliverable PLAN.md asks for: "one command turns
model output into a clean ward-level predictor table + a GeoPackage,
reproducibly." build_predictor_table() is that one command.

Tested end-to-end (including real file I/O to a temp GeoPackage) with tiny
synthetic rasters/boundaries in tests/test_pipeline.py -- fast and offline,
same style as every other src/ module. The real version, against real H3
model output and real district boundaries, is demonstrated in
notebooks/05_vectorize_zonal_stats.ipynb.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterstats
from affine import Affine
from rasterio.features import shapes
from shapely.geometry import shape

from src.vectorize import clean_polygons


def vectorize_mask(mask: np.ndarray, transform: Affine, crs, min_area: float, simplify_tolerance: float) -> gpd.GeoDataFrame:
    """Boolean raster -> clean polygons (the mask's True regions only)."""
    polygons = [shape(geom) for geom, value in shapes(mask.astype("uint8"), transform=transform) if value == 1]
    gdf = gpd.GeoDataFrame({"id": range(len(polygons))}, geometry=polygons, crs=crs)
    if len(gdf) == 0:
        return gdf
    return clean_polygons(gdf, min_area=min_area, simplify_tolerance=simplify_tolerance)


def zonal_predictor_table(
    predictor_rasters: dict[str, np.ndarray],
    transform: Affine,
    boundaries: gpd.GeoDataFrame,
    boundary_name_col: str,
) -> pd.DataFrame:
    """One row per boundary unit, one column per predictor raster's zonal
    mean -- the table every later phase's health-data join depends on."""
    table = pd.DataFrame({boundary_name_col: boundaries[boundary_name_col].values})
    for name, raster in predictor_rasters.items():
        stats = rasterstats.zonal_stats(boundaries, raster, affine=transform, stats=["mean"], nodata=np.nan)
        table[name] = [s["mean"] for s in stats]
    return table.set_index(boundary_name_col)


def build_predictor_table(
    predictor_rasters: dict[str, np.ndarray],
    transform: Affine,
    crs,
    boundaries: gpd.GeoDataFrame,
    boundary_name_col: str,
    binary_masks_to_vectorize: dict[str, np.ndarray],
    out_gpkg: str | Path,
    min_area: float = 100.0,
    simplify_tolerance: float = 5.0,
) -> tuple[pd.DataFrame, dict[str, gpd.GeoDataFrame]]:
    """The one-command pipeline: real model output (rasters) + real admin
    boundaries in -> a zonal predictor table + a GeoPackage (boundaries
    joined with the table, plus one layer per vectorized mask) out.

    Writing the same rasters/boundaries through this function twice
    produces the same GeoPackage and table -- reproducible, not a one-off
    notebook sequence of manual steps.
    """
    table = zonal_predictor_table(predictor_rasters, transform, boundaries, boundary_name_col)

    vectorized = {
        name: vectorize_mask(mask, transform, crs, min_area, simplify_tolerance)
        for name, mask in binary_masks_to_vectorize.items()
    }

    boundaries_with_table = boundaries.merge(table, left_on=boundary_name_col, right_index=True)
    out_gpkg = Path(out_gpkg)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    boundaries_with_table.to_file(out_gpkg, layer="boundaries_with_predictors", driver="GPKG")
    for name, gdf in vectorized.items():
        if len(gdf) > 0:
            gdf.to_file(out_gpkg, layer=name, driver="GPKG")

    return table, vectorized
