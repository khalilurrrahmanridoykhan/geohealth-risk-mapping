"""Phase H4 -- cleaning up raw vectorized polygons: fixing invalid
geometries, dropping slivers, and simplifying, in the order that actually
matters (fix before measuring area, simplify last so tolerance is applied
to the final shape, not one about to be dropped).

Pure geopandas operations, unit-tested with small synthetic geometries in
tests/test_vectorize.py -- no raster I/O here (that's src/tiling.py's
rasterio.features.shapes call, already exercised for real in
notebooks/01_fundamentals.ipynb).
"""

from __future__ import annotations

import geopandas as gpd


def fix_invalid_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Self-intersecting ("bowtie") polygons are a real, common output of
    rasterio.features.shapes on noisy real data -- buffer(0) is the
    standard trick to repair them without changing valid geometries."""
    result = gdf.copy()
    invalid = ~result.geometry.is_valid
    result.loc[invalid, "geometry"] = result.loc[invalid, "geometry"].buffer(0)
    return result


def drop_small_polygons(gdf: gpd.GeoDataFrame, min_area: float) -> gpd.GeoDataFrame:
    """Drops polygons below min_area (in the GeoDataFrame's CRS units --
    square meters for a projected UTM CRS). Real vectorized rasters produce
    plenty of single/few-pixel noise polygons alongside real features (see
    notebooks/01_fundamentals.ipynb's water polygons); this is not optional
    for real imagery."""
    return gdf[gdf.geometry.area >= min_area].reset_index(drop=True)


def simplify_polygons(gdf: gpd.GeoDataFrame, tolerance: float) -> gpd.GeoDataFrame:
    """Simplifies with topology preservation (no gaps/overlaps introduced
    between adjacent polygons) -- applied last, after cleanup, so the
    tolerance describes the final shape rather than one about to be
    dropped as a sliver."""
    result = gdf.copy()
    result["geometry"] = result.geometry.simplify(tolerance, preserve_topology=True)
    return result


def clean_polygons(gdf: gpd.GeoDataFrame, min_area: float, simplify_tolerance: float) -> gpd.GeoDataFrame:
    """The full cleanup pipeline in the order that matters: fix invalid
    geometries first (an invalid polygon's .area can be meaningless),
    then drop slivers, then simplify last."""
    return simplify_polygons(drop_small_polygons(fix_invalid_geometries(gdf), min_area), simplify_tolerance)
