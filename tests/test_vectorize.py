import geopandas as gpd
from shapely.geometry import Polygon

from src.vectorize import clean_polygons, drop_small_polygons, fix_invalid_geometries, simplify_polygons

# A self-intersecting "bowtie" polygon -- a real, common output of
# rasterio.features.shapes on noisy real data.
BOWTIE = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
SQUARE_10x10 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
SLIVER_1x1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


def make_gdf(geoms):
    return gpd.GeoDataFrame({"id": range(len(geoms))}, geometry=geoms, crs="EPSG:32646")


def test_fix_invalid_geometries_repairs_a_bowtie():
    gdf = make_gdf([BOWTIE])
    assert not gdf.geometry.iloc[0].is_valid
    fixed = fix_invalid_geometries(gdf)
    assert fixed.geometry.iloc[0].is_valid


def test_fix_invalid_geometries_leaves_valid_geometries_unchanged():
    gdf = make_gdf([SQUARE_10x10])
    fixed = fix_invalid_geometries(gdf)
    assert fixed.geometry.iloc[0].equals(SQUARE_10x10)


def test_drop_small_polygons_removes_slivers_but_keeps_real_features():
    gdf = make_gdf([SLIVER_1x1, SQUARE_10x10])
    result = drop_small_polygons(gdf, min_area=4)
    assert len(result) == 1
    assert result.geometry.iloc[0].equals(SQUARE_10x10)


def test_drop_small_polygons_keeps_everything_when_threshold_is_low():
    gdf = make_gdf([SLIVER_1x1, SQUARE_10x10])
    result = drop_small_polygons(gdf, min_area=0.5)
    assert len(result) == 2


def test_simplify_polygons_preserves_topology_and_returns_same_count():
    # A polygon with a redundant near-collinear vertex that a tolerance should remove.
    wiggly = Polygon([(0, 0), (5, 0.01), (10, 0), (10, 10), (0, 10)])
    gdf = make_gdf([wiggly])
    result = simplify_polygons(gdf, tolerance=1.0)
    assert len(result) == 1
    assert result.geometry.iloc[0].is_valid


def test_clean_polygons_runs_the_full_pipeline_in_order():
    gdf = make_gdf([BOWTIE, SLIVER_1x1, SQUARE_10x10])
    result = clean_polygons(gdf, min_area=4, simplify_tolerance=0.5)
    # BOWTIE (area 0 as an invalid self-intersecting shape -- becomes a
    # multipolygon of two triangles totaling area 2 after buffer(0), still
    # below the min_area=4 threshold) and the 1x1 sliver are both dropped;
    # only the real 10x10 square survives.
    assert len(result) == 1
    assert result.geometry.iloc[0].is_valid
