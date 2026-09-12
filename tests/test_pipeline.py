import geopandas as gpd
import numpy as np
from affine import Affine
from shapely.geometry import box

from src.pipeline import build_predictor_table, vectorize_mask, zonal_predictor_table

# A tiny 10x10 raster, 1 unit per pixel, split into a left half (district A)
# and right half (district B) -- easy to reason about by hand.
TRANSFORM = Affine.translation(0, 0) @ Affine.scale(1, -1)
DISTRICT_A = box(0, -10, 5, 0)
DISTRICT_B = box(5, -10, 10, 0)
BOUNDARIES = gpd.GeoDataFrame({"name": ["A", "B"]}, geometry=[DISTRICT_A, DISTRICT_B], crs="EPSG:32646")


def test_zonal_predictor_table_computes_per_district_means():
    raster = np.zeros((10, 10))
    raster[:, :5] = 1.0  # left half (district A) = 1, right half (district B) = 0
    table = zonal_predictor_table({"my_index": raster}, TRANSFORM, BOUNDARIES, "name")
    assert table.loc["A", "my_index"] == 1.0
    assert table.loc["B", "my_index"] == 0.0


def test_zonal_predictor_table_handles_multiple_rasters():
    r1 = np.full((10, 10), 2.0)
    r2 = np.full((10, 10), 5.0)
    table = zonal_predictor_table({"r1": r1, "r2": r2}, TRANSFORM, BOUNDARIES, "name")
    assert list(table.columns) == ["r1", "r2"]
    assert (table["r1"] == 2.0).all()
    assert (table["r2"] == 5.0).all()


def test_vectorize_mask_produces_clean_polygons_from_a_boolean_raster():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True  # one solid 6x6 block
    gdf = vectorize_mask(mask, TRANSFORM, "EPSG:32646", min_area=1.0, simplify_tolerance=0.1)
    assert len(gdf) == 1
    assert gdf.geometry.iloc[0].is_valid
    assert abs(gdf.geometry.iloc[0].area - 36.0) < 1e-6


def test_vectorize_mask_drops_single_pixel_noise():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    mask[0, 0] = True  # a single-pixel sliver far from the real block
    gdf = vectorize_mask(mask, TRANSFORM, "EPSG:32646", min_area=2.0, simplify_tolerance=0.1)
    assert len(gdf) == 1  # the 1x1 sliver (area 1.0) is dropped, the 6x6 block survives


def test_vectorize_mask_handles_an_all_false_mask():
    mask = np.zeros((10, 10), dtype=bool)
    gdf = vectorize_mask(mask, TRANSFORM, "EPSG:32646", min_area=1.0, simplify_tolerance=0.1)
    assert len(gdf) == 0


def test_build_predictor_table_writes_a_reproducible_geopackage(tmp_path):
    raster = np.zeros((10, 10))
    raster[:, :5] = 1.0
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True

    out_path = tmp_path / "test_output.gpkg"
    table1, vectorized1 = build_predictor_table(
        {"my_index": raster}, TRANSFORM, "EPSG:32646", BOUNDARIES, "name",
        {"my_mask": mask}, out_path, min_area=1.0, simplify_tolerance=0.1,
    )
    assert out_path.exists()

    # Running it again on the same inputs reproduces the same table and layers.
    table2, vectorized2 = build_predictor_table(
        {"my_index": raster}, TRANSFORM, "EPSG:32646", BOUNDARIES, "name",
        {"my_mask": mask}, out_path, min_area=1.0, simplify_tolerance=0.1,
    )
    assert table1.equals(table2)
    assert len(vectorized1["my_mask"]) == len(vectorized2["my_mask"])

    boundaries_layer = gpd.read_file(out_path, layer="boundaries_with_predictors")
    assert "my_index" in boundaries_layer.columns
    mask_layer = gpd.read_file(out_path, layer="my_mask")
    assert len(mask_layer) == 1


def test_build_predictor_table_skips_empty_vectorized_layers(tmp_path):
    raster = np.ones((10, 10))
    empty_mask = np.zeros((10, 10), dtype=bool)
    out_path = tmp_path / "test_empty.gpkg"

    table, vectorized = build_predictor_table(
        {"my_index": raster}, TRANSFORM, "EPSG:32646", BOUNDARIES, "name",
        {"empty": empty_mask}, out_path, min_area=1.0, simplify_tolerance=0.1,
    )
    assert len(vectorized["empty"]) == 0
    import fiona
    assert "empty" not in fiona.listlayers(out_path)
