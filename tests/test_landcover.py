import numpy as np
import pytest

from src.landcover import (
    BARE,
    BUILT_UP,
    VEGETATION,
    WATER,
    build_feature_stack,
    reclassify_worldcover,
    spatial_block_split,
)


def test_reclassify_worldcover_maps_vegetation_classes():
    wc = np.array([10, 20, 30, 40, 95])  # tree, shrub, grass, cropland, mangrove
    assert (reclassify_worldcover(wc) == VEGETATION).all()


def test_reclassify_worldcover_maps_built_up():
    wc = np.array([50, 50])
    assert (reclassify_worldcover(wc) == BUILT_UP).all()


def test_reclassify_worldcover_maps_water_including_wetland():
    wc = np.array([80, 90])  # water bodies, herbaceous wetland
    assert (reclassify_worldcover(wc) == WATER).all()


def test_reclassify_worldcover_maps_bare_classes():
    wc = np.array([60, 70, 100])
    assert (reclassify_worldcover(wc) == BARE).all()


def test_reclassify_worldcover_flags_unknown_codes_as_minus_one():
    wc = np.array([50, 999])
    result = reclassify_worldcover(wc)
    assert result[0] == BUILT_UP
    assert result[1] == -1


def test_build_feature_stack_shape_and_band_order():
    green = np.full((3, 3), 500.0)
    red = np.full((3, 3), 400.0)
    nir = np.full((3, 3), 3000.0)
    swir = np.full((3, 3), 800.0)
    features = build_feature_stack(green, red, nir, swir)
    assert features.shape == (7, 3, 3)
    assert np.allclose(features[0], 500.0)  # green passed through unchanged
    assert np.allclose(features[3], 800.0)  # swir passed through unchanged


def test_spatial_block_split_holds_out_the_right_fraction_of_columns():
    mask = spatial_block_split(height=10, width=100, test_fraction=0.3, axis="x")
    assert mask.shape == (10, 100)
    assert mask[:, :70].sum() == 0  # nothing held out on the left 70%
    assert mask[:, 70:].sum() == 10 * 30  # everything held out on the right 30%


def test_spatial_block_split_axis_y_splits_by_row():
    mask = spatial_block_split(height=100, width=10, test_fraction=0.2, axis="y")
    assert mask[:80, :].sum() == 0
    assert mask[80:, :].sum() == 20 * 10


def test_spatial_block_split_rejects_an_invalid_fraction():
    with pytest.raises(ValueError):
        spatial_block_split(10, 10, test_fraction=1.5)


def test_spatial_block_split_rejects_an_invalid_axis():
    with pytest.raises(ValueError):
        spatial_block_split(10, 10, axis="z")
