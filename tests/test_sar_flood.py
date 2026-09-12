import numpy as np
import pytest

from src.sar_flood import db_from_power, flood_extent, otsu_threshold, water_mask_from_backscatter


def test_db_from_power_matches_the_standard_formula():
    power = np.array([1.0, 0.1, 0.01])
    result = db_from_power(power)
    assert np.allclose(result, [0.0, -10.0, -20.0])


def test_db_from_power_turns_non_positive_values_into_nan():
    power = np.array([1.0, 0.0, -0.5])
    result = db_from_power(power)
    assert result[0] == 0.0
    assert np.isnan(result[1])
    assert np.isnan(result[2])


def test_otsu_threshold_separates_a_clearly_bimodal_distribution():
    # Water backscatter clustered around -20dB, land clustered around -5dB.
    rng = np.random.default_rng(42)
    water = rng.normal(-20, 1.0, 1000)
    land = rng.normal(-5, 1.0, 1000)
    values = np.concatenate([water, land])
    threshold = otsu_threshold(values)
    assert -20 < threshold < -5
    # And it actually separates most of each cluster correctly.
    assert (water < threshold).mean() > 0.9
    assert (land < threshold).mean() < 0.1


def test_otsu_threshold_ignores_nan_values():
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(-20, 1.0, 500), rng.normal(-5, 1.0, 500), [np.nan] * 100])
    threshold = otsu_threshold(values)
    assert np.isfinite(threshold)


def test_otsu_threshold_raises_on_all_nan_input():
    with pytest.raises(ValueError):
        otsu_threshold(np.array([np.nan, np.nan]))


def test_water_mask_from_backscatter_uses_the_threshold_correctly():
    vv_db = np.array([-25.0, -10.0, -18.0, -5.0])
    mask = water_mask_from_backscatter(vv_db, threshold_db=-15.0)
    assert list(mask) == [True, False, True, False]


def test_water_mask_from_backscatter_never_classifies_nan_as_water():
    vv_db = np.array([-25.0, np.nan])
    mask = water_mask_from_backscatter(vv_db, threshold_db=-15.0)
    assert list(mask) == [True, False]


def test_flood_extent_is_water_now_but_not_before():
    before = np.array([True, False, False, True])
    during = np.array([True, True, False, False])
    result = flood_extent(before, during)
    # index 0: water both times -> not "newly" flooded
    # index 1: dry before, water now -> newly flooded
    # index 2: dry both times -> not flooded
    # index 3: water before, dry now -> not newly flooded (water receded, not new)
    assert list(result) == [False, True, False, False]
