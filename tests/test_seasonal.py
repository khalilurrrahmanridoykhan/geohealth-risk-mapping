import numpy as np
import pytest

from src.seasonal import green_up, seasonal_ndvi_range, water_persistence


def test_water_persistence_is_one_for_a_pixel_that_is_always_water():
    masks = [np.array([[True, False]]), np.array([[True, False]]), np.array([[True, False]])]
    result = water_persistence(masks)
    assert result[0, 0] == 1.0
    assert result[0, 1] == 0.0


def test_water_persistence_is_fractional_for_seasonal_water():
    masks = [np.array([[True]]), np.array([[False]]), np.array([[False]])]
    result = water_persistence(masks)
    assert abs(result[0, 0] - 1 / 3) < 1e-9


def test_water_persistence_rejects_an_empty_list():
    with pytest.raises(ValueError):
        water_persistence([])


def test_seasonal_ndvi_range_is_zero_for_a_permanently_stable_pixel():
    layers = [np.array([[0.5]]), np.array([[0.5]]), np.array([[0.5]])]
    result = seasonal_ndvi_range(layers)
    assert result[0, 0] == 0.0


def test_seasonal_ndvi_range_captures_the_real_swing():
    layers = [np.array([[0.1]]), np.array([[0.6]]), np.array([[0.3]])]  # dry, wet, post
    result = seasonal_ndvi_range(layers)
    assert abs(result[0, 0] - 0.5) < 1e-9  # 0.6 - 0.1


def test_seasonal_ndvi_range_rejects_an_empty_list():
    with pytest.raises(ValueError):
        seasonal_ndvi_range([])


def test_green_up_is_positive_when_wet_season_is_greener():
    dry = np.array([[0.1, 0.4]])
    wet = np.array([[0.5, 0.3]])
    result = green_up(dry, wet)
    assert result[0, 0] > 0  # greened up
    assert result[0, 1] < 0  # actually greener in the dry season (e.g. irrigated cropland)
