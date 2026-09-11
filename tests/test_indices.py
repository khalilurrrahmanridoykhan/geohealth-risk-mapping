import numpy as np

from src.indices import ndbi, ndvi, ndwi


def test_ndvi_is_high_for_dense_vegetation():
    nir = np.array([[800.0]])
    red = np.array([[100.0]])
    assert ndvi(nir, red)[0, 0] > 0.7


def test_ndvi_is_near_zero_for_bare_soil():
    nir = np.array([[300.0]])
    red = np.array([[280.0]])
    assert abs(ndvi(nir, red)[0, 0]) < 0.1


def test_ndvi_handles_the_zero_denominator_case():
    nir = np.array([[0.0]])
    red = np.array([[0.0]])
    assert ndvi(nir, red)[0, 0] == 0.0


def test_ndwi_is_high_for_open_water():
    green = np.array([[600.0]])
    nir = np.array([[100.0]])
    assert ndwi(green, nir)[0, 0] > 0.5


def test_ndwi_is_negative_for_dry_land():
    green = np.array([[300.0]])
    nir = np.array([[700.0]])
    assert ndwi(green, nir)[0, 0] < 0


def test_ndbi_is_high_for_built_up_surface():
    swir = np.array([[700.0]])
    nir = np.array([[300.0]])
    assert ndbi(swir, nir)[0, 0] >= 0.4


def test_ndbi_is_negative_for_vegetation():
    swir = np.array([[200.0]])
    nir = np.array([[800.0]])
    assert ndbi(swir, nir)[0, 0] < 0


def test_indices_operate_elementwise_on_a_full_array():
    nir = np.array([[800.0, 300.0], [100.0, 500.0]])
    red = np.array([[100.0, 280.0], [600.0, 500.0]])
    result = ndvi(nir, red)
    assert result.shape == (2, 2)
    assert result[1, 1] == 0.0  # equal bands -> exactly zero
