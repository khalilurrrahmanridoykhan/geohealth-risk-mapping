import numpy as np
import pytest

from src.imagery import (
    always_cloudy_fraction,
    composite_cloud_masked,
    get_imagery,
    scl_cloud_mask,
    utm_epsg_from_sentinel2_id,
)


def test_scl_cloud_mask_keeps_clear_land_water_snow_classes():
    scl = np.array([4, 5, 6, 7, 11])  # vegetation, bare soil, water, unclassified, snow
    assert scl_cloud_mask(scl).all()


def test_scl_cloud_mask_drops_nodata_shadow_and_cloud_classes():
    scl = np.array([0, 1, 2, 3, 8, 9, 10])  # nodata, saturated, dark, shadow, cloud med/high, cirrus
    assert not scl_cloud_mask(scl).any()


def test_composite_ignores_cloudy_observations():
    # 3 time steps, 1 pixel. Clear value is consistently 100; the cloudy
    # observation (999) should be excluded from the median entirely.
    band_stack = np.array([[[100.0]], [[999.0]], [[102.0]]])  # (time, y, x)
    scl_stack = np.array([[[4]], [[9]], [[4]]])  # clear, cloud-high, clear
    result = composite_cloud_masked(band_stack, scl_stack)
    assert result[0, 0] == 101.0  # median of the two clear observations only


def test_composite_falls_back_to_plain_median_when_always_cloudy():
    # A pixel that's cloudy in every single observation still gets a real
    # number back (the plain median), not NaN.
    band_stack = np.array([[[50.0]], [[60.0]], [[70.0]]])
    scl_stack = np.array([[[9]], [[9]], [[9]]])  # cloud-high every time
    result = composite_cloud_masked(band_stack, scl_stack)
    assert result[0, 0] == 60.0


def test_composite_handles_a_mix_of_always_clear_and_always_cloudy_pixels():
    band_stack = np.array(
        [[[100.0, 50.0]], [[102.0, 60.0]], [[98.0, 70.0]]]
    )  # (time=3, y=1, x=2)
    scl_stack = np.array([[[4, 9]], [[4, 9]], [[4, 9]]])  # left always clear, right always cloudy
    result = composite_cloud_masked(band_stack, scl_stack)
    assert result[0, 0] == 100.0  # median of clear observations
    assert result[0, 1] == 60.0  # fallback plain median


def test_always_cloudy_fraction_is_zero_when_every_pixel_has_a_clear_look():
    scl_stack = np.array([[[4, 6]], [[9, 9]], [[4, 6]]])  # left always clear, right cloudy once
    assert always_cloudy_fraction(scl_stack) == 0.0


def test_always_cloudy_fraction_counts_pixels_with_zero_clear_observations():
    scl_stack = np.array([[[4, 9]], [[4, 9]], [[4, 9]]])  # left always clear, right always cloudy
    assert always_cloudy_fraction(scl_stack) == 0.5


def test_utm_epsg_from_sentinel2_id_parses_the_mgrs_zone():
    assert utm_epsg_from_sentinel2_id("S2C_MSIL2A_20260214T042901_R133_T46QBM_20260214T080510") == 32646


def test_utm_epsg_from_sentinel2_id_handles_a_different_zone():
    assert utm_epsg_from_sentinel2_id("S2A_MSIL2A_20260101T000000_R000_T45QYE_20260101T000000") == 32645


def test_utm_epsg_from_sentinel2_id_raises_on_an_unrecognized_id():
    with pytest.raises(ValueError):
        utm_epsg_from_sentinel2_id("not-a-real-sentinel-2-id")


def test_get_imagery_rejects_an_unknown_source():
    with pytest.raises(ValueError):
        get_imagery((90.0, 23.0, 90.1, 23.1), "2026-01-01/2026-01-31", source="not_a_real_source")


def test_get_imagery_rejects_an_unknown_sensor():
    # sensor='sentinel-1' is a real, implemented path as of Phase H5 (uses
    # the sentinel-1-rtc collection -- see _get_sentinel1_composite's
    # docstring) -- exercised for real, network-required, in
    # notebooks/06_sar_flood_mapping.ipynb, not here. This test only checks
    # the fast, offline validation.
    with pytest.raises(ValueError):
        get_imagery((90.0, 23.0, 90.1, 23.1), "2026-01-01/2026-01-31", sensor="not_a_real_sensor")
