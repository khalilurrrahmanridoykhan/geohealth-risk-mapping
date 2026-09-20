import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from src.weather import parse_power_response, population_weighted_centroid, to_iso_weeks


def _write_raster(path, array, nodata=-99999.0):
    transform = from_origin(0.0, 4.0, 1.0, 1.0)  # 4x4 cells, 1 degree each, top-left at (0, 4)
    with rasterio.open(path, "w", driver="GTiff", height=4, width=4, count=1, dtype="float32",
                       crs="EPSG:4326", transform=transform, nodata=nodata) as dst:
        dst.write(array.astype("float32"), 1)


def test_centroid_is_pulled_towards_the_populated_cell(tmp_path):
    population = np.zeros((4, 4))
    population[0, 3] = 1000.0  # top-right cell: centre (3.5, 3.5)
    path = tmp_path / "pop.tif"
    _write_raster(path, population)
    lon, lat = population_weighted_centroid(str(path), box(0, 0, 4, 4))
    assert (lon, lat) == pytest.approx((3.5, 3.5))


def test_centroid_of_uniform_population_is_the_geometric_centre(tmp_path):
    path = tmp_path / "pop.tif"
    _write_raster(path, np.ones((4, 4)))
    lon, lat = population_weighted_centroid(str(path), box(0, 0, 4, 4))
    assert (lon, lat) == pytest.approx((2.0, 2.0))


def test_centroid_ignores_nodata_cells(tmp_path):
    population = np.full((4, 4), -99999.0)
    population[3, 0] = 10.0  # bottom-left cell: centre (0.5, 0.5)
    path = tmp_path / "pop.tif"
    _write_raster(path, population)
    lon, lat = population_weighted_centroid(str(path), box(0, 0, 4, 4))
    assert (lon, lat) == pytest.approx((0.5, 0.5))


def test_centroid_falls_back_to_a_point_inside_the_polygon_when_nobody_lives_there(tmp_path):
    path = tmp_path / "pop.tif"
    _write_raster(path, np.zeros((4, 4)))
    geometry = box(0, 0, 4, 4)
    lon, lat = population_weighted_centroid(str(path), geometry)
    assert geometry.contains(geometry.representative_point())
    assert (lon, lat) == pytest.approx((geometry.representative_point().x, geometry.representative_point().y))


def _payload(rain, temp, rh, start="20260105"):
    days = pd.date_range(start, periods=len(rain)).strftime("%Y%m%d")
    return {"properties": {"parameter": {
        "PRECTOTCORR": dict(zip(days, rain)), "T2M": dict(zip(days, temp)), "RH2M": dict(zip(days, rh)),
    }}}


def test_parse_power_response_renames_columns_and_turns_fill_values_into_nan():
    daily = parse_power_response(_payload([0.0, -999.0, 5.0], [20.0, 21.0, 22.0], [80.0, 81.0, 82.0]))
    assert list(daily.columns) == ["rain_mm", "temp_c", "rh_pct"]
    assert daily["rain_mm"].isna().tolist() == [False, True, False]
    assert daily.index[0] == pd.Timestamp("2026-01-05")


def test_to_iso_weeks_sums_rain_and_averages_temperature():
    # 2026-01-05 is a Monday, ISO week 2
    daily = parse_power_response(_payload([1.0] * 7 + [2.0] * 7, [20.0] * 7 + [30.0] * 7, [80.0] * 14))
    weekly = to_iso_weeks(daily)
    assert weekly.loc[(2026, 2), "rain_mm"] == 7.0
    assert weekly.loc[(2026, 3), "rain_mm"] == 14.0
    assert weekly.loc[(2026, 3), "temp_c"] == 30.0


def test_to_iso_weeks_drops_a_partial_trailing_week():
    daily = parse_power_response(_payload([1.0] * 10, [20.0] * 10, [80.0] * 10))
    assert list(to_iso_weeks(daily).index) == [(2026, 2)]  # the 3 extra days are not a full week


def test_to_iso_weeks_does_not_report_zero_rain_for_a_week_with_missing_days():
    rain = [1.0] * 7
    rain[3] = -999.0
    weekly = to_iso_weeks(parse_power_response(_payload(rain, [20.0] * 7, [80.0] * 7)))
    assert np.isnan(weekly.loc[(2026, 2), "rain_mm"])
