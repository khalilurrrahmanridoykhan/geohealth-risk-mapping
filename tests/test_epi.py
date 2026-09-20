import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from src.epi import (
    crude_incidence,
    dispersion,
    fit_count_model,
    fit_negbin_mle,
    irr_table,
    queen_weights,
    residual_moran,
    standardize,
)


def _synthetic_counts(n: int, true_irr_per_sd: float, seed: int = 1, extra_dispersion: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    population = rng.integers(50_000, 500_000, size=n)
    x = standardize(pd.Series(rng.normal(size=n)))
    log_rate = np.log(2e-4) + np.log(true_irr_per_sd) * x
    if extra_dispersion:
        log_rate = log_rate + rng.normal(0, extra_dispersion, size=n)
    return pd.DataFrame({"population": population, "x": x, "cases": rng.poisson(population * np.exp(log_rate))})


def _grid(n_side: int) -> gpd.GeoDataFrame:
    cells = [box(i, j, i + 1, j + 1) for j in range(n_side) for i in range(n_side)]
    return gpd.GeoDataFrame({"id": range(len(cells))}, geometry=cells)


def test_crude_incidence_is_per_100k_by_default():
    incidence = crude_incidence(pd.Series([50, 10]), pd.Series([100_000, 20_000]))
    assert incidence.tolist() == [50.0, 50.0]


def test_standardize_gives_mean_zero_and_unit_sample_sd():
    z = standardize(pd.Series([1.0, 2.0, 3.0, 4.0]))
    assert z.mean() == pytest.approx(0)
    assert z.std(ddof=1) == pytest.approx(1)


def test_poisson_glm_with_offset_recovers_the_true_rate_ratio():
    data = _synthetic_counts(n=4000, true_irr_per_sd=1.5)
    irr = irr_table(fit_count_model(data, "cases ~ x", "population"))
    assert irr.loc["x", "IRR"] == pytest.approx(1.5, rel=0.03)
    assert irr.loc["x", "ci_low"] < 1.5 < irr.loc["x", "ci_high"]
    assert irr.loc["Intercept", "IRR"] == pytest.approx(2e-4, rel=0.03)  # baseline rate per person


def test_offset_is_what_removes_the_population_effect():
    # counts scale with population but the rate is constant: with the offset the
    # predictor is (correctly) null; a model without it would call population a risk factor
    data = _synthetic_counts(n=4000, true_irr_per_sd=1.0)
    irr = irr_table(fit_count_model(data, "cases ~ x", "population"))
    assert irr.loc["x", "ci_low"] < 1.0 < irr.loc["x", "ci_high"]


def test_dispersion_is_near_one_for_poisson_data_and_large_when_overdispersed():
    poisson = fit_count_model(_synthetic_counts(2000, 1.3), "cases ~ x", "population")
    overdispersed = fit_count_model(_synthetic_counts(2000, 1.3, extra_dispersion=0.8), "cases ~ x", "population")
    assert dispersion(poisson) == pytest.approx(1.0, abs=0.15)
    assert dispersion(overdispersed) > 5


def test_quasi_poisson_widens_intervals_but_keeps_the_estimate():
    data = _synthetic_counts(500, 1.3, extra_dispersion=0.8)
    poisson = irr_table(fit_count_model(data, "cases ~ x", "population"))
    quasi = irr_table(fit_count_model(data, "cases ~ x", "population", family="quasi-poisson"))
    assert quasi.loc["x", "IRR"] == pytest.approx(poisson.loc["x", "IRR"])
    assert (quasi.loc["x", "ci_high"] - quasi.loc["x", "ci_low"]) > 2 * (
        poisson.loc["x", "ci_high"] - poisson.loc["x", "ci_low"]
    )


def test_negbin_requires_an_explicit_alpha():
    with pytest.raises(ValueError, match="alpha"):
        fit_count_model(_synthetic_counts(50, 1.2), "cases ~ x", "population", family="negbin")


def test_negbin_fits_with_a_supplied_alpha():
    result = fit_count_model(_synthetic_counts(200, 1.4), "cases ~ x", "population", family="negbin", alpha=0.5)
    assert irr_table(result).loc["x", "IRR"] == pytest.approx(1.4, rel=0.25)


def test_negbin_mle_recovers_the_dispersion_and_the_rate_ratio():
    # log-normal extra variation sigma=0.6 -> NB alpha of roughly exp(sigma^2)-1 ~ 0.43
    data = _synthetic_counts(3000, 1.4, extra_dispersion=0.6)
    result, alpha = fit_negbin_mle(data, "cases ~ x", "population")
    assert alpha == pytest.approx(0.43, abs=0.12)
    assert irr_table(result).loc["x", "IRR"] == pytest.approx(1.4, rel=0.06)


def test_negbin_mle_alpha_is_near_zero_for_plain_poisson_data():
    _, alpha = fit_negbin_mle(_synthetic_counts(3000, 1.4), "cases ~ x", "population")
    assert alpha < 0.05


def test_unknown_family_raises():
    with pytest.raises(ValueError, match="family"):
        fit_count_model(_synthetic_counts(50, 1.2), "cases ~ x", "population", family="gaussian")


def test_queen_weights_on_a_grid_counts_edge_and_corner_neighbours():
    weights = queen_weights(_grid(3))
    assert sorted(weights.cardinalities.values()) == [3, 3, 3, 3, 5, 5, 5, 5, 8]


def test_queen_weights_rejects_islands():
    far_apart = gpd.GeoDataFrame({"id": [0, 1, 2]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(10, 10, 11, 11)])
    with pytest.raises(ValueError, match="no neighbours"):
        queen_weights(far_apart)


def test_residual_moran_flags_a_clear_spatial_gradient():
    grid = _grid(6)
    gradient = np.array([c.centroid.x + c.centroid.y for c in grid.geometry])
    moran = residual_moran(gradient, queen_weights(grid), permutations=999)
    assert moran.I > 0.5
    assert moran.p_sim < 0.01


def test_residual_moran_is_unremarkable_for_spatially_random_noise():
    grid = _grid(6)
    noise = np.random.default_rng(3).normal(size=len(grid))
    moran = residual_moran(noise, queen_weights(grid), permutations=999)
    assert abs(moran.I) < 0.3
    assert moran.p_sim > 0.05


def test_residual_moran_is_reproducible_for_a_fixed_seed():
    grid = _grid(4)
    values = np.random.default_rng(5).normal(size=len(grid))
    weights = queen_weights(grid)
    assert residual_moran(values, weights, 999, seed=7).p_sim == residual_moran(values, weights, 999, seed=7).p_sim
