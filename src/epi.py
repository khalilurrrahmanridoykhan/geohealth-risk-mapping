"""Phase H9 -- small spatial-epidemiology toolkit: incidence with a population
offset, Poisson / negative-binomial / quasi-Poisson count models, incidence
rate ratios, and a residual spatial-autocorrelation test (Moran's I).

Kept deliberately thin over statsmodels / esda / libpysal -- the value here is
the tested, repeatable wiring (offset handled the same way every time, IRRs
per standard deviation so predictors are comparable, residual Moran's I on
the *right* residuals), not a new estimator. Tested on synthetic data with a
known answer in tests/test_epi.py; the real fit is in
notebooks/10_spatial_epidemiology.ipynb.
"""

from __future__ import annotations

import esda
import geopandas as gpd
import libpysal
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def crude_incidence(cases: pd.Series, population: pd.Series, per: int = 100_000) -> pd.Series:
    """Cases per `per` people. Raw counts mostly rank units by population;
    incidence is what makes them comparable."""
    return cases / population * per


def standardize(values: pd.Series) -> pd.Series:
    """z-score (sample SD), so a model coefficient is 'per 1 SD of the
    predictor' and predictors on different scales are comparable."""
    return (values - values.mean()) / values.std(ddof=1)


def fit_count_model(
    data: pd.DataFrame,
    formula: str,
    population_col: str,
    family: str = "poisson",
    alpha: float | None = None,
):
    """GLM for counts with a log(population) offset, so coefficients describe
    the *rate* (cases per person), not the raw count.

    family: "poisson"; "quasi-poisson" (Poisson mean model, standard errors
    scaled by the Pearson dispersion -- the pragmatic fix for overdispersion
    when n is tiny); "negbin" (negative binomial with a fixed `alpha`; with
    n this small alpha cannot be estimated reliably, so it must be supplied).
    """
    offset = np.log(data[population_col])
    if family == "poisson":
        return smf.glm(formula, data, family=sm.families.Poisson(), offset=offset).fit()
    if family == "quasi-poisson":
        return smf.glm(formula, data, family=sm.families.Poisson(), offset=offset).fit(scale="X2")
    if family == "negbin":
        if alpha is None:
            raise ValueError("family='negbin' needs an explicit alpha")
        return smf.glm(formula, data, family=sm.families.NegativeBinomial(alpha=alpha), offset=offset).fit()
    raise ValueError(f"unknown family {family!r}")


def fit_negbin_mle(data: pd.DataFrame, formula: str, population_col: str):
    """Negative-binomial (NB2) model with the dispersion `alpha` estimated by
    maximum likelihood, then refit as a GLM at that alpha so it shares
    `irr_table` / residual handling with the other families. Returns
    (glm_result, alpha). The GLM's intervals treat alpha as known; NB2's mean
    and dispersion parameters are close to orthogonal so this barely changes
    them, but alpha itself is very uncertain with few units -- inspect it,
    don't over-read it."""
    mle = smf.negativebinomial(formula, data, exposure=data[population_col]).fit(disp=0, maxiter=500)
    if not mle.mle_retvals["converged"]:
        raise RuntimeError("negative-binomial MLE did not converge")
    alpha = float(mle.params["alpha"])
    return fit_count_model(data, formula, population_col, family="negbin", alpha=alpha), alpha


def dispersion(result) -> float:
    """Pearson chi-square / residual df of a fitted GLM. ~1 means the Poisson
    variance assumption holds; well above 1 means overdispersion (standard
    errors too small, p-values too optimistic)."""
    return float(result.pearson_chi2 / result.df_resid)


def irr_table(result) -> pd.DataFrame:
    """Incidence rate ratios (exp of the coefficients) with 95% CIs and
    p-values. The intercept's IRR is the baseline rate per person."""
    ci = result.conf_int()
    return pd.DataFrame(
        {
            "IRR": np.exp(result.params),
            "ci_low": np.exp(ci[0]),
            "ci_high": np.exp(ci[1]),
            "p": result.pvalues,
        }
    )


def queen_weights(boundaries: gpd.GeoDataFrame) -> libpysal.weights.W:
    """Queen-contiguity weights from polygons that share borders (same source
    file, so the borders match exactly). Raises if any unit ends up with no
    neighbour -- Moran's I is meaningless with islands."""
    weights = libpysal.weights.Queen.from_dataframe(boundaries, use_index=False)
    if weights.islands:
        raise ValueError(f"units with no neighbours: {weights.islands}")
    return weights


def residual_moran(residuals: np.ndarray, weights: libpysal.weights.W, permutations: int = 9999, seed: int = 0):
    """Moran's I of model residuals with a permutation p-value (two-sided).

    Spatial autocorrelation left in the residuals means the model missed
    spatial structure and its standard errors are too optimistic. With few
    units the test has very little power: a non-significant result is weak
    evidence of no autocorrelation, not proof of it.
    """
    weights.transform = "r"
    np.random.seed(seed)  # esda draws its permutations from numpy's global RNG
    return esda.Moran(np.asarray(residuals, dtype=float), weights, permutations=permutations, two_tailed=True)
