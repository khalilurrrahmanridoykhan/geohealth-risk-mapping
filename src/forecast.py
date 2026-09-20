"""Phase H10 -- short-horizon dengue risk forecasting, built to be validated
honestly on very little data.

Design choices, all made before looking at any model output:

* **Target = log growth**, `log1p(inc[t+h]) - log1p(inc[t])` (inc = admissions
  per 100k). A model that always predicts 0 growth *is* the persistence
  baseline, so "does the model beat persistence?" is literally "is the
  predicted growth better than zero?". Predicting growth rather than the level
  also lets tree models cope with an epidemic that is growing out of the range
  they were trained on (trees cannot extrapolate a level).
* **Rolling-origin (expanding-window) backtest**: for origin week `o`, a model
  may train only on rows whose *target* week is <= `o`. Anything else leaks the
  future.
* **Fixed hyperparameters**, never tuned against the backtest -- with a few
  hundred rows any tuning against the same folds is optimistic.
* Uncertainty from the model's own past out-of-sample errors (conformal-style),
  using only errors that were already observable at the origin.

Tested on synthetic data in tests/test_forecast.py (including leakage tests);
the real run is notebooks/11_risk_scoring.ipynb.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_SETS = {
    "autoregressive": ["log_inc0", "d1", "d2"],
    "plus_weather": ["log_inc0", "d1", "d2", "rain_recent", "rain_earlier", "temp_recent", "rh_recent"],
    "plus_static": [
        "log_inc0", "d1", "d2", "rain_recent", "rain_earlier", "temp_recent", "rh_recent", "log_density", "water_occ",
    ],
}


def build_panel(
    cases: pd.DataFrame,
    population: pd.Series,
    weather: dict[str, pd.DataFrame],
    static: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """One row per (origin week, division), with only information available at
    the end of the origin week.

    cases: weekly admissions, index = week number, columns = division.
    weather: {'rain_mm'|'temp_c'|'rh_pct': weekly frame, same layout}; may
        extend before week 1 (negative/zero week numbers are fine).
    static: division-indexed frame with `log_density` and `water_occ`.
    Rows whose target lies beyond the data have NaN `y`/`growth` -- those are
    the rows to forecast from.
    """
    inc = cases.div(population, axis="columns") * 1e5
    log_inc = np.log1p(inc)
    frames = []
    for division in cases.columns:
        li = log_inc[division]
        rain = weather["rain_mm"][division]
        frame = pd.DataFrame(index=cases.index)
        frame["division"] = division
        frame["log_inc0"] = li
        frame["d1"] = li - li.shift(1)
        frame["d2"] = li.shift(1) - li.shift(2)
        frame["rain_recent"] = rain.rolling(4).sum().reindex(cases.index)          # weeks t-3..t
        frame["rain_earlier"] = rain.shift(4).rolling(4).sum().reindex(cases.index)  # weeks t-7..t-4
        frame["temp_recent"] = weather["temp_c"][division].rolling(4).mean().reindex(cases.index)
        frame["rh_recent"] = weather["rh_pct"][division].rolling(4).mean().reindex(cases.index)
        frame["log_density"] = static.loc[division, "log_density"]
        frame["water_occ"] = static.loc[division, "water_occ"]
        frame["y"] = li.shift(-horizon)
        frame["growth"] = frame["y"] - frame["log_inc0"]
        frames.append(frame)
    panel = pd.concat(frames)
    panel.index.name = "week"
    panel = panel.reset_index()
    panel["target_week"] = panel["week"] + horizon
    return panel.dropna(subset=["log_inc0", "d1", "d2", "rain_recent", "rain_earlier", "temp_recent", "rh_recent"])


class ZeroGrowth:
    """Persistence: predicts no change. The baseline every model must beat."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


def backtest(panel: pd.DataFrame, model_factory, features: list[str], origins, horizon: int) -> pd.DataFrame:
    """Rolling-origin backtest. At each origin week a fresh model is trained on
    rows whose target week is <= origin, then predicts the growth from that
    origin week to origin + horizon for every division.

    Returns one row per (origin, division): the observed and predicted
    log1p-incidence at origin + horizon, and the persistence prediction.
    """
    labelled = panel.dropna(subset=["growth"])
    rows = []
    for origin in origins:
        train = labelled[labelled["target_week"] <= origin]
        test = labelled[labelled["week"] == origin]
        if train.empty or test.empty:
            continue
        model = model_factory()
        model.fit(train[features].to_numpy(), train["growth"].to_numpy())
        growth_hat = np.asarray(model.predict(test[features].to_numpy()), dtype=float)
        rows.append(
            pd.DataFrame(
                {
                    "origin": origin,
                    "division": test["division"].to_numpy(),
                    "y_true": test["y"].to_numpy(),
                    "y_pred": test["log_inc0"].to_numpy() + growth_hat,
                    "y_persistence": test["log_inc0"].to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def rmse_log(predictions: pd.DataFrame, column: str = "y_pred") -> float:
    """Root-mean-square error on the log1p-incidence scale (a multiplicative
    error: 0.1 is roughly 10% off)."""
    return float(np.sqrt(np.mean((predictions["y_true"] - predictions[column]) ** 2)))


def mean_weekly_spearman(predictions: pd.DataFrame, column: str = "y_pred") -> float:
    """Average across origins of the Spearman rank correlation between
    predicted and observed incidence over divisions -- how well the *ranking*
    is right, which is what a ranked risk list is used for."""
    scores = [
        group["y_true"].corr(group[column], method="spearman") for _, group in predictions.groupby("origin")
    ]
    return float(np.nanmean(scores))


def top_k_hit_rate(predictions: pd.DataFrame, k: int = 2, column: str = "y_pred") -> float:
    """Average fraction of the k divisions predicted highest that are truly in
    the top k that week."""
    hits = []
    for _, group in predictions.groupby("origin"):
        predicted = set(group.nlargest(k, column)["division"])
        actual = set(group.nlargest(k, "y_true")["division"])
        hits.append(len(predicted & actual) / k)
    return float(np.mean(hits))


def add_conformal_intervals(predictions: pd.DataFrame, horizon: int, level: float = 0.8, min_errors: int = 40) -> pd.DataFrame:
    """Prediction intervals from the model's own past errors, respecting time:
    the error for origin o' is only observable once week o' + horizon has
    happened, so an interval at origin o may use only origins o' <= o - horizon.
    Origins with fewer than `min_errors` usable past errors get NaN bounds.

    Adds `lo`/`hi` (log1p-incidence scale) and `covered` (observed inside)."""
    predictions = predictions.copy()
    errors = predictions["y_true"] - predictions["y_pred"]
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    lows, highs = [], []
    for origin in predictions["origin"]:
        usable = errors[predictions["origin"] <= origin - horizon]
        if len(usable) < min_errors:
            lows.append(np.nan)
            highs.append(np.nan)
        else:
            lows.append(usable.quantile(lo))
            highs.append(usable.quantile(hi))
    predictions["lo"] = predictions["y_pred"] + np.array(lows)
    predictions["hi"] = predictions["y_pred"] + np.array(highs)
    inside = (predictions["y_true"] >= predictions["lo"]) & (predictions["y_true"] <= predictions["hi"])
    predictions["covered"] = inside.astype(float).where(predictions["lo"].notna())  # 1.0 / 0.0, NaN if no interval
    return predictions
