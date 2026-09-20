"""Phase H10 -- short-horizon dengue risk forecasting, built to be validated
honestly on very little data.

Design choices made before the first real backtest run: the growth target, the
rolling-origin scheme, the feature sets, the fixed hyperparameters, and the
first origin (week 20). Added *after* the first run: `TrendGrowth` (an
alternative baseline, on the thought that persistence -- zero growth -- is weak
in a growing epidemic; on the real data it turned out *worse* than persistence)
and `MeanEnsemble` (the PLAN's GLM + XGBoost combination). Disclosed in the
notebook; every configuration run is reported, not only the good ones.

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


class TrendGrowth:
    """Trend extrapolation: assumes the average of the last two weekly log
    changes (features d1, d2) continues for `horizon` weeks. An alternative
    baseline to persistence (which predicts zero growth by construction);
    weekly changes are noisy and mean-reverting, so on the real 2026 data it
    does worse than persistence. `d1_col`/`d2_col` are the positions of d1
    and d2 in the feature matrix."""

    def __init__(self, horizon: int, d1_col: int, d2_col: int):
        self.horizon, self.d1_col, self.d2_col = horizon, d1_col, d2_col

    def fit(self, X, y):
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.horizon * (X[:, self.d1_col] + X[:, self.d2_col]) / 2


class MeanEnsemble:
    """Average of the growth predictions of several models."""

    def __init__(self, factories):
        self.factories = list(factories)
        self.models = []

    def fit(self, X, y):
        self.models = [factory().fit(X, y) for factory in self.factories]
        return self

    def predict(self, X):
        return np.mean([np.asarray(m.predict(X), dtype=float) for m in self.models], axis=0)


def model_factories(horizon: int, features: list[str]) -> dict:
    """The fixed model set used for every backtest: ridge regression, a shallow
    gradient-boosted tree model, their mean ensemble, and (when d1/d2 are among
    the features) the trend baseline. Hyperparameters are set once here and
    never tuned against the backtest."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    def ridge():
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))

    def xgboost():
        return XGBRegressor(
            n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8,
            min_child_weight=5, reg_lambda=5, random_state=0, n_jobs=1,
        )

    factories = {"ridge": ridge, "xgboost": xgboost, "ensemble": lambda: MeanEnsemble([ridge, xgboost])}
    if {"d1", "d2"} <= set(features):
        factories["trend"] = lambda: TrendGrowth(horizon, features.index("d1"), features.index("d2"))
    return factories


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


def paired_block_bootstrap_rmse_diff(
    a: pd.DataFrame, b: pd.DataFrame, block: int = 4, n_boot: int = 2000, seed: int = 0, column: str = "y_pred"
) -> tuple[float, float, float]:
    """RMSE(a) - RMSE(b) over the same backtest points, with a 95% interval
    from a moving-block bootstrap over *origin weeks* (blocks keep neighbouring,
    autocorrelated weeks together; resampling single rows would make the
    interval far too narrow). Negative = a is better. Returns (diff, lo, hi).
    """
    merged = a.merge(b, on=["origin", "division"], suffixes=("_a", "_b"))
    if len(merged) != len(a) or len(merged) != len(b):
        raise ValueError("the two backtests must cover the same origins and divisions")
    per_origin = merged.groupby("origin").apply(
        lambda g: pd.Series(
            {
                "mse_a": ((g["y_true_a"] - g[f"{column}_a"]) ** 2).mean(),
                "mse_b": ((g["y_true_b"] - g[f"{column}_b"]) ** 2).mean(),
            }
        ),
        include_groups=False,
    ).sort_index()
    mse_a, mse_b = per_origin["mse_a"].to_numpy(), per_origin["mse_b"].to_numpy()
    n = len(per_origin)
    block = min(block, n)
    diff = float(np.sqrt(mse_a.mean()) - np.sqrt(mse_b.mean()))
    rng = np.random.default_rng(seed)
    starts_available = n - block + 1
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, starts_available, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        diffs[i] = np.sqrt(mse_a[idx].mean()) - np.sqrt(mse_b[idx].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return diff, float(lo), float(hi)
