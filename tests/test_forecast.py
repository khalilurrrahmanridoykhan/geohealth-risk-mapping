import numpy as np
import pandas as pd
import pytest

from src.forecast import (
    FEATURE_SETS,
    ZeroGrowth,
    add_conformal_intervals,
    backtest,
    build_panel,
    mean_weekly_spearman,
    rmse_log,
    top_k_hit_rate,
)

DIVISIONS = ["A", "B", "C"]
POPULATION = pd.Series({"A": 100_000.0, "B": 200_000.0, "C": 400_000.0})
STATIC = pd.DataFrame({"log_density": [1.0, 2.0, 3.0], "water_occ": [5.0, 6.0, 7.0]}, index=DIVISIONS)


def _inputs(n_weeks=20, growth=1.1, seed=0):
    """Cases grow geometrically at `growth` per week with mild noise; weather
    starts 10 weeks before week 1."""
    rng = np.random.default_rng(seed)
    weeks = pd.RangeIndex(1, n_weeks + 1)
    cases = pd.DataFrame(
        {d: (50 * (growth ** np.arange(n_weeks)) * rng.uniform(0.95, 1.05, n_weeks)).round() for d in DIVISIONS},
        index=weeks,
    )
    weather_weeks = pd.RangeIndex(-9, n_weeks + 1)
    weather = {
        "rain_mm": pd.DataFrame({d: np.arange(len(weather_weeks), dtype=float) for d in DIVISIONS}, index=weather_weeks),
        "temp_c": pd.DataFrame(25.0, index=weather_weeks, columns=DIVISIONS),
        "rh_pct": pd.DataFrame(80.0, index=weather_weeks, columns=DIVISIONS),
    }
    return cases, weather


def test_panel_features_and_target_alignment():
    cases, weather = _inputs()
    panel = build_panel(cases, POPULATION, weather, STATIC, horizon=2)
    row = panel[(panel["division"] == "A") & (panel["week"] == 10)].iloc[0]
    inc = lambda w: cases.loc[w, "A"] / POPULATION["A"] * 1e5
    assert row["log_inc0"] == pytest.approx(np.log1p(inc(10)))
    assert row["d1"] == pytest.approx(np.log1p(inc(10)) - np.log1p(inc(9)))
    assert row["d2"] == pytest.approx(np.log1p(inc(9)) - np.log1p(inc(8)))
    assert row["y"] == pytest.approx(np.log1p(inc(12)))
    assert row["growth"] == pytest.approx(row["y"] - row["log_inc0"])
    assert row["target_week"] == 12


def test_panel_rain_windows_use_only_weeks_up_to_the_origin():
    cases, weather = _inputs()
    panel = build_panel(cases, POPULATION, weather, STATIC, horizon=2)
    rain = weather["rain_mm"]["A"]  # rain in week w equals w + 9
    row = panel[(panel["division"] == "A") & (panel["week"] == 10)].iloc[0]
    assert row["rain_recent"] == rain.loc[7:10].sum()
    assert row["rain_earlier"] == rain.loc[3:6].sum()


def test_panel_features_never_depend_on_data_after_the_origin_week():
    cases, weather = _inputs()
    base = build_panel(cases, POPULATION, weather, STATIC, horizon=2)
    tampered_cases = cases.copy()
    tampered_cases.loc[15:] = 999_999.0
    tampered_weather = {k: v.copy() for k, v in weather.items()}
    tampered_weather["rain_mm"].loc[15:] = 12345.0
    tampered = build_panel(tampered_cases, POPULATION, tampered_weather, STATIC, horizon=2)
    features = FEATURE_SETS["plus_static"]
    early_base = base[base["week"] <= 14].set_index(["division", "week"])[features]
    early_tampered = tampered[tampered["week"] <= 14].set_index(["division", "week"])[features]
    pd.testing.assert_frame_equal(early_base, early_tampered)


def test_panel_keeps_unlabelled_rows_to_forecast_from():
    cases, weather = _inputs(n_weeks=12)
    panel = build_panel(cases, POPULATION, weather, STATIC, horizon=2)
    last = panel[panel["week"] == 12]
    assert len(last) == 3 and last["y"].isna().all()


def test_backtest_never_trains_on_rows_whose_target_is_after_the_origin():
    cases, weather = _inputs(n_weeks=25)
    panel = build_panel(cases, POPULATION, weather, STATIC, horizon=3)
    seen = []

    class Recorder(ZeroGrowth):
        def fit(self, X, y):
            seen.append(len(y))
            return self

    backtest(panel, Recorder, FEATURE_SETS["autoregressive"], origins=[12, 15, 18], horizon=3)
    labelled = panel.dropna(subset=["growth"])
    expected = [int((labelled["target_week"] <= o).sum()) for o in (12, 15, 18)]
    assert seen == expected  # exactly the rows resolved by each origin, no more


def test_backtest_predictions_are_persistence_plus_predicted_growth():
    cases, weather = _inputs(n_weeks=25)
    panel = build_panel(cases, POPULATION, weather, STATIC, horizon=2)
    result = backtest(panel, ZeroGrowth, FEATURE_SETS["autoregressive"], origins=[15], horizon=2)
    assert len(result) == 3
    assert (result["y_pred"] == result["y_persistence"]).all()


def test_a_linear_model_beats_persistence_on_steadily_growing_data():
    from sklearn.linear_model import Ridge

    cases, weather = _inputs(n_weeks=30, growth=1.15)
    panel = build_panel(cases, POPULATION, weather, STATIC, horizon=2)
    origins = range(14, 28)
    persistence = backtest(panel, ZeroGrowth, FEATURE_SETS["autoregressive"], origins, 2)
    ridge = backtest(panel, lambda: Ridge(alpha=1e-3), FEATURE_SETS["autoregressive"], origins, 2)
    assert rmse_log(ridge) < rmse_log(persistence) / 2


def test_rmse_log_is_zero_for_perfect_predictions():
    frame = pd.DataFrame({"origin": [1, 1], "division": ["A", "B"], "y_true": [1.0, 2.0], "y_pred": [1.0, 2.0]})
    assert rmse_log(frame) == 0.0


def test_spearman_is_one_for_a_correct_ranking_and_minus_one_for_a_reversed_one():
    base = {"origin": [1, 1, 1], "division": ["A", "B", "C"], "y_true": [1.0, 2.0, 3.0]}
    assert mean_weekly_spearman(pd.DataFrame({**base, "y_pred": [10.0, 20.0, 30.0]})) == pytest.approx(1.0)
    assert mean_weekly_spearman(pd.DataFrame({**base, "y_pred": [30.0, 20.0, 10.0]})) == pytest.approx(-1.0)


def test_top_k_hit_rate():
    frame = pd.DataFrame(
        {"origin": [1] * 4, "division": list("ABCD"), "y_true": [4.0, 3.0, 2.0, 1.0], "y_pred": [4.0, 1.0, 3.0, 2.0]}
    )
    assert top_k_hit_rate(frame, k=2) == 0.5  # predicted {A, C}, actual {A, B}


def _errors_frame(errors_by_origin: dict[int, float]) -> pd.DataFrame:
    rows = [
        {"origin": o, "division": d, "y_true": e, "y_pred": 0.0}
        for o, e in errors_by_origin.items()
        for d in ("A", "B")
    ]
    return pd.DataFrame(rows)


def test_intervals_use_only_errors_observable_at_the_origin():
    horizon, min_errors = 2, 4
    origins = list(range(1, 11))
    base = _errors_frame({o: (o % 3) - 1.0 for o in origins})
    tampered = base.copy()
    tampered.loc[tampered["origin"] >= 9, "y_true"] = 500.0  # errors that only become known at origin >= 11
    at_origin_10_base = add_conformal_intervals(base, horizon, min_errors=min_errors).query("origin == 10")
    at_origin_10_tampered = add_conformal_intervals(tampered, horizon, min_errors=min_errors).query("origin == 10")
    # origin 10 may use origins <= 8 only, so tampering with origins 9 and 10 cannot change its bounds
    assert at_origin_10_base["lo"].tolist() == at_origin_10_tampered["lo"].tolist()
    assert at_origin_10_base["hi"].tolist() == at_origin_10_tampered["hi"].tolist()


def test_intervals_are_nan_until_enough_errors_have_accumulated():
    # 5 origins x 2 divisions = 10 errors in total, so 20 usable errors can never be reached
    result = add_conformal_intervals(_errors_frame({o: 0.5 for o in range(1, 6)}), horizon=2, min_errors=20)
    assert result["lo"].isna().all() and result["covered"].isna().all()


def test_intervals_appear_as_soon_as_enough_errors_have_accumulated():
    # origin 5 may use origins <= 3: 3 origins x 2 divisions = 6 errors
    result = add_conformal_intervals(_errors_frame({o: 0.5 for o in range(1, 6)}), horizon=2, min_errors=6)
    assert result.query("origin == 5")["lo"].notna().all()
    assert result.query("origin == 4")["lo"].isna().all()


def test_intervals_reach_roughly_nominal_coverage_on_stationary_errors():
    rng = np.random.default_rng(0)
    origins = list(range(1, 201))
    frame = pd.DataFrame(
        [{"origin": o, "division": d, "y_true": rng.normal(), "y_pred": 0.0} for o in origins for d in ("A", "B", "C")]
    )
    result = add_conformal_intervals(frame, horizon=2, level=0.8, min_errors=40)
    assert result["covered"].dropna().astype(float).mean() == pytest.approx(0.8, abs=0.06)
