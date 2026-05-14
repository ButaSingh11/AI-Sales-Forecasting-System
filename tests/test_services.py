import numpy as np
import pandas as pd

from services.chatbot_service import answer_from_dataframe, build_hybrid_chat_response, detect_query_intent
from services.evaluation_service import (
    compare_all_models,
    forecast_confidence_score,
    get_best_model,
    rolling_backtest_all_models,
)
from services.forecasting_service import (
    MODEL_REGISTRY,
    exp_smoothing,
    holts_double,
    linear_trend,
    moving_average,
    prepare_monthly_series,
    random_forest_forecast,
    seasonal_naive,
    seasonal_trend,
    smart_ensemble_forecast,
)
from services.insight_service import detect_anomalies
from services.model_service import calculate_regression_metrics, safe_mape
from services.preprocessing_service import clean_dataframe, clean_numeric_strings


def sample_daily_data(periods: int = 365) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", periods=periods, freq="D")
    sales = 1000 + np.linspace(0, 250, periods) + rng.normal(0, 75, periods)
    sales[periods // 2] = sales.mean() * 4
    return pd.DataFrame({"Date": dates, "Sales": sales})


def test_detect_anomalies_returns_dataframe():
    anomalies = detect_anomalies(sample_daily_data(), method="zscore", threshold=2.5)
    assert isinstance(anomalies, pd.DataFrame)
    assert "severity" in anomalies.columns


def test_get_best_model_returns_lowest_mape_row():
    df = pd.DataFrame({
        "model_key": ["a", "b", "c"],
        "MAPE (%)": [20.0, 10.0, 15.0],
        "Accuracy": ["Poor", "Good", "Acceptable"],
    })
    best = get_best_model(df)
    assert best["model_key"] == "b"
    assert best["mape"] == 10.0


def test_forecast_confidence_score_returns_expected_shape():
    score = forecast_confidence_score(
        mape=10,
        volatility=12,
        data_points=24,
        anomaly_count=1,
    )
    assert isinstance(score, dict)
    assert score["level"] in ["High", "Moderate", "Low"]
    assert "reason" in score


def test_random_forest_forecast_returns_requested_periods():
    monthly = prepare_monthly_series(sample_daily_data(periods=540))
    forecast = random_forest_forecast(monthly, periods=3)
    assert isinstance(forecast, pd.DataFrame)
    assert len(forecast) == 3
    assert "Forecast" in forecast.columns


def test_holts_uses_recent_trend_when_smoothed_trend_is_flat():
    monthly = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=6, freq="MS"),
        "Sales": [100, 100, 120, 140, 160, 180],
    })

    forecast = holts_double(monthly, periods=3, alpha=1.0, beta=0.0)

    assert forecast["Forecast"].is_monotonic_increasing
    assert forecast["Forecast"].nunique() == 3


def test_smart_ensemble_forecast_blends_multiple_models():
    monthly = pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=18, freq="MS"),
        "Sales": [1000, 1040, 1075, 1110, 1160, 1200, 1260, 1310, 1360, 1425, 1490, 1550, 1620, 1680, 1760, 1840, 1930, 2020],
    })

    forecast = smart_ensemble_forecast(monthly, periods=4)

    assert len(forecast) == 4
    assert forecast["Forecast"].nunique() > 1
    assert forecast.attrs["ensemble_models"]


def test_moving_average_adds_damped_recent_trend():
    monthly = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=8, freq="MS"),
        "Sales": [100, 105, 111, 118, 126, 135, 145, 156],
    })

    forecast = moving_average(monthly, periods=6, window=3)

    assert len(forecast) == 6
    assert forecast["Forecast"].nunique() > 1


def test_exp_smoothing_carries_damped_recent_trend():
    monthly = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=8, freq="MS"),
        "Sales": [100, 110, 125, 145, 170, 200, 235, 275],
    })

    forecast = exp_smoothing(monthly, periods=4, alpha=0.35)

    assert len(forecast) == 4
    assert forecast["Forecast"].nunique() == 4
    assert forecast["Forecast"].is_monotonic_increasing


def test_seasonal_trend_adjusts_last_year_values():
    first_year = [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320]
    second_year = [value * 1.25 for value in first_year]
    monthly = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=24, freq="MS"),
        "Sales": first_year + second_year,
    })

    forecast = seasonal_trend(monthly, periods=3)

    assert len(forecast) == 3
    assert forecast["Forecast"].iloc[0] > second_year[0]
    assert forecast["Forecast"].iloc[1] > second_year[1]


def test_seasonal_naive_no_longer_copies_previous_year_exactly():
    first_year = [100, 125, 150, 175, 210, 240, 260, 275, 290, 310, 330, 350]
    second_year = [value * 1.20 for value in first_year]
    monthly = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=24, freq="MS"),
        "Sales": first_year + second_year,
    })

    forecast = seasonal_naive(monthly, periods=6)
    last_year_same_months = np.array(second_year[:6], dtype=float)

    assert not np.allclose(forecast["Forecast"].to_numpy(dtype=float), last_year_same_months)
    assert (forecast["Forecast"].to_numpy(dtype=float) > last_year_same_months).all()


def test_planning_models_do_not_return_flat_paths_on_trending_data():
    monthly = pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=24, freq="MS"),
        "Sales": [
            100, 112, 128, 145, 168, 190, 215, 238, 260, 284, 306, 330,
            124, 140, 160, 182, 210, 238, 270, 300, 330, 362, 395, 430,
        ],
    })
    model_keys = [
        "smart_ensemble",
        "moving_average",
        "linear_trend",
        "seasonal_naive",
        "seasonal_trend",
        "exp_smoothing",
        "holts",
        "random_forest",
    ]

    for model_key in model_keys:
        forecast = MODEL_REGISTRY[model_key](monthly, periods=6)
        unique_values = np.unique(np.round(forecast["Forecast"].to_numpy(dtype=float), 2))
        assert len(forecast) == 6
        assert unique_values.size > 1, f"{model_key} returned a flat forecast path"


def test_chatbot_intent_detection():
    assert detect_query_intent("show trend") == "trend"
    assert detect_query_intent("detect anomaly") == "anomaly"
    assert detect_query_intent("forecast next month") == "forecast"


def test_safe_mape_and_regression_metrics():
    y_true = np.array([100, 200, 300], dtype=float)
    y_pred = np.array([110, 190, 310], dtype=float)

    mape = safe_mape(y_true, y_pred)
    metrics = calculate_regression_metrics(y_true, y_pred)

    assert mape > 0
    assert metrics["MAE"] > 0
    assert "RMSE" in metrics
    assert "MAPE (%)" in metrics


def test_compare_all_models_returns_sorted_results():
    monthly = prepare_monthly_series(sample_daily_data(periods=540))
    registry = {
        "moving_average": (moving_average, {"window": 3}),
        "linear_trend": (linear_trend, {}),
        "exp_smoothing": (exp_smoothing, {"alpha": 0.3}),
    }
    comparison_df = compare_all_models(monthly, registry)

    assert not comparison_df.empty
    assert list(comparison_df["MAPE (%)"]) == sorted(comparison_df["MAPE (%)"].tolist())


def test_rolling_backtest_all_models_returns_split_counts():
    monthly = prepare_monthly_series(sample_daily_data(periods=540))
    registry = {
        "moving_average": (moving_average, {"window": 3}),
        "linear_trend": (linear_trend, {}),
    }

    backtest_df = rolling_backtest_all_models(
        monthly,
        registry,
        min_train_size=6,
        horizon=1,
        max_splits=4,
    )

    assert not backtest_df.empty
    assert "Splits" in backtest_df.columns
    assert backtest_df["Splits"].min() > 0


def test_clean_dataframe_preserves_business_dimensions_by_default():
    df = pd.DataFrame({
        "Order Date": pd.date_range("2024-01-01", periods=4, freq="D"),
        "Revenue": ["₹1,000", "₹1,200", "₹900", "₹1,100"],
        "Category": ["Electronics", "Apparel", "Electronics", "Beauty"],
        "Region": ["North", "South", "North", "West"],
    })

    clean_df, _ = clean_dataframe(df, "Order Date", "Revenue")

    assert "Date" in clean_df.columns
    assert "Sales" in clean_df.columns
    assert "Category" in clean_df.columns
    assert "Region" in clean_df.columns
    assert "Category_Electronics" not in clean_df.columns
    assert clean_df["Sales"].sum() == 4200


def test_clean_numeric_strings_handles_rupee_symbol():
    df = pd.DataFrame({"Sales": ["₹1,000", "₹2.5L", "₹1.2Cr"]})
    clean_df = clean_numeric_strings(df)

    assert clean_df["Sales"].tolist() == [1000.0, 250000.0, 12000000.0]


def sample_category_monthly_data() -> pd.DataFrame:
    rows = []
    for idx, month in enumerate(pd.date_range("2023-01-01", periods=12, freq="MS")):
        rows.append({"Date": month, "Sales": 1000 + idx * 20, "Category": "Electronics"})
        rows.append({"Date": month, "Sales": 500 + idx * 10, "Category": "Apparel"})
    return pd.DataFrame(rows)


def test_chatbot_specific_category_forecast_uses_requested_horizon():
    result = answer_from_dataframe(
        "Forecast Electronics category for next 2 years",
        sample_category_monthly_data(),
    )

    assert result["intent"] == "forecast"
    assert "24 months (2 years)" in result["formatted_answer"]
    assert "Electronics Category Forecast" in result["formatted_answer"]
    assert "Forecast path is" in result["formatted_answer"]
    assert "Apparel" not in result["formatted_answer"]
    assert len(result["chart"]["data"]) == 24
    assert {"Lower", "Upper"}.issubset(result["table"][0].keys())


def test_chatbot_groupwise_category_forecast_has_separate_totals():
    result = answer_from_dataframe(
        "Forecast by category for next 2 years",
        sample_category_monthly_data(),
    )

    assert result["intent"] == "forecast"
    assert "24 months (2 years)" in result["formatted_answer"]

    chart_rows = result["chart"]["data"]
    labels = {row["label"] for row in chart_rows}
    totals = {row["sales"] for row in chart_rows}

    assert labels == {"Electronics", "Apparel"}
    assert len(totals) == 2


def test_chatbot_answers_model_questions_without_gemini():
    result = answer_from_dataframe(
        "Which model is best and what is MAPE?",
        sample_category_monthly_data(),
    )

    assert result["intent"] == "model_info"
    assert "Model Answer" in result["formatted_answer"]
    assert "MAPE" in result["formatted_answer"]


def test_chatbot_answers_data_quality_questions_without_gemini():
    result = answer_from_dataframe(
        "What columns and missing values are in my data?",
        sample_category_monthly_data(),
    )

    assert result["intent"] == "data_info"
    assert "Data Answer" in result["formatted_answer"]
    assert "Columns detected" in result["formatted_answer"]


def test_chatbot_answers_project_questions_without_gemini():
    result = answer_from_dataframe(
        "What does this project do and what modules are included?",
        sample_category_monthly_data(),
    )

    assert result["intent"] == "project_info"
    assert "Project Answer" in result["formatted_answer"]
    assert "Forecasting" in result["formatted_answer"]


def test_chatbot_gives_clear_general_answer_for_broad_questions():
    result = answer_from_dataframe(
        "help me understand what is going on",
        sample_category_monthly_data(),
    )

    assert result["intent"] == "general_help"
    assert "**Answer**" in result["formatted_answer"]
    assert "**Key details**" in result["formatted_answer"]
    assert "**What to do next**" in result["formatted_answer"]


def test_hybrid_chatbot_without_gemini_still_answers_broad_data_question():
    result = build_hybrid_chat_response(
        "help me understand what is going on",
        sample_category_monthly_data(),
        api_key=None,
    )

    assert result["mode"] == "direct_data"
    assert result["intent"] == "general_help"
    assert "Your data is loaded" in result["final_answer"]
