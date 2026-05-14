
import pandas as pd
import numpy as np
from typing import Optional
from sklearn.ensemble import RandomForestRegressor


MODEL_LABELS = {
    "smart_ensemble": "Smart Ensemble",
    "moving_average": "Moving Average",
    "linear_trend": "Linear Trend",
    "seasonal_naive": "Seasonal Naive",
    "seasonal_trend": "Seasonal Trend",
    "exp_smoothing": "Exponential Smoothing",
    "holts": "Holt's Double Smoothing",
    "random_forest": "Random Forest",
}

DEFAULT_MODEL_KWARGS = {
    "smart_ensemble": {},
    "moving_average": {"window": 3},
    "linear_trend": {},
    "seasonal_naive": {"season": 12},
    "seasonal_trend": {"season": 12},
    "exp_smoothing": {"alpha": 0.3},
    "holts": {"alpha": 0.3, "beta": 0.1},
    "random_forest": {
        "n_estimators": 120,
        "max_depth": 8,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42,
        "n_jobs": 1,
    },
}

MODEL_FALLBACK_ORDER = [
    "smart_ensemble",
    "holts",
    "exp_smoothing",
    "linear_trend",
    "seasonal_trend",
    "moving_average",
    "seasonal_naive",
]


def prepare_monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily data to monthly totals."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    monthly = df.groupby(df["Date"].dt.to_period("M"))["Sales"].sum().reset_index()
    monthly["Date"] = monthly["Date"].dt.to_timestamp()
    return monthly.sort_values("Date").reset_index(drop=True)


def get_model_label(model_key: str) -> str:
    return MODEL_LABELS.get(model_key, model_key.replace("_", " ").title())


def get_default_model_registry(series: Optional[pd.DataFrame] = None,
                               include_random_forest: bool = True) -> dict:
    registry = {
        "smart_ensemble": (smart_ensemble_forecast, DEFAULT_MODEL_KWARGS["smart_ensemble"].copy()),
        "moving_average": (moving_average, DEFAULT_MODEL_KWARGS["moving_average"].copy()),
        "linear_trend": (linear_trend, DEFAULT_MODEL_KWARGS["linear_trend"].copy()),
        "seasonal_trend": (seasonal_trend, DEFAULT_MODEL_KWARGS["seasonal_trend"].copy()),
        "exp_smoothing": (exp_smoothing, DEFAULT_MODEL_KWARGS["exp_smoothing"].copy()),
        "holts": (holts_double, DEFAULT_MODEL_KWARGS["holts"].copy()),
    }
    if include_random_forest and series is not None and len(series) >= 8:
        registry["random_forest"] = (
            random_forest_forecast,
            DEFAULT_MODEL_KWARGS["random_forest"].copy(),
        )
    return registry


def moving_average(series: pd.DataFrame, periods: int,
                   window: int = 3, ci_z: float = 1.96,
                   trend_weight: float = 0.35,
                   damping: float = 0.85) -> pd.DataFrame:
    values = series["Sales"].values
    ma_val = np.mean(values[-window:])
    recent_trend = _recent_monthly_trend(values)
    damped_steps = np.array([
        sum(damping ** step for step in range(horizon))
        for horizon in range(1, periods + 1)
    ], dtype=float)
    forecast = ma_val + trend_weight * recent_trend * damped_steps
    forecast = np.maximum(forecast, 0)
    std = np.std(values[-window:])
    upper = forecast + ci_z * std
    lower = np.maximum(forecast - ci_z * std, 0)
    dates = pd.date_range(series["Date"].iloc[-1] + pd.DateOffset(months=1),
                          periods=periods, freq="MS")
    return pd.DataFrame({
        "Date": dates,
        "Forecast": forecast,
        "Upper": upper,
        "Lower": lower,
    })


def linear_trend(series: pd.DataFrame, periods: int,
                 ci_z: float = 1.96) -> pd.DataFrame:
    x = np.arange(len(series))
    y = series["Sales"].values
    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)
    residuals = y - poly(x)
    std = np.std(residuals)
    x_future = np.arange(len(series), len(series) + periods)
    forecast = poly(x_future)
    upper = forecast + ci_z * std
    lower = np.maximum(forecast - ci_z * std, 0)
    dates = pd.date_range(series["Date"].iloc[-1] + pd.DateOffset(months=1),
                          periods=periods, freq="MS")
    return pd.DataFrame({
        "Date": dates,
        "Forecast": forecast,
        "Upper": upper,
        "Lower": lower,
    })


def seasonal_naive(series: pd.DataFrame, periods: int,
                   season: int = 12, ci_z: float = 1.96) -> pd.DataFrame:
    if len(series) >= season:
        return seasonal_trend(series, periods, season=season, ci_z=ci_z, trend_strength=0.6)

    values = series["Sales"].values
    forecast = [values[max(len(values) - season + (i % season), 0)]
                for i in range(periods)]
    forecast = np.array(forecast)
    std = np.std(np.diff(values[-season:])) if len(values) >= season else np.std(values)
    upper = forecast + ci_z * std
    lower = np.maximum(forecast - ci_z * std, 0)
    dates = pd.date_range(series["Date"].iloc[-1] + pd.DateOffset(months=1),
                          periods=periods, freq="MS")
    return pd.DataFrame({
        "Date": dates,
        "Forecast": forecast,
        "Upper": upper,
        "Lower": lower,
    })


def _seasonal_growth_factor(values: np.ndarray, season: int = 12) -> float:
    if len(values) < season * 2:
        return 1.0

    current = np.asarray(values[-season:], dtype=float)
    previous = np.asarray(values[-season * 2:-season], dtype=float)
    valid = previous > 0
    if not valid.any():
        return 1.0

    ratios = current[valid] / previous[valid]
    ratios = ratios[np.isfinite(ratios)]
    if len(ratios) == 0:
        return 1.0

    return float(np.clip(np.median(ratios), 0.65, 1.45))


def seasonal_trend(series: pd.DataFrame, periods: int,
                   season: int = 12, ci_z: float = 1.96,
                   trend_strength: float = 1.0) -> pd.DataFrame:
    values = series["Sales"].values
    if len(values) < season:
        return linear_trend(series, periods, ci_z=ci_z)

    base = np.array([
        values[max(len(values) - season + (i % season), 0)]
        for i in range(periods)
    ], dtype=float)

    growth_factor = _seasonal_growth_factor(values, season=season)
    annual_growth = (growth_factor - 1.0) * trend_strength
    forecast = np.array([
        base_value * max(0.0, 1.0 + annual_growth * ((i // season) + 1))
        for i, base_value in enumerate(base)
    ], dtype=float)

    if len(values) >= season * 2:
        fitted = values[-season * 2:-season] * growth_factor
        actual = values[-season:]
        std = float(np.std(actual - fitted))
    else:
        std = float(np.std(np.diff(values[-season:])))

    upper = forecast + ci_z * std
    lower = np.maximum(forecast - ci_z * std, 0)
    dates = pd.date_range(series["Date"].iloc[-1] + pd.DateOffset(months=1),
                          periods=periods, freq="MS")
    return pd.DataFrame({
        "Date": dates,
        "Forecast": forecast,
        "Upper": upper,
        "Lower": lower,
    })


def exp_smoothing(series: pd.DataFrame, periods: int,
                  alpha: float = 0.3, trend_weight: float = 0.45,
                  damping: float = 0.88, ci_z: float = 1.96) -> pd.DataFrame:
    values = series["Sales"].values
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])

    recent_trend = _recent_monthly_trend(values)
    smoothed_trend = _recent_monthly_trend(np.array(smoothed))
    trend = trend_weight * recent_trend + (1 - trend_weight) * smoothed_trend
    future_steps = np.arange(1, periods + 1, dtype=float)
    damped_steps = np.array([
        sum(damping ** step for step in range(int(horizon)))
        for horizon in future_steps
    ])
    forecast = smoothed[-1] + trend * damped_steps

    residuals = values - np.array(smoothed)
    std = np.std(residuals)
    upper = forecast + ci_z * std
    lower = np.maximum(forecast - ci_z * std, 0)
    dates = pd.date_range(series["Date"].iloc[-1] + pd.DateOffset(months=1),
                          periods=periods, freq="MS")
    return pd.DataFrame({
        "Date": dates,
        "Forecast": forecast,
        "Upper": upper,
        "Lower": lower,
    })


def _recent_monthly_trend(values: np.ndarray, max_window: int = 6) -> float:
    """Estimate the latest month-to-month slope from the recent sales path."""
    if len(values) < 2:
        return 0.0

    recent = np.asarray(values[-min(max_window, len(values)):], dtype=float)
    if len(recent) < 2:
        return 0.0

    x = np.arange(len(recent), dtype=float)
    slope = float(np.polyfit(x, recent, 1)[0])
    return slope if np.isfinite(slope) else 0.0


def _ensure_forecast_has_movement(forecast: np.ndarray,
                                  history_values: np.ndarray,
                                  min_periods: int = 3,
                                  trend_weight: float = 0.25,
                                  damping: float = 0.82) -> np.ndarray:
    forecast = np.asarray(forecast, dtype=float)
    if len(forecast) < min_periods or len(history_values) < 2:
        return np.maximum(forecast, 0)

    if np.unique(np.round(forecast, 2)).size > 1:
        return np.maximum(forecast, 0)

    recent_trend = _recent_monthly_trend(history_values)
    flat_threshold = max(float(np.mean(np.abs(history_values))) * 0.001, 1e-9)
    if not np.isfinite(recent_trend) or abs(recent_trend) <= flat_threshold:
        return np.maximum(forecast, 0)

    damped_steps = np.array([
        sum(damping ** step for step in range(horizon))
        for horizon in range(1, len(forecast) + 1)
    ], dtype=float)
    adjusted = forecast + trend_weight * recent_trend * damped_steps
    return np.maximum(adjusted, 0)


def holts_double(series: pd.DataFrame, periods: int,
                 alpha: float = 0.3, beta: float = 0.1,
                 ci_z: float = 1.96) -> pd.DataFrame:
    if len(series) < 2:
        raise ValueError("Holt's model requires at least 2 monthly observations.")

    values = series["Sales"].values
    level = values[0]
    trend = values[1] - values[0]
    levels, trends = [level], [trend]

    for v in values[1:]:
        pl, pt = levels[-1], trends[-1]
        l = alpha * v + (1 - alpha) * (pl + pt)
        t = beta * (l - pl) + (1 - beta) * pt
        levels.append(l)
        trends.append(t)

    forecast_trend = trends[-1]
    recent_trend = _recent_monthly_trend(values)
    flat_threshold = max(float(np.mean(np.abs(values))) * 0.001, 1e-9)
    if abs(forecast_trend) <= flat_threshold and abs(recent_trend) > flat_threshold:
        forecast_trend = recent_trend

    forecast = np.array([levels[-1] + (i + 1) * forecast_trend for i in range(periods)])
    fitted = np.array([levels[i] + trends[i] for i in range(len(values))])
    std = np.std(values - fitted)
    upper = forecast + ci_z * std
    lower = np.maximum(forecast - ci_z * std, 0)
    dates = pd.date_range(series["Date"].iloc[-1] + pd.DateOffset(months=1),
                          periods=periods, freq="MS")
    return pd.DataFrame({
        "Date": dates,
        "Forecast": forecast,
        "Upper": upper,
        "Lower": lower,
    })


def _validation_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs((actual - predicted) / np.maximum(np.abs(actual), 1))) * 100)


def smart_ensemble_forecast(series: pd.DataFrame,
                            periods: int,
                            ci_z: float = 1.96,
                            top_n: int = 3) -> pd.DataFrame:
    """
    Blend the strongest recent forecast models using inverse-error weights.

    The ensemble validates candidate models on the latest holdout window, then
    forecasts with the best performers on the full history and blends their
    paths. This keeps the forecast responsive without making one model carry
    every business decision.
    """
    series = series.copy()
    series["Date"] = pd.to_datetime(series["Date"])
    series = series.sort_values("Date").reset_index(drop=True)

    if len(series) < 6:
        return holts_double(series, periods, ci_z=ci_z)

    holdout_size = max(2, min(6, len(series) // 4))
    train = series.iloc[:-holdout_size].copy()
    test = series.iloc[-holdout_size:].copy()

    candidate_models = [
        ("holts", holts_double, DEFAULT_MODEL_KWARGS["holts"].copy()),
        ("linear_trend", linear_trend, DEFAULT_MODEL_KWARGS["linear_trend"].copy()),
        ("exp_smoothing", exp_smoothing, DEFAULT_MODEL_KWARGS["exp_smoothing"].copy()),
        ("moving_average", moving_average, DEFAULT_MODEL_KWARGS["moving_average"].copy()),
    ]
    if len(series) >= 12:
        candidate_models.append(("seasonal_trend", seasonal_trend, DEFAULT_MODEL_KWARGS["seasonal_trend"].copy()))

    scored_models = []
    for model_key, fn, kwargs in candidate_models:
        kwargs["ci_z"] = ci_z
        try:
            validation_df = fn(train, len(test), **kwargs)
            mape = _validation_mape(test["Sales"].to_numpy(), validation_df["Forecast"].to_numpy())
            future_df = fn(series, periods, **kwargs)
        except Exception:
            continue

        if future_df.empty or not np.isfinite(mape):
            continue
        scored_models.append((model_key, max(mape, 0.01), future_df))

    if not scored_models:
        return holts_double(series, periods, ci_z=ci_z)

    scored_models = sorted(scored_models, key=lambda item: item[1])[:max(1, top_n)]
    raw_weights = np.array([1.0 / (mape + 1.0) for _, mape, _ in scored_models], dtype=float)
    weights = raw_weights / raw_weights.sum()

    forecast_stack = np.vstack([
        model_df["Forecast"].to_numpy(dtype=float)
        for _, _, model_df in scored_models
    ])
    forecast = np.average(forecast_stack, axis=0, weights=weights)
    forecast = _ensure_forecast_has_movement(
        forecast,
        series["Sales"].to_numpy(dtype=float),
        trend_weight=0.18,
        damping=0.80,
    )

    upper_stack = np.vstack([
        model_df["Upper"].to_numpy(dtype=float)
        for _, _, model_df in scored_models
    ])
    lower_stack = np.vstack([
        model_df["Lower"].to_numpy(dtype=float)
        for _, _, model_df in scored_models
    ])
    upper = np.maximum(np.average(upper_stack, axis=0, weights=weights), forecast)
    lower = np.minimum(np.average(lower_stack, axis=0, weights=weights), forecast)
    lower = np.maximum(lower, 0)

    dates = scored_models[0][2]["Date"].reset_index(drop=True)
    result = pd.DataFrame({
        "Date": dates,
        "Forecast": np.maximum(forecast, 0),
        "Upper": upper,
        "Lower": lower,
    })
    result.attrs["ensemble_models"] = [
        {"model_key": model_key, "validation_mape": mape, "weight": float(weight)}
        for (model_key, mape, _), weight in zip(scored_models, weights)
    ]
    return result


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ML-ready time series features.

    Features:
    - lag_1, lag_2, lag_3
    - rolling_mean_3
    - rolling_std_3
    - month, quarter, year
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["lag_1"] = df["Sales"].shift(1)
    df["lag_2"] = df["Sales"].shift(2)
    df["lag_3"] = df["Sales"].shift(3)

    df["rolling_mean_3"] = df["Sales"].shift(1).rolling(3).mean()
    df["rolling_std_3"] = df["Sales"].shift(1).rolling(3).std()

    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df["year"] = df["Date"].dt.year

    return df


def prepare_supervised_series(series: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a monthly series into a supervised ML training set.
    Rows with incomplete lag/rolling features are dropped.
    """
    feature_df = create_time_features(series)
    feature_df = feature_df.dropna().reset_index(drop=True)
    return feature_df


def _build_future_feature_row(history_df: pd.DataFrame, future_date: pd.Timestamp) -> pd.DataFrame:
    """
    Build the next feature row from the rolling history for recursive forecasting.
    """
    sales_history = history_df["Sales"].tolist()
    if len(sales_history) < 3:
        raise ValueError("Random Forest forecast requires at least 6 monthly observations to build stable lag features.")

    last3 = sales_history[-3:]
    return pd.DataFrame([{
        "lag_1": sales_history[-1],
        "lag_2": sales_history[-2],
        "lag_3": sales_history[-3],
        "rolling_mean_3": float(np.mean(last3)),
        "rolling_std_3": float(np.std(last3, ddof=0)),
        "month": future_date.month,
        "quarter": future_date.quarter,
        "year": future_date.year,
    }])


def random_forest_forecast(series: pd.DataFrame,
                           periods: int,
                           n_estimators: int = 120,
                           max_depth: int | None = 8,
                           min_samples_split: int = 2,
                           min_samples_leaf: int = 1,
                           random_state: int = 42,
                           n_jobs: int = 1,
                           ci_z: float = 1.96) -> pd.DataFrame:
    """
    Forecast future sales using RandomForestRegressor on lag/time features.

    This uses recursive forecasting:
    1. Train on historical monthly data with engineered features
    2. Predict next month
    3. Append prediction to history
    4. Repeat until requested forecast horizon is complete
    """
    series = series.copy()
    series["Date"] = pd.to_datetime(series["Date"])
    series = series.sort_values("Date").reset_index(drop=True)

    if len(series) < 6:
        raise ValueError(
            "Random Forest forecasting requires at least 6 monthly observations."
        )

    supervised_df = prepare_supervised_series(series)
    if supervised_df.empty:
        raise ValueError(
            "Not enough data to train Random Forest after feature engineering."
        )

    feature_cols = [
        "lag_1", "lag_2", "lag_3",
        "rolling_mean_3", "rolling_std_3",
        "month", "quarter", "year",
    ]
    X_train = supervised_df[feature_cols]
    y_train = supervised_df["Sales"]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    residual_std = float(np.std(y_train - train_pred))

    history_df = series.copy()
    future_dates = pd.date_range(
        history_df["Date"].iloc[-1] + pd.DateOffset(months=1),
        periods=periods,
        freq="MS",
    )

    forecasts = []
    for future_date in future_dates:
        future_X = _build_future_feature_row(history_df, future_date)
        pred = float(model.predict(future_X)[0])
        pred = max(pred, 0.0)
        forecasts.append(pred)

        history_df = pd.concat(
            [
                history_df,
                pd.DataFrame({"Date": [future_date], "Sales": [pred]}),
            ],
            ignore_index=True,
        )

    forecast = np.array(forecasts)
    forecast = _ensure_forecast_has_movement(
        forecast,
        series["Sales"].to_numpy(dtype=float),
        trend_weight=0.20,
        damping=0.78,
    )
    upper = forecast + ci_z * residual_std
    lower = np.maximum(forecast - ci_z * residual_std, 0)

    return pd.DataFrame({
        "Date": future_dates,
        "Forecast": forecast,
        "Upper": upper,
        "Lower": lower,
    })


MODEL_REGISTRY = {
    "smart_ensemble": smart_ensemble_forecast,
    "moving_average": moving_average,
    "linear_trend": linear_trend,
    "seasonal_naive": seasonal_naive,
    "seasonal_trend": seasonal_trend,
    "exp_smoothing": exp_smoothing,
    "holts": holts_double,
    "random_forest": random_forest_forecast,
}


def run_forecast(series: pd.DataFrame, model_key: str,
                 periods: int, **kwargs) -> pd.DataFrame:
    """Run a forecast by model key."""
    fn = MODEL_REGISTRY.get(model_key)
    if fn is None:
        raise ValueError(
            f"Unknown model: {model_key}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return fn(series, periods, **kwargs)


def run_forecast_with_fallback(series: pd.DataFrame,
                               periods: int,
                               preferred_model: str = "holts",
                               ci_z: float = 1.96,
                               model_kwargs: Optional[dict] = None) -> tuple[pd.DataFrame, str]:
    """
    Run a forecast with a preferred model and safe fallbacks.

    This is useful for grouped/category chatbot forecasts where some groups may
    have less history than the main dataset. The first successful model is used.
    """
    ordered_models = [preferred_model] + [
        key for key in MODEL_FALLBACK_ORDER if key != preferred_model
    ]

    last_error: Exception | None = None
    for model_key in ordered_models:
        kwargs = DEFAULT_MODEL_KWARGS.get(model_key, {}).copy()
        if model_key == preferred_model and model_kwargs:
            kwargs.update(model_kwargs)
        kwargs["ci_z"] = ci_z

        try:
            return run_forecast(series, model_key, periods, **kwargs), model_key
        except Exception as exc:
            last_error = exc

    raise ValueError(f"Unable to forecast with available models: {last_error}")


def compare_forecast_models(series: pd.DataFrame,
                            include_random_forest: bool = True) -> pd.DataFrame:
    """Compare supported forecast models on the supplied monthly series."""
    from services.evaluation_service import compare_all_models

    registry = get_default_model_registry(
        series=series,
        include_random_forest=include_random_forest,
    )
    return compare_all_models(series, registry)


def rolling_backtest_forecast_models(series: pd.DataFrame,
                                     min_train_size: int = 6,
                                     horizon: int = 1,
                                     max_splits: int = 8,
                                     include_random_forest: bool = True) -> pd.DataFrame:
    """Run expanding-window backtests for all supported models."""
    from services.evaluation_service import rolling_backtest_all_models

    registry = get_default_model_registry(
        series=series,
        include_random_forest=include_random_forest,
    )
    return rolling_backtest_all_models(
        series,
        registry,
        min_train_size=min_train_size,
        horizon=horizon,
        max_splits=max_splits,
    )


def select_best_forecast_model(series: pd.DataFrame,
                               fallback_model: str = "holts",
                               include_random_forest: bool = True) -> tuple[str, pd.DataFrame]:
    """Return the best model key by recent validation, with a safe fallback."""
    try:
        comparison_df = compare_forecast_models(series, include_random_forest=include_random_forest)
    except Exception:
        return fallback_model, pd.DataFrame()

    if comparison_df is None or comparison_df.empty or "model_key" not in comparison_df.columns:
        return fallback_model, pd.DataFrame()

    if len(series) >= 10:
        try:
            backtest_df = rolling_backtest_forecast_models(
                series,
                min_train_size=max(6, min(12, len(series) // 2)),
                horizon=1,
                max_splits=6,
                include_random_forest=include_random_forest,
            )
            if backtest_df is not None and not backtest_df.empty and "model_key" in backtest_df.columns:
                return str(backtest_df.iloc[0]["model_key"]), backtest_df
        except Exception:
            pass

    return str(comparison_df.iloc[0]["model_key"]), comparison_df


def forecast_with_best_model(series: pd.DataFrame,
                             periods: int,
                             ci_z: float = 1.96,
                             include_random_forest: bool = True) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    """Select the best model for a series and return forecast, model key, and comparison."""
    best_model_key, comparison_df = select_best_forecast_model(
        series,
        include_random_forest=include_random_forest,
    )
    forecast_df, model_key = run_forecast_with_fallback(
        series,
        periods,
        preferred_model=best_model_key,
        ci_z=ci_z,
    )
    return forecast_df, model_key, comparison_df


def forecast_grouped_series(df: pd.DataFrame,
                            group_col: str,
                            periods: int,
                            max_groups: int = 8,
                            ci_z: float = 1.96) -> list[dict]:
    """
    Forecast each group independently using that group's best model.

    Returns one dict per group with forecast_df, model_key, comparison_df, and
    total_forecast. Groups with too little monthly history are skipped.
    """
    if group_col not in df.columns:
        raise ValueError(f"Column not found: {group_col}")

    grouped_sales = df.groupby(group_col)["Sales"].sum().sort_values(ascending=False)
    selected_groups = grouped_sales.head(max_groups).index.tolist()
    results: list[dict] = []

    for group_value in selected_groups:
        group_df = df[df[group_col] == group_value].copy()
        monthly_group = prepare_monthly_series(group_df)
        if len(monthly_group) < 3:
            continue

        forecast_df, model_key, comparison_df = forecast_with_best_model(
            monthly_group,
            periods,
            ci_z=ci_z,
            include_random_forest=False,
        )
        results.append({
            "group": group_value,
            "forecast_df": forecast_df,
            "model_key": model_key,
            "comparison_df": comparison_df,
            "total_forecast": float(forecast_df["Forecast"].sum()),
        })

    return results


def get_random_forest_feature_importance(series: pd.DataFrame) -> pd.DataFrame:
    """Train the shared Random Forest feature pipeline and return importance rows."""
    rf_df = prepare_supervised_series(series)
    if len(rf_df) < 4:
        return pd.DataFrame(columns=["Feature", "Importance", "Feature Type"])

    feature_cols = [
        "lag_1", "lag_2", "lag_3",
        "rolling_mean_3", "rolling_std_3",
        "month", "quarter", "year",
    ]

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=8,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(rf_df[feature_cols], rf_df["Sales"])

    feature_labels = {
        "lag_1": "Previous month sales",
        "lag_2": "Sales two months ago",
        "lag_3": "Sales three months ago",
        "rolling_mean_3": "3-month sales average",
        "rolling_std_3": "3-month volatility",
        "month": "Month of year",
        "quarter": "Quarter",
        "year": "Year trend",
    }
    feature_types = {
        "lag_1": "Lag feature",
        "lag_2": "Lag feature",
        "lag_3": "Lag feature",
        "rolling_mean_3": "Lag feature",
        "rolling_std_3": "Lag feature",
        "month": "Calendar feature",
        "quarter": "Calendar feature",
        "year": "Calendar feature",
    }

    importance_df = pd.DataFrame({
        "Feature": [feature_labels[col] for col in feature_cols],
        "Importance": model.feature_importances_,
        "Feature Type": [feature_types[col] for col in feature_cols],
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    importance_df["Importance"] = (importance_df["Importance"] * 100).round(1)
    return importance_df
