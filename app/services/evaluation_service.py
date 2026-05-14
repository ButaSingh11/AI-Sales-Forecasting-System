
import numpy as np
import pandas as pd
from typing import Callable, Optional


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Compute MAE, RMSE, MAPE, and R² between actual and predicted arrays.
    """
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)

    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mape = float(
        np.mean(np.abs((actual - predicted) / np.maximum(np.abs(actual), 1))) * 100
    )
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape, "R²": r2}


def accuracy_rating(mape: float) -> tuple:
    """Return (label, css_class) based on MAPE value."""
    if mape < 10:
        return "Excellent", "acc-good"
    elif mape < 20:
        return "Good", "acc-ok"
    elif mape < 50:
        return "Acceptable", "acc-ok"
    else:
        return "Poor", "acc-bad"


def train_test_split_series(series: pd.DataFrame,
                            test_ratio: float = 0.2) -> tuple:
    """Split a time series DataFrame into train and test sets."""
    n = len(series)
    split = max(int(n * (1 - test_ratio)), 3)
    return series.iloc[:split].copy(), series.iloc[split:].copy()


def evaluate_model(series: pd.DataFrame,
                   forecast_fn: Callable,
                   fn_kwargs: Optional[dict] = None,
                   test_ratio: float = 0.2) -> Optional[dict]:
    """
    Evaluate a forecasting function on a hold-out test set.

    Parameters
    ----------
    series      : Full monthly series DataFrame with 'Date' and 'Sales'
    forecast_fn : Forecasting function (takes series, periods, **kwargs)
    fn_kwargs   : Extra keyword arguments for the forecast function
    test_ratio  : Fraction of data to use as test set

    Returns
    -------
    dict with MAE, RMSE, MAPE, R² or None if insufficient data
    """
    fn_kwargs = fn_kwargs or {}
    train, test = train_test_split_series(series, test_ratio)

    if len(test) < 2:
        return None

    pred_df = forecast_fn(train, len(test), **fn_kwargs)
    return compute_metrics(test["Sales"].values, pred_df["Forecast"].values)


def compare_all_models(series: pd.DataFrame,
                       model_registry: dict) -> pd.DataFrame:
    """
    Evaluate all models in the registry and return a sorted comparison DataFrame.

    Parameters
    ----------
    series         : Full monthly series
    model_registry : Dict of {model_key: function} or {model_key: (fn, kwargs)}

    Returns
    -------
    DataFrame sorted by MAPE ascending
    """
    rows = []
    for key, model_value in model_registry.items():
        if isinstance(model_value, tuple):
            fn, kwargs = model_value
        else:
            fn, kwargs = model_value, {}

        metrics = evaluate_model(series, fn, kwargs)
        if metrics:
            acc_label, _ = accuracy_rating(metrics["MAPE (%)"])
            rows.append({
                "model_key": key,
                "MAE": round(metrics["MAE"], 2),
                "RMSE": round(metrics["RMSE"], 2),
                "MAPE (%)": round(metrics["MAPE (%)"], 2),
                "R²": round(metrics["R²"], 3),
                "Accuracy": acc_label,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("MAPE (%)").reset_index(drop=True)
    return df


def rolling_backtest_model(series: pd.DataFrame,
                           forecast_fn: Callable,
                           fn_kwargs: Optional[dict] = None,
                           min_train_size: int = 6,
                           horizon: int = 1,
                           max_splits: int = 8) -> Optional[dict]:
    """
    Expanding-window backtest for a forecasting function.

    The model is repeatedly trained on history up to a point and asked to
    forecast the next horizon. Results are averaged across windows.
    """
    fn_kwargs = fn_kwargs or {}
    series = series.sort_values("Date").reset_index(drop=True)

    if len(series) < min_train_size + horizon:
        return None

    starts = list(range(min_train_size, len(series) - horizon + 1))
    if len(starts) > max_splits:
        starts = starts[-max_splits:]

    actual_values = []
    predicted_values = []
    split_count = 0

    for split_idx in starts:
        train = series.iloc[:split_idx].copy()
        test = series.iloc[split_idx:split_idx + horizon].copy()
        try:
            pred_df = forecast_fn(train, len(test), **fn_kwargs)
        except Exception:
            continue

        actual_values.extend(test["Sales"].to_numpy(dtype=float))
        predicted_values.extend(pred_df["Forecast"].to_numpy(dtype=float))
        split_count += 1

    if split_count == 0:
        return None

    metrics = compute_metrics(np.array(actual_values), np.array(predicted_values))
    metrics["Splits"] = split_count
    metrics["Horizon"] = horizon
    return metrics


def rolling_backtest_all_models(series: pd.DataFrame,
                                model_registry: dict,
                                min_train_size: int = 6,
                                horizon: int = 1,
                                max_splits: int = 8) -> pd.DataFrame:
    rows = []
    for key, model_value in model_registry.items():
        if isinstance(model_value, tuple):
            fn, kwargs = model_value
        else:
            fn, kwargs = model_value, {}

        metrics = rolling_backtest_model(
            series,
            fn,
            kwargs,
            min_train_size=min_train_size,
            horizon=horizon,
            max_splits=max_splits,
        )
        if metrics:
            r2_value = metrics.get("R2", metrics.get("R²", 0.0))
            rows.append({
                "model_key": key,
                "MAE": round(metrics["MAE"], 2),
                "RMSE": round(metrics["RMSE"], 2),
                "MAPE (%)": round(metrics["MAPE (%)"], 2),
                "R²": round(r2_value, 3),
                "Splits": metrics["Splits"],
                "Horizon": metrics["Horizon"],
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("MAPE (%)").reset_index(drop=True)
    return df


def get_best_model(model_comparison_df: pd.DataFrame) -> Optional[dict]:
    """
    Return the best model recommendation from a model comparison DataFrame.

    Expected columns:
    - model_key
    - MAPE (%)
    Optional:
    - Accuracy

    Returns
    -------
    dict like:
    {
        "model_key": "holts",
        "mape": 8.73,
        "accuracy_label": "Excellent"
    }
    """
    if model_comparison_df is None or model_comparison_df.empty:
        return None

    best_row = model_comparison_df.sort_values("MAPE (%)", ascending=True).iloc[0]
    accuracy_label = best_row["Accuracy"] if "Accuracy" in model_comparison_df.columns else accuracy_rating(best_row["MAPE (%)"])[0]

    return {
        "model_key": best_row["model_key"],
        "mape": float(best_row["MAPE (%)"]),
        "accuracy_label": accuracy_label,
    }


def forecast_confidence_score(mape: float,
                              volatility: float,
                              data_points: int,
                              anomaly_count: int) -> dict:
    """
    Estimate forecast reliability using a simple weighted scoring approach.

    Parameters
    ----------
    mape         : Forecast error percentage
    volatility   : Coefficient of variation or similar variability measure (%)
    data_points  : Number of historical observations
    anomaly_count: Number of anomalies detected in the history

    Returns
    -------
    dict like:
    {
        "score": 78,
        "level": "High",
        "reason": "Low error, stable series, sufficient history, few anomalies."
    }
    """
    score = 100

    # Penalize forecast error
    if mape < 10:
        score -= 0
    elif mape < 20:
        score -= 10
    elif mape < 35:
        score -= 25
    else:
        score -= 40

    # Penalize volatility
    if volatility < 10:
        score -= 0
    elif volatility < 20:
        score -= 8
    elif volatility < 35:
        score -= 18
    else:
        score -= 30

    # Penalize limited historical data
    if data_points >= 24:
        score -= 0
    elif data_points >= 12:
        score -= 10
    elif data_points >= 6:
        score -= 22
    else:
        score -= 35

    # Penalize anomaly count
    if anomaly_count == 0:
        score -= 0
    elif anomaly_count <= 2:
        score -= 5
    elif anomaly_count <= 5:
        score -= 12
    else:
        score -= 20

    score = max(0, min(100, int(round(score))))

    if score >= 75:
        level = "High"
    elif score >= 50:
        level = "Moderate"
    else:
        level = "Low"

    reasons = []
    reasons.append("low error" if mape < 20 else "forecast error is relatively high")
    reasons.append("stable series" if volatility < 20 else "series is volatile")
    reasons.append("sufficient history" if data_points >= 12 else "limited history")
    reasons.append("few anomalies" if anomaly_count <= 2 else "multiple anomalies detected")

    return {
        "score": score,
        "level": level,
        "reason": ", ".join(reasons).capitalize() + ".",
    }
