"""
train_model.py
--------------
Trains all forecasting models on the full dataset, evaluates them,
saves each model safely using model_service, and separately saves the best model.

Usage:
    python -m models.train_model --data data/raw/sample_sales.csv --periods 12
"""

import argparse
import sys
import os
import shutil
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.services.forecasting_service import (
    prepare_monthly_series,
    moving_average,
    linear_trend,
    seasonal_naive,
    exp_smoothing,
    holts_double,
    random_forest_forecast,
)
from app.services.evaluation_service import evaluate_model
from app.services.model_service import save_model


def train_all(data_path: str, forecast_periods: int = 12,
              save: bool = True) -> dict:
    """
    Train all available models on the dataset at data_path.
    Returns dict of:
        {model_key: {forecast_df, metrics, file_path}}

    Also identifies the best model by lowest MAPE and, if save=True,
    stores a separate best_model.pkl copy.
    """
    # Load data
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    series = prepare_monthly_series(df)

    models = {
        "moving_average": (moving_average, {"window": 3}),
        "linear_trend": (linear_trend, {}),
        "seasonal_naive": (seasonal_naive, {}),
        "exp_smoothing": (exp_smoothing, {"alpha": 0.3}),
        "holts": (holts_double, {"alpha": 0.3, "beta": 0.1}),
        "random_forest": (
            random_forest_forecast,
            {
                "n_estimators": 300,
                "max_depth": 8,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42,
            },
        ),
    }

    results = {}
    best_model_key = None
    best_mape = float("inf")
    best_file_path = None

    for key, (fn, kwargs) in models.items():
        print(f"Training {key}...")
        forecast_df = fn(series, forecast_periods, **kwargs)
        metrics = evaluate_model(series, fn, kwargs)

        file_path = None
        if save:
            file_path = save_model(
                model_key=key,
                kwargs=kwargs,
                model_name=key,
                metadata={
                    "periods": forecast_periods,
                    "metrics": metrics,
                    "data": data_path,
                },
            )
            print(f"  Saved → {file_path}")

        if metrics:
            print(f"  MAPE: {metrics['MAPE (%)']:.1f}%  R²: {metrics['R²']:.3f}")
            if metrics["MAPE (%)"] < best_mape:
                best_mape = metrics["MAPE (%)"]
                best_model_key = key
                best_file_path = file_path

        results[key] = {
            "forecast_df": forecast_df,
            "metrics": metrics,
            "file_path": file_path,
        }

    # Save best model separately
    if save and best_model_key and best_file_path:
        best_model_copy = os.path.join(os.path.dirname(best_file_path), "best_model.pkl")
        shutil.copyfile(best_file_path, best_model_copy)

        # copy metadata sidecar too if it exists
        src_meta = best_file_path.replace(".pkl", "_meta.json")
        dst_meta = best_model_copy.replace(".pkl", "_meta.json")
        if os.path.exists(src_meta):
            shutil.copyfile(src_meta, dst_meta)

        print("\n=== Best Model Selected ===")
        print(f"Best model : {best_model_key}")
        print(f"Best MAPE  : {best_mape:.2f}%")
        print(f"Saved copy : {best_model_copy}")

        results["best_model"] = {
            "model_key": best_model_key,
            "MAPE (%)": best_mape,
            "file_path": best_model_copy,
        }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all forecasting models")
    parser.add_argument("--data", default="data/raw/sample_sales.csv",
                        help="Path to the sales CSV file")
    parser.add_argument("--periods", type=int, default=12,
                        help="Number of months to forecast")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving models to disk")
    args = parser.parse_args()

    results = train_all(args.data, args.periods, save=not args.no_save)

    print("\n=== Training Complete ===")
    for key, r in results.items():
        if key == "best_model":
            continue
        m = r.get("metrics")
        if m:
            print(f"{key:20s}  MAPE={m['MAPE (%)']:.1f}%  R²={m['R²']:.3f}")

    if "best_model" in results:
        print("\n=== Recommended Model ===")
        print(f"{results['best_model']['model_key']}  (MAPE={results['best_model']['MAPE (%)']:.2f}%)")
