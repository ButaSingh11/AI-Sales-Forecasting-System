"""
predict_model.py
----------------
Load a saved model (metadata-based format) and generate predictions on new data.

Usage:
    python -m models.predict_model \
        --model models/saved_models/best_model.pkl \
        --data  data/raw/sample_sales.csv \
        --periods 6
"""

import argparse
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from services.model_service import load_model
from services.forecasting_service import prepare_monthly_series


def predict(model_path: str, data_path: str,
            periods: int = 12) -> pd.DataFrame:
    """
    Load a saved model and generate forecast for `periods` months.

    Parameters
    ----------
    model_path : str
        Path to the saved .pkl model file
    data_path : str
        Path to the input sales CSV
    periods : int
        Number of months to forecast

    Returns
    -------
    pd.DataFrame
        DataFrame with Date, Forecast, Upper, Lower columns
    """
    # Load model payload from metadata-based save format
    model_payload = load_model(model_path)
    fn = model_payload["model_fn"]
    kwargs = model_payload["kwargs"]
    name = model_payload["name"]
    model_key = model_payload["model_key"]
    metadata = model_payload.get("metadata", {})

    print(f"Loaded model: {name}")
    print(f"Model key: {model_key}")
    print(f"Saved kwargs: {kwargs}")
    if metadata:
        print(f"Metadata: {metadata}")

    # Load and prepare data
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    series = prepare_monthly_series(df)

    print(f"Data loaded: {len(series)} monthly records")
    print(f"Generating {periods}-month forecast...")

    # Generate forecast
    forecast_df = fn(series, periods, **kwargs)

    print("\nForecast:")
    print(forecast_df.to_string(index=False))

    return forecast_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions from a saved model")
    parser.add_argument("--model", required=True,
                        help="Path to saved .pkl model file")
    parser.add_argument("--data", default="data/raw/sample_sales.csv",
                        help="Path to sales CSV file")
    parser.add_argument("--periods", type=int, default=12,
                        help="Number of months to forecast")
    parser.add_argument("--output", default=None,
                        help="Optional: save forecast to this CSV path")
    args = parser.parse_args()

    forecast = predict(args.model, args.data, args.periods)

    if args.output:
        forecast.to_csv(args.output, index=False)
        print(f"\nForecast saved to: {args.output}")
