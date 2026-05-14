import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from services.forecasting_service import MODEL_REGISTRY


@dataclass
class ModelComparisonResult:
    comparison_df: pd.DataFrame
    best_model_name: str
    best_model: object
    trained_models: Dict[str, object]
    best_metrics: Dict
    recommendation_reason: str
    feature_importance_df: pd.DataFrame


def _saved_models_dir() -> str:
    project_root = Path(__file__).resolve().parents[2]
    return str(project_root / "models" / "saved_models")


def save_model(model_key: str,
               kwargs: Optional[Dict] = None,
               model_name: Optional[str] = None,
               metadata: Optional[Dict] = None) -> str:
    """
    Save a lightweight model payload describing which forecasting function to use.

    The actual forecast functions are code-defined, so we persist the model key,
    kwargs, display name, and metadata rather than trying to pickle executable code.
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_key: {model_key}")

    kwargs = kwargs or {}
    metadata = metadata or {}
    model_name = model_name or model_key

    save_dir = _saved_models_dir()
    os.makedirs(save_dir, exist_ok=True)

    base_name = model_name.replace(" ", "_").lower()
    file_path = os.path.join(save_dir, f"{base_name}.pkl")
    meta_path = file_path.replace(".pkl", "_meta.json")

    payload = {
        "model_key": model_key,
        "kwargs": kwargs,
        "name": model_name,
        "metadata": metadata,
    }

    with open(file_path, "wb") as f:
        pickle.dump(payload, f)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return file_path


def load_model(model_path: str) -> Dict:
    """
    Load a saved lightweight model payload and attach the executable forecast fn.
    """
    with open(model_path, "rb") as f:
        payload = pickle.load(f)

    if "model_key" not in payload:
        raise ValueError("Saved model payload is missing 'model_key'.")

    model_key = payload["model_key"]
    model_fn = MODEL_REGISTRY.get(model_key)
    if model_fn is None:
        raise ValueError(f"Saved model references unknown model_key: {model_key}")

    return {
        "model_key": model_key,
        "model_fn": model_fn,
        "kwargs": payload.get("kwargs", {}),
        "name": payload.get("name", model_key),
        "metadata": payload.get("metadata", {}),
    }


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    denominator = np.where(np.abs(y_true) < 1e-8, 1.0, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100)


def build_regression_models(random_state: int = 42) -> Dict[str, object]:
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
    }


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    mape = safe_mape(y_true, y_pred)

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2 Score": round(r2, 4),
        "MAPE (%)": round(mape, 2),
    }


def train_and_compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: Optional[List[str]] = None,
    random_state: int = 42,
) -> ModelComparisonResult:
    if X_train is None or X_test is None or y_train is None or y_test is None:
        raise ValueError("Training and testing data must not be None.")

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Training and testing data must not be empty.")

    if feature_names is None:
        if isinstance(X_train, pd.DataFrame):
            feature_names = list(X_train.columns)
        else:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    models = build_regression_models(random_state=random_state)
    comparison_rows = []
    trained_models = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = calculate_regression_metrics(y_test, predictions)

        comparison_rows.append(
            {
                "Model": model_name,
                **metrics,
            }
        )
        trained_models[model_name] = model

    comparison_df = pd.DataFrame(comparison_rows).sort_values("RMSE").reset_index(drop=True)

    best_model_name = comparison_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    best_metrics = comparison_df.iloc[0].to_dict()

    feature_importance_df = get_feature_importance_dataframe(
        model_name=best_model_name,
        model=best_model,
        feature_names=feature_names,
    )

    recommendation_reason = build_model_recommendation_reason(
        best_model_name=best_model_name,
        comparison_df=comparison_df,
    )

    return ModelComparisonResult(
        comparison_df=comparison_df,
        best_model_name=best_model_name,
        best_model=best_model,
        trained_models=trained_models,
        best_metrics=best_metrics,
        recommendation_reason=recommendation_reason,
        feature_importance_df=feature_importance_df,
    )


def train_best_model_on_full_data(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
):
    models = build_regression_models(random_state=random_state)

    if model_name not in models:
        raise ValueError(f"Unsupported model name: {model_name}")

    model = models[model_name]
    model.fit(X, y)
    return model


def get_feature_importance_dataframe(
    model_name: str,
    model,
    feature_names: List[str],
) -> pd.DataFrame:
    if model_name != "Random Forest":
        return pd.DataFrame(columns=["Feature", "Importance"])

    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["Feature", "Importance"])

    fi_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    fi_df["Importance"] = fi_df["Importance"].round(4)
    return fi_df.reset_index(drop=True)


def build_model_recommendation_reason(
    best_model_name: str,
    comparison_df: pd.DataFrame,
    anomaly_count: int = 0,
    data_points: Optional[int] = None,
) -> str:
    if comparison_df is None or comparison_df.empty:
        return "No model recommendation is available because model comparison results are empty."

    best_row = comparison_df.iloc[0]
    reason_parts = [
        f"**{best_model_name}** is recommended because it achieved the lowest RMSE ({best_row['RMSE']}) and strong MAE ({best_row['MAE']})."
    ]

    if best_model_name == "Random Forest":
        reason_parts.append(
            "It is effective when sales behavior is non-linear and influenced by lag values, trends, and calendar-based features."
        )
    elif best_model_name == "Gradient Boosting":
        reason_parts.append(
            "It is strong at learning complex patterns and improving difficult prediction areas step by step."
        )
    else:
        reason_parts.append(
            "It works best when the relationship in the data is more direct, stable, and close to linear."
        )

    if data_points is not None:
        if data_points >= 90:
            reason_parts.append("The dataset has enough historical depth to support a more reliable recommendation.")
        else:
            reason_parts.append("The historical data is somewhat limited, so accuracy can improve as more records are added.")

    if anomaly_count > 0:
        reason_parts.append(
            f"{anomaly_count} anomaly point(s) were detected, so the forecast should be interpreted with additional business context."
        )

    return " ".join(reason_parts)


def choose_best_model_name(comparison_df: pd.DataFrame) -> str:
    if comparison_df is None or comparison_df.empty:
        raise ValueError("Comparison dataframe is empty. Cannot choose best model.")

    sorted_df = comparison_df.sort_values("RMSE").reset_index(drop=True)
    return str(sorted_df.iloc[0]["Model"])


def convert_comparison_df_to_csv_bytes(comparison_df: pd.DataFrame) -> bytes:
    if comparison_df is None:
        comparison_df = pd.DataFrame()

    return comparison_df.to_csv(index=False).encode("utf-8")


def summarize_model_performance(comparison_df: pd.DataFrame) -> Dict[str, str]:
    if comparison_df is None or comparison_df.empty:
        return {
            "headline": "No model summary available",
            "summary": "Model comparison results are empty, so no model can be recommended.",
        }

    best_row = comparison_df.sort_values("RMSE").iloc[0]
    worst_row = comparison_df.sort_values("RMSE").iloc[-1]

    headline = f"{best_row['Model']} performed best on the current dataset."

    summary = (
        f"{best_row['Model']} achieved the lowest RMSE ({best_row['RMSE']}) and MAE ({best_row['MAE']}). "
        f"The weakest model in this comparison was {worst_row['Model']} with RMSE {worst_row['RMSE']}. "
        f"This comparison helps the system choose the most suitable model instead of using one fixed approach for every dataset."
    )

    return {
        "headline": headline,
        "summary": summary,
    }
