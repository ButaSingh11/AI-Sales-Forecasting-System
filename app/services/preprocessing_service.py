import pandas as pd
import numpy as np
import re
from typing import Tuple, Optional, Dict, Any, List


def load_file(file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load CSV or Excel file. Returns (df, error_message)."""
    try:
        if file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)
        return df, None
    except Exception as e:
        return None, str(e)


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect the date column by name or dtype."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        if any(k in col.lower() for k in ["date", "time", "day", "timestamp"]):
            return col
    return None


def detect_sales_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect the sales/revenue target column."""
    keywords = ["sale", "sales", "revenue", "amount", "total", "price", "value", "income"]
    for col in df.columns:
        if any(k in col.lower() for k in keywords):
            return col
    # fallback: first numeric column
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None


def clean_numeric_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean currency-like numeric strings into numeric values.

    Handles:
    - ₹10,000
    - 12,500
    - 5k
    - 2.5L
    - 1.2Cr
    """
    df = df.copy()

    def parse_value(v):
        if pd.isna(v):
            return v
        if isinstance(v, (int, float, np.number)):
            return v

        s = str(v).strip()
        if s == "":
            return np.nan

        s = s.replace("₹", "").replace(",", "").replace(" ", "")
        s_lower = s.lower()

        multiplier = 1.0
        if s_lower.endswith("cr"):
            multiplier = 1e7
            s_lower = s_lower[:-2]
        elif s_lower.endswith("l"):
            multiplier = 1e5
            s_lower = s_lower[:-1]
        elif s_lower.endswith("k"):
            multiplier = 1e3
            s_lower = s_lower[:-1]

        s_lower = re.sub(r"[^0-9.\-]", "", s_lower)
        if s_lower in ("", "-", ".", "-."):
            return v

        try:
            return float(s_lower) * multiplier
        except Exception:
            return v

    for col in df.columns:
        if df[col].dtype == object:
            cleaned = df[col].map(parse_value)
            numeric_ratio = pd.to_numeric(cleaned, errors="coerce").notna().mean()
            if numeric_ratio >= 0.6:
                df[col] = pd.to_numeric(cleaned, errors="coerce")

    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Try to parse any date-like columns with simple heuristics."""
    df = df.copy()
    for col in df.columns:
        if any(k in col.lower() for k in ["date", "time", "day", "timestamp"]):
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() >= 0.6:
                    df[col] = parsed
            except Exception:
                pass
    return df


def detect_outliers(df: pd.DataFrame,
                    sales_col: str,
                    method: str = "iqr") -> pd.DataFrame:
    """
    Detect outliers in the sales column.
    Returns outlier rows with outlier_score and outlier_method.
    """
    if sales_col not in df.columns:
        return pd.DataFrame()

    temp = df.copy()
    series = pd.to_numeric(temp[sales_col], errors="coerce")

    if method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            return pd.DataFrame()
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (series < lower) | (series > upper)
        out = temp.loc[mask].copy()
        out["outlier_score"] = (
            np.maximum((series.loc[mask] - upper).abs(), (lower - series.loc[mask]).abs()) / max(iqr, 1)
        ).round(2)
        out["outlier_method"] = "iqr"
        return out

    if method == "zscore":
        mean = series.mean()
        std = series.std()
        if pd.isna(std) or std == 0:
            return pd.DataFrame()
        z = ((series - mean) / std).abs()
        mask = z > 3
        out = temp.loc[mask].copy()
        out["outlier_score"] = z.loc[mask].round(2)
        out["outlier_method"] = "zscore"
        return out

    raise ValueError("Unknown outlier method. Use 'iqr' or 'zscore'.")


def schema_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a compact schema summary."""
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
        "duplicates": int(df.duplicated().sum()),
    }


def create_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Create useful time-based features if a date column exists."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if df[date_col].notna().sum() == 0:
        return df

    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["weekday"] = df[date_col].dt.day_name()
    df["quarter"] = df[date_col].dt.quarter
    df["is_month_end"] = df[date_col].dt.is_month_end.astype(int)
    df["is_month_start"] = df[date_col].dt.is_month_start.astype(int)
    return df


def create_lag_features(df: pd.DataFrame,
                        sales_col: str,
                        date_col: Optional[str] = None,
                        lags: List[int] = [1, 2, 3]) -> pd.DataFrame:
    """Create lag features from target variable."""
    df = df.copy()
    if sales_col not in df.columns:
        return df

    if date_col and date_col in df.columns:
        df = df.sort_values(date_col).reset_index(drop=True)

    sales = pd.to_numeric(df[sales_col], errors="coerce")
    for lag in lags:
        df[f"lag_{lag}"] = sales.shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame,
                            sales_col: str,
                            date_col: Optional[str] = None,
                            windows: List[int] = [7, 30]) -> pd.DataFrame:
    """Create rolling averages and rolling std trends."""
    df = df.copy()
    if sales_col not in df.columns:
        return df

    if date_col and date_col in df.columns:
        df = df.sort_values(date_col).reset_index(drop=True)

    sales = pd.to_numeric(df[sales_col], errors="coerce")
    for w in windows:
        df[f"rolling_mean_{w}"] = sales.rolling(w, min_periods=1).mean()
        df[f"rolling_std_{w}"] = sales.rolling(w, min_periods=1).std()
    return df


def encode_categoricals(df: pd.DataFrame,
                        exclude_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    One-hot encode categorical columns.
    Returns encoded dataframe and list of encoded source columns.
    """
    df = df.copy()
    exclude_cols = exclude_cols or []

    cat_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c not in exclude_cols
    ]

    if not cat_cols:
        return df, []

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df, cat_cols


def scale_numeric_features(df: pd.DataFrame,
                           exclude_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply simple min-max scaling to numeric predictors.
    Keeps target and date-like excluded columns unchanged.
    """
    df = df.copy()
    exclude_cols = exclude_cols or []

    scaled_cols = []
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in num_cols:
        if col in exclude_cols:
            continue
        col_min = df[col].min()
        col_max = df[col].max()
        if pd.isna(col_min) or pd.isna(col_max):
            continue
        if col_max != col_min:
            df[col] = (df[col] - col_min) / (col_max - col_min)
            scaled_cols.append(col)

    return df, scaled_cols


def validate_dataframe(df: pd.DataFrame,
                       date_col: Optional[str],
                       sales_col: Optional[str]) -> list:
    """
    Run validation checks. Returns list of
    (check_name, detail, level) tuples.
    """
    checks = []
    n = len(df)

    checks.append((
        "Dataset Size",
        f"{n:,} rows × {df.shape[1]} columns",
        "ok" if n >= 100 else ("warn" if n >= 30 else "error"),
    ))

    checks.append((
        "Date Column Detected",
        f"`{date_col}`" if date_col else "No date column found",
        "ok" if date_col else "warn",
    ))

    checks.append((
        "Target / Sales Column",
        f"`{sales_col}`" if sales_col else "No likely sales target found",
        "ok" if sales_col else "warn",
    ))

    total_missing = df.isnull().sum().sum()
    miss_pct = total_missing / (df.shape[0] * df.shape[1]) * 100 if df.shape[0] and df.shape[1] else 0
    checks.append((
        "Missing Values",
        f"{total_missing} nulls ({miss_pct:.1f}%)",
        "ok" if miss_pct == 0 else ("warn" if miss_pct < 20 else "error"),
    ))

    dups = df.duplicated().sum()
    checks.append((
        "Duplicate Rows",
        f"{dups} duplicates found",
        "ok" if dups == 0 else "warn",
    ))

    if sales_col and sales_col in df.columns:
        neg = (pd.to_numeric(df[sales_col], errors="coerce") < 0).sum()
        checks.append((
            "Negative Target Values",
            f"{int(neg)} negative value(s)",
            "ok" if neg == 0 else "warn",
        ))

        outliers = detect_outliers(df, sales_col, method="iqr")
        checks.append((
            "Outlier Detection",
            f"{len(outliers)} outlier row(s) detected",
            "ok" if len(outliers) == 0 else "warn",
        ))

    return checks


def generate_data_insights(df: pd.DataFrame,
                           date_col: Optional[str],
                           sales_col: Optional[str]) -> Dict[str, Any]:
    """
    Generate summary insights like:
    - last 30 days trend
    - monthly comparison
    - anomalies
    - increasing/decreasing trend
    """
    insights = {
        "messages": [],
        "feature_list": [],
        "suggestions": [],
    }

    if not sales_col or sales_col not in df.columns:
        insights["suggestions"].append("A clear sales/target column was not detected. Rename your target column to something like Sales or Revenue.")
        return insights

    temp = df.copy()
    temp[sales_col] = pd.to_numeric(temp[sales_col], errors="coerce")

    if date_col and date_col in temp.columns:
        temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
        temp = temp.sort_values(date_col)

        if temp[date_col].notna().sum() > 0:
            # Last 30 days trend
            recent_30 = temp[temp[date_col] >= temp[date_col].max() - pd.Timedelta(days=30)]
            prev_30 = temp[
                (temp[date_col] < temp[date_col].max() - pd.Timedelta(days=30)) &
                (temp[date_col] >= temp[date_col].max() - pd.Timedelta(days=60))
            ]

            if len(recent_30) > 0 and len(prev_30) > 0:
                recent_sum = recent_30[sales_col].sum()
                prev_sum = prev_30[sales_col].sum()
                pct = ((recent_sum - prev_sum) / prev_sum * 100) if prev_sum else 0.0
                insights["messages"].append(
                    f"Last 30 days trend: {pct:+.1f}% compared with the previous 30-day period."
                )

            # Monthly comparison
            temp["month_period"] = temp[date_col].dt.to_period("M")
            monthly = temp.groupby("month_period")[sales_col].sum()

            if len(monthly) >= 2:
                current = monthly.iloc[-1]
                previous = monthly.iloc[-2]
                pct = ((current - previous) / previous * 100) if previous else 0.0
                direction = "increasing" if pct > 0 else ("decreasing" if pct < 0 else "stable")
                insights["messages"].append(
                    f"Monthly comparison: latest month is {pct:+.1f}% vs previous month, indicating a {direction} sales trend."
                )

            if len(monthly) >= 6:
                first_half = monthly.iloc[:len(monthly)//2].mean()
                second_half = monthly.iloc[len(monthly)//2:].mean()
                long_trend = ((second_half - first_half) / first_half * 100) if first_half else 0.0
                insights["messages"].append(
                    f"Overall time trend across the dataset is {long_trend:+.1f}% from earlier periods to later periods."
                )

            outliers = detect_outliers(temp, sales_col, method="iqr")
            if len(outliers) > 0:
                insights["messages"].append(
                    f"Detected {len(outliers)} unusual spike/drop point(s) using IQR-based outlier detection."
                )

        else:
            insights["suggestions"].append("A date-like column was found but could not be parsed reliably. Use a standard date format.")
    else:
        insights["suggestions"].append("No date column detected. Add a date column to unlock time-based forecasting, lag features, and trend analysis.")

    if len(df) < 30:
        insights["suggestions"].append("Dataset is quite small. More historical rows will improve forecasting quality.")

    return insights


def clean_dataframe(df: pd.DataFrame,
                    date_col: Optional[str],
                    sales_col: Optional[str],
                    prepare_model_features: bool = False) -> Tuple[pd.DataFrame, list]:
    """
    Clean dataframe and return (cleaned_df, steps_log).
    steps_log contains (label, detail, level) tuples.

    By default this keeps business-facing columns such as Category and Region
    intact so the Analysis and Chatbot pages can still group and explain data.
    Set prepare_model_features=True only when a downstream model needs encoded
    and scaled predictors.
    """
    steps = []
    df = df.copy()

    # 0. Numeric string cleaning
    before_dtypes = df.dtypes.astype(str).to_dict()
    df = clean_numeric_strings(df)
    after_dtypes = df.dtypes.astype(str).to_dict()
    changed_cols = [c for c in df.columns if before_dtypes.get(c) != after_dtypes.get(c)]
    steps.append((
        "💱 Numeric / Currency Cleaning",
        f"Converted likely numeric text columns: {', '.join(changed_cols) if changed_cols else 'No conversion needed'}. This helps models use sales-like values correctly.",
        "ok",
    ))

    # 1. Smarter date parsing
    df = parse_dates(df)
    if not date_col:
        date_col = detect_date_column(df)
    if not sales_col:
        sales_col = detect_sales_column(df)

    steps.append((
        "📅 Date Parsing",
        f"Detected date column: `{date_col}`." if date_col else "No reliable date column detected.",
        "ok" if date_col else "warn",
    ))

    # 2. Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    steps.append((
        "🗑️ Duplicate Treatment",
        f"Removed {removed} duplicate row(s). This avoids biasing training with repeated records.",
        "ok" if removed == 0 else "warn",
    ))

    # 3. Missing values
    critical = [c for c in [date_col, sales_col] if c]
    before = len(df)
    if critical:
        df = df.dropna(subset=critical)
    dropped = before - len(df)

    num_cols = df.select_dtypes(include="number").columns.tolist()
    filled_num = int(sum(df[c].isnull().sum() for c in num_cols))
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    obj_cols = df.select_dtypes(include="object").columns.tolist()
    filled_txt = int(sum(df[c].isnull().sum() for c in obj_cols))
    for c in obj_cols:
        mode = df[c].mode()
        df[c] = df[c].fillna(mode[0] if not mode.empty else "Unknown")

    steps.append((
        "🧹 Missing Value Handling",
        f"Dropped {dropped} row(s) missing critical fields. Filled {filled_num} numeric null(s) with median and {filled_txt} text null(s) with mode. This preserves usable data while keeping important columns reliable.",
        "ok" if dropped == 0 and filled_num == 0 and filled_txt == 0 else "warn",
    ))

    # 4. Outlier handling
    outlier_count = 0
    if sales_col and sales_col in df.columns:
        outliers = detect_outliers(df, sales_col, method="iqr")
        outlier_count = len(outliers)
    steps.append((
        "🧠 Outlier Detection",
        f"Detected {outlier_count} outlier row(s) using the IQR method. Outliers are highlighted because unusual spikes can distort forecasting and business interpretation.",
        "ok" if outlier_count == 0 else "warn",
    ))

    # 5. Negative target clipping
    neg_count = 0
    if sales_col and sales_col in df.columns:
        sales_numeric = pd.to_numeric(df[sales_col], errors="coerce")
        neg_count = int((sales_numeric < 0).sum())
        df[sales_col] = sales_numeric.clip(lower=0)

    steps.append((
        "📉 Negative Value Treatment",
        f"Clipped {neg_count} negative target value(s) to zero. Negative sales usually indicate bad input for standard forecasting.",
        "ok" if neg_count == 0 else "warn",
    ))

    # 6. Standardize names
    rename_map = {}
    if date_col and date_col != "Date":
        rename_map[date_col] = "Date"
    if sales_col and sales_col != "Sales":
        rename_map[sales_col] = "Sales"

    if rename_map:
        df = df.rename(columns=rename_map)
        date_col = "Date" if "Date" in df.columns else date_col
        sales_col = "Sales" if "Sales" in df.columns else sales_col

    rename_text = ", ".join([f"'{old}'→'{new}'" for old, new in rename_map.items()]) if rename_map else "No renaming needed"
    steps.append((
        "🏷️ Column Standardization",
        rename_text,
        "ok",
    ))

    # 7. Feature engineering
    created_features = []

    if date_col and date_col in df.columns:
        before_cols = set(df.columns)
        df = create_time_features(df, date_col)
        created_features += [c for c in df.columns if c not in before_cols]

    if sales_col and sales_col in df.columns:
        before_cols = set(df.columns)
        df = create_lag_features(df, sales_col, date_col=date_col, lags=[1, 2, 3])
        df = create_rolling_features(df, sales_col, date_col=date_col, windows=[7, 30])
        created_features += [c for c in df.columns if c not in before_cols]

    steps.append((
        "⚙️ Feature Engineering",
        (
            f"Created features: {', '.join(created_features) if created_features else 'No new features created'}. "
            f"Time features help the model learn calendar effects, lag features help it learn previous sales behavior, "
            f"and rolling features help it capture recent momentum and smoothing."
        ),
        "ok" if created_features else "warn",
    ))

    if not prepare_model_features:
        steps.append((
            "Business Columns Preserved",
            (
                "Kept categorical and numeric business columns in their original form. "
                "This keeps Category, Region, Product, Orders, and similar fields available for analysis and chatbot answers."
            ),
            "ok",
        ))

        summary = schema_summary(df)
        steps.append((
            "Final Schema Summary",
            f"{summary['rows']:,} rows x {summary['columns']} columns after preprocessing.",
            "ok",
        ))

        df = df.reset_index(drop=True)
        steps.append((
            "Analysis-Ready Output",
            "Dataset has been cleaned and prepared for analysis, forecasting, scenario simulation, and chatbot use.",
            "ok",
        ))

        return df, steps

    # 8. Encoding
    exclude_for_encoding = []
    if date_col and date_col in df.columns:
        exclude_for_encoding.append(date_col)
    df, encoded_cols = encode_categoricals(df, exclude_cols=exclude_for_encoding)

    steps.append((
        "🔤 Categorical Encoding",
        (
            f"Encoded columns: {', '.join(encoded_cols) if encoded_cols else 'No categorical encoding needed'}. "
            f"This helps machine learning models understand non-numeric categories."
        ),
        "ok",
    ))

    # 9. Scaling
    exclude_for_scaling = ["Sales", "Date"]
    scaled_df, scaled_cols = scale_numeric_features(df, exclude_cols=[c for c in exclude_for_scaling if c in df.columns])
    df = scaled_df

    steps.append((
        "📏 Scaling / Normalization",
        (
            f"Scaled numeric feature columns: {', '.join(scaled_cols) if scaled_cols else 'No scaling applied'}. "
            f"Scaling helps some models compare features on a similar range."
        ),
        "ok",
    ))

    # 10. Final schema
    summary = schema_summary(df)
    steps.append((
        "🧾 Final Schema Summary",
        f"{summary['rows']:,} rows × {summary['columns']} columns after preprocessing.",
        "ok",
    ))

    df = df.reset_index(drop=True)
    steps.append((
        "✅ Model-Ready Output",
        f"Dataset has been cleaned, enriched, and prepared for training, forecasting, and chatbot analysis.",
        "ok",
    ))

    return df, steps
