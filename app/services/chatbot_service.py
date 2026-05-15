import pandas as pd
import requests
import re
from typing import List, Dict, Optional, Tuple

from app.services.insight_service import detect_anomalies
from app.services.forecasting_service import (
    compare_forecast_models,
    forecast_with_best_model,
    get_model_label,
)
from app.services.evaluation_service import forecast_confidence_score
from app.utils.app_helpers import format_inr, normalize_text_columns, repair_text_encoding


def build_data_summary(df: pd.DataFrame) -> str:
    """Build a statistical summary of the dataset for Gemini context injection."""
    df = normalize_text_columns(df)
    df["Date"] = pd.to_datetime(df["Date"])

    total_sales = df["Sales"].sum()
    avg_daily = df["Sales"].mean()
    date_min = df["Date"].min().strftime("%d %b %Y")
    date_max = df["Date"].max().strftime("%d %b %Y")
    total_rows = len(df)

    df["Month"] = df["Date"].dt.to_period("M")
    monthly = df.groupby("Month")["Sales"].sum()
    best_month = monthly.idxmax()
    worst_month = monthly.idxmin()
    monthly_avg = monthly.mean()

    if len(monthly) >= 2:
        first_3 = monthly.iloc[:3].mean()
        last_3 = monthly.iloc[-3:].mean()
        growth = (last_3 - first_3) / first_3 * 100 if first_3 > 0 else 0
    else:
        growth = 0

    summary = f"""=== SALES DATASET SUMMARY ===
Date Range    : {date_min} to {date_max}
Total Records : {total_rows:,} rows
Total Revenue : ₹{total_sales:,.0f}
Avg Daily Rev : ₹{avg_daily:,.0f}
Avg Monthly   : ₹{monthly_avg:,.0f}
Best Month    : {best_month} (₹{monthly.max():,.0f})
Worst Month   : {worst_month} (₹{monthly.min():,.0f})
Overall Trend : {growth:+.1f}% (first 3 vs last 3 months avg)
"""

    if "Category" in df.columns:
        cat_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
        summary += "\n=== REVENUE BY CATEGORY ===\n"
        for cat, val in cat_sales.items():
            summary += f"{cat}: ₹{val:,.0f} ({val/total_sales*100:.1f}%)\n"

    if "Region" in df.columns:
        reg_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
        summary += "\n=== REVENUE BY REGION ===\n"
        for reg, val in reg_sales.items():
            summary += f"{reg}: ₹{val:,.0f} ({val/total_sales*100:.1f}%)\n"

    df["Year"] = df["Date"].dt.year
    yr_sales = df.groupby("Year")["Sales"].sum()
    summary += "\n=== REVENUE BY YEAR ===\n"
    for yr, val in yr_sales.items():
        summary += f"{yr}: ₹{val:,.0f}\n"

    return summary.strip()


def build_system_prompt(data_summary: str) -> str:
    return f"""You are an expert AI Sales Analyst embedded inside an AI-powered sales forecasting application.

You have access to a statistical summary of the user's sales data and can answer questions about trends, performance, forecasting, strategy, and business insights.

Here is the user's sales data summary:

{data_summary}

Guidelines:
- Keep explanations simple but intelligent
- Match the answer format to the user's question instead of repeating the same template every time
- Start with a direct conclusion, then add only the most useful supporting numbers
- Use short headings, short bullets, and practical recommendations when helpful
- Explain what the data shows: trends, seasonality, anomalies, recent changes
- Mention time-based patterns like daily, monthly, or recent movement
- Mention preprocessing only when it is relevant to the question:
  - missing value handling
  - duplicate removal
  - outlier detection/treatment
  - encoding/scaling/transformation only if relevant
- If possible, compare before vs after cleaning quality
- Provide the prediction / answer clearly
- Explain what influenced the answer:
  - lag values
  - recent trend
  - rolling averages
  - seasonality
- If relevant, simulate a simple scenario like “what if sales increase by 20%”
- If graphs are available conceptually, describe what they show
- Highlight unusual spikes or anomalies if relevant
- Compare values when useful, such as this month vs last month
- Preserve all directly computed numeric values exactly
- Be honest if something is outside the available data context
- Avoid long generic explanations for simple questions

Tone:
Professional, friendly, data-driven, practical.
"""


def call_gemini_api(messages: List[Dict],
                    system_prompt: str,
                    api_key: str,
                    model: str = "gemini-2.5-flash",
                    max_output_tokens: int = 1024,
                    temperature: float = 0.5) -> str:
    """Call Gemini generateContent REST API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({
            "role": role,
            "parts": [{"text": msg["content"]}]
        })

    payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)

        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return "⚠️ Gemini returned no response."

            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(part.get("text", "") for part in parts if "text" in part).strip()
            return text or "⚠️ Gemini returned an empty response."

        if resp.status_code in (401, 403):
            return "❌ Invalid Gemini API key or permission denied."
        if resp.status_code == 429:
            return "⚠️ Gemini rate limit reached. Please wait a moment and try again."

        return f"❌ Gemini API error {resp.status_code}: {resp.text[:250]}"

    except requests.exceptions.Timeout:
        return "⚠️ Gemini request timed out. Please try again."
    except Exception as e:
        return f"❌ Connection error: {str(e)}"


def trim_history(messages: List[Dict], max_turns: int = 10) -> List[Dict]:
    return messages[-(max_turns * 2):]


def detect_query_intent(query: str) -> str:
    q = query.lower().strip()

    weak_terms = [
        "weak",
        "weakest",
        "worst",
        "lowest",
        "least",
        "poor",
        "poorest",
        "underperform",
        "under performing",
        "under-performing",
        "low performing",
        "low-performing",
    ]

    if any(k in q for k in ["anomaly", "anomalies", "spike", "drop", "unusual", "outlier", "outliers"]):
        return "anomaly"
    if any(k in q for k in [
        "recommendation",
        "recommendations",
        "recommend",
        "business advice",
        "business insight",
        "business insights",
        "action plan",
        "next steps",
        "what should i do",
        "what to do",
        "strategy",
        "strategies",
        "decision",
        "decisions",
    ]):
        return "recommendations"
    if any(k in q for k in ["model", "algorithm", "mape", "rmse", "mae", "r-squared", "r squared", "r2", "accuracy", "confidence interval", "feature importance", "smart ensemble", "holt", "random forest", "linear trend", "moving average", "exponential smoothing", "seasonal naive"]):
        return "model_info"
    if (
        any(k in q for k in ["project", "application", "module", "workflow", "feature", "tech stack", "technology", "architecture"])
        or re.search(r"\bapp\b", q)
    ):
        return "project_info"
    if any(k in q for k in ["column", "columns", "missing", "duplicate", "duplicates", "clean", "cleaning", "preprocess", "preprocessing", "data quality", "shape", "rows", "fields", "schema"]):
        return "data_info"
    if any(k in q for k in ["scenario", "what if", "simulate", "marketing", "churn", "discount"]):
        return "scenario"
    if (
        any(k in q for k in ["forecast", "forecasting", "predict", "prediction", "next month", "next 6", "future sales", "future revenue"])
        or re.search(r"\bnext\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)[-\s]*(month|months|year|years)\b", q)
    ):
        return "forecast"
    if any(k in q for k in ["compare", "comparison", "vs", "versus", "last 3 months", "last month vs"]):
        return "comparison"
    if any(k in q for k in ["total sales", "total revenue", "overall sales", "average sales", "avg sales", "date range", "record count", "how many records"]):
        return "summary"
    if "region" in q and any(term in q for term in weak_terms):
        return "weakest_region"
    if "category" in q and any(term in q for term in weak_terms):
        return "weakest_category"
    if any(term in q for term in ["product", "item", "sku", "brand"]) and any(term in q for term in weak_terms):
        return "weakest_product"
    if "month" in q and any(term in q for term in weak_terms):
        return "weakest_month"
    if any(k in q for k in ["best month", "highest month", "top month", "peak month"]):
        return "best_month"
    if any(k in q for k in ["best category", "top category", "highest category"]):
        return "best_category"
    if any(k in q for k in ["best product", "top product", "highest product", "best item", "top item", "highest item"]):
        return "best_product"
    if any(k in q for k in ["best region", "top region", "highest region"]):
        return "best_region"
    if any(k in q for k in ["trend", "growth", "decline", "upward", "downward", "momentum"]):
        return "trend"
    if any(k in q for k in ["summary", "overview", "dataset", "overall", "tell me about my data"]):
        return "summary"

    return "fallback"


def _format_inr(value: float) -> str:
    return format_inr(value)


def _monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp["Date"] = pd.to_datetime(temp["Date"])
    temp["Month"] = temp["Date"].dt.to_period("M").dt.to_timestamp()
    monthly = temp.groupby("Month")["Sales"].sum().reset_index()
    return monthly.sort_values("Month").reset_index(drop=True)


def _number_from_text(value: str) -> Optional[int]:
    value = value.lower().strip()
    word_numbers = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "twenty four": 24,
        "twenty-four": 24,
    }
    if value.isdigit():
        return int(value)
    return word_numbers.get(value)


def _parse_forecast_horizon(query: str, default_months: int = 6, max_months: int = 60) -> int:
    """Extract a requested forecast horizon from natural language."""
    q = query.lower()
    number_pattern = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty[- ]four)"

    if "next month" in q:
        return 1
    if re.search(r"\bnext\s+year\b", q):
        return 12

    year_match = re.search(number_pattern + r"[-\s]*(?:year|years|yr|yrs)\b", q)
    if year_match:
        value = _number_from_text(year_match.group(1))
        if value:
            return max(1, min(value * 12, max_months))

    month_match = re.search(number_pattern + r"[-\s]*(?:month|months|mo|mos)\b", q)
    if month_match:
        value = _number_from_text(month_match.group(1))
        if value:
            return max(1, min(value, max_months))

    next_number_match = re.search(r"\bnext\s+" + number_pattern + r"\b", q)
    if next_number_match:
        value = _number_from_text(next_number_match.group(1))
        if value:
            return max(1, min(value, max_months))

    return default_months


def _format_horizon_label(months: int) -> str:
    if months == 1:
        return "1 month"
    if months % 12 == 0:
        years = months // 12
        return f"{months} months ({years} {'year' if years == 1 else 'years'})"
    return f"{months} months"


def _long_horizon_warning(months: int) -> str:
    if months >= 36:
        return "Long-horizon warning: forecasts beyond 3 years are highly uncertain and should be treated as planning scenarios, not fixed targets."
    if months >= 24:
        return "Long-horizon warning: 2-year forecasts carry wider uncertainty, so review them with business context and update them as new months arrive."
    if months >= 12:
        return "Planning note: forecasts beyond 12 months are useful for direction, but near-term months are usually more reliable."
    return ""


def _forecast_confidence_context(monthly: pd.DataFrame,
                                 comparison_df: pd.DataFrame,
                                 model_key: str) -> Dict[str, object]:
    if comparison_df is None or comparison_df.empty:
        return {
            "level": "Not rated",
            "score": None,
            "mape": None,
            "reason": "Not enough holdout history to estimate forecast confidence.",
        }

    model_rows = comparison_df[comparison_df["model_key"] == model_key]
    best_row = model_rows.iloc[0] if not model_rows.empty else comparison_df.iloc[0]
    mape = float(best_row["MAPE (%)"])
    volatility = float(monthly["Sales"].std() / monthly["Sales"].mean() * 100) if monthly["Sales"].mean() else 0.0
    anomaly_input = monthly.rename(columns={"Month": "Date"})[["Date", "Sales"]].copy()
    anomaly_count = len(detect_anomalies(
        anomaly_input,
        method="rolling_deviation",
        threshold=2.5,
        rolling_window=6,
    ))
    confidence = forecast_confidence_score(
        mape=mape,
        volatility=volatility,
        data_points=len(monthly),
        anomaly_count=anomaly_count,
    )
    confidence["mape"] = mape
    confidence["volatility"] = volatility
    confidence["anomaly_count"] = anomaly_count
    return confidence


def _forecast_from_monthly(monthly: pd.DataFrame,
                           horizon: int,
                           include_random_forest: bool = True) -> Tuple[List[Tuple[pd.Timestamp, float, float, float]], float, float, str, Dict[str, object]]:
    recent_window = min(6, len(monthly))
    recent_avg = float(monthly["Sales"].tail(recent_window).mean())
    recent_growth = (
        monthly["Sales"]
        .pct_change()
        .tail(recent_window)
        .replace([float("inf"), -float("inf")], pd.NA)
        .dropna()
        .mean()
    )
    recent_growth = 0.0 if pd.isna(recent_growth) else float(recent_growth)
    recent_growth = max(min(recent_growth, 0.15), -0.15)

    service_series = monthly.rename(columns={"Month": "Date"})[["Date", "Sales"]].copy()
    try:
        forecast_df, model_key, comparison_df = forecast_with_best_model(
            service_series,
            horizon,
            include_random_forest=include_random_forest,
        )
    except TypeError as exc:
        if "include_random_forest" not in str(exc):
            raise
        forecast_df, model_key, comparison_df = forecast_with_best_model(
            service_series,
            horizon,
        )
    forecast_rows = [
        (
            pd.to_datetime(row["Date"]),
            float(row["Forecast"]),
            float(row.get("Lower", row["Forecast"])),
            float(row.get("Upper", row["Forecast"])),
        )
        for _, row in forecast_df.iterrows()
    ]
    confidence = _forecast_confidence_context(monthly, comparison_df, model_key)
    confidence["ensemble_models"] = forecast_df.attrs.get("ensemble_models", [])

    return forecast_rows, recent_avg, recent_growth, model_key, confidence


def _forecast_path_facts(forecast_rows: List[Tuple[pd.Timestamp, float, float, float]],
                         recent_avg: float,
                         recent_growth: float,
                         confidence: Dict[str, object]) -> List[str]:
    values = [value for _, value, _, _ in forecast_rows]
    if not values:
        return []

    first_month, first_value, _, _ = forecast_rows[0]
    last_month, last_value, _, _ = forecast_rows[-1]
    total_change = ((last_value - first_value) / max(abs(first_value), 1)) * 100
    direction = "rising" if total_change > 2 else ("softening" if total_change < -2 else "mostly stable")
    peak_month, peak_value, _, _ = max(forecast_rows, key=lambda row: row[1])
    low_month, low_value, _, _ = min(forecast_rows, key=lambda row: row[1])

    facts = [
        f"Forecast path is {direction}: {first_month.strftime('%b %Y')} {_format_inr(first_value)} to {last_month.strftime('%b %Y')} {_format_inr(last_value)} ({total_change:+.1f}%).",
        f"Peak forecast month: {peak_month.strftime('%b %Y')} at {_format_inr(peak_value)}; lowest forecast month: {low_month.strftime('%b %Y')} at {_format_inr(low_value)}.",
        f"Recent monthly average context: {_format_inr(recent_avg)}.",
        f"Recent historical growth context: {recent_growth * 100:+.1f}% per month.",
    ]

    ensemble_models = confidence.get("ensemble_models") or []
    if ensemble_models:
        blend_text = ", ".join(
            f"{get_model_label(item['model_key'])} {item['weight'] * 100:.0f}%"
            for item in ensemble_models[:3]
        )
        facts.append(f"Smart Ensemble blend: {blend_text}.")

    return facts


def _find_dimension_filter(query: str, df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    """Find a mentioned category/product/region value in the user's query."""
    q = query.lower()
    preferred_cols = [
        "Category",
        "Product",
        "Product Name",
        "Item",
        "Item Name",
        "SKU",
        "Brand",
        "Region",
    ]
    candidate_cols = [col for col in preferred_cols if col in df.columns]
    candidate_cols += [
        col for col in df.select_dtypes(include=["object", "category"]).columns
        if col not in candidate_cols
    ]

    if "category" in q and "Category" in candidate_cols:
        candidate_cols = ["Category"] + [col for col in candidate_cols if col != "Category"]
    elif "region" in q and "Region" in candidate_cols:
        candidate_cols = ["Region"] + [col for col in candidate_cols if col != "Region"]

    for col in candidate_cols:
        values = (
            df[col]
            .dropna()
            .astype(str)
            .sort_values(key=lambda s: s.str.len(), ascending=False)
            .unique()
        )
        for value in values:
            value_lower = value.lower()
            if value_lower and re.search(rf"(?<!\w){re.escape(value_lower)}(?!\w)", q):
                return col, value

    return None


def _build_single_forecast_answer(data: pd.DataFrame,
                                  horizon: int,
                                  scope_label: str = "overall sales") -> Optional[Dict[str, object]]:
    monthly = _monthly_sales(data)
    if len(monthly) < 3:
        return None

    forecast_rows, recent_avg, recent_growth, model_key, confidence = _forecast_from_monthly(monthly, horizon)
    total_forecast = sum(value for _, value, _, _ in forecast_rows)
    horizon_label = _format_horizon_label(horizon)

    visible_rows = forecast_rows if horizon <= 24 else forecast_rows[:12] + forecast_rows[-3:]
    facts = [
        f"{month.strftime('%b %Y')}: {_format_inr(value)}"
        for month, value, _, _ in visible_rows
    ]
    if horizon > 24:
        facts.insert(12, f"... {horizon - 15} middle months omitted for readability ...")

    horizon_warning = _long_horizon_warning(horizon)
    if horizon_warning:
        facts.append(horizon_warning)

    facts += _forecast_path_facts(forecast_rows, recent_avg, recent_growth, confidence)
    facts += [
        f"Model used: {get_model_label(model_key)}.",
        (
            f"Forecast confidence: {confidence['level']}"
            + (f" ({confidence['score']}/100)." if confidence.get("score") is not None else ".")
        ),
        (
            f"Validation MAPE: {confidence['mape']:.1f}%."
            if confidence.get("mape") is not None else confidence["reason"]
        ),
    ]

    title = f"{scope_label.title()} Forecast"
    headline = f"Expected sales for the next {horizon_label}: {_format_inr(total_forecast)}."
    answer = (
        f"The next {horizon_label} forecast for {scope_label} is approximately "
        f"{_format_inr(total_forecast)} in total."
    )

    return {
        "intent": "forecast",
        "answer": answer,
        "formatted_answer": format_smart_response(
            title,
            headline,
            facts,
            "This chatbot forecast uses the shared forecasting service. Use the Forecasting page for full model comparison and confidence review."
        ),
        "chart": {
            "type": "line",
            "title": f"{scope_label.title()} Forecast",
            "data": [
                {"label": month.strftime("%b %Y"), "sales": float(value)}
                for month, value, _, _ in forecast_rows
            ],
        },
        "table": [
            {
                "Month": month.strftime("%b %Y"),
                "Forecast": round(float(value), 2),
                "Lower": round(float(lower), 2),
                "Upper": round(float(upper), 2),
                "Scope": scope_label,
                "Model": get_model_label(model_key),
                "Confidence": confidence["level"],
            }
            for month, value, lower, upper in forecast_rows
        ],
    }


def _find_product_dimension(df: pd.DataFrame) -> Optional[str]:
    """Pick the most likely product-like grouping column for chatbot questions."""
    preferred_columns = [
        "Product",
        "Product Name",
        "Item",
        "Item Name",
        "SKU",
        "Category",
    ]
    lower_map = {col.lower(): col for col in df.columns}

    for col in preferred_columns:
        if col.lower() in lower_map:
            return lower_map[col.lower()]

    keyword_options = [
        "product",
        "item",
        "sku",
        "category",
        "brand",
    ]
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in keyword_options):
            return col

    return None


def _is_groupwise_forecast_query(query: str) -> bool:
    q = query.lower()
    return (
        detect_query_intent(query) == "forecast"
        and any(term in q for term in [
            "each product",
            "for each product",
            "every product",
            "for every product",
            "all products",
            "by product",
            "product wise",
            "product-wise",
            "each category",
            "for each category",
            "every category",
            "for every category",
            "all categories",
            "by category",
            "category wise",
            "category-wise",
            "categories",
        ])
    )


def _is_monthly_category_query(query: str) -> bool:
    q = query.lower()
    return (
        any(term in q for term in ["every month", "month by month", "monthly", "each month", "by month"])
        and any(term in q for term in ["category", "categories", "by category", "category wise", "category-wise"])
        and any(term in q for term in ["sales", "revenue"])
    )


def _is_category_breakdown_query(query: str) -> bool:
    q = query.lower()
    category_terms = [
        "every category",
        "for every category",
        "each category",
        "for each category",
        "all categories",
        "by category",
        "category wise",
        "category-wise",
    ]
    return any(term in q for term in category_terms)


def _is_region_breakdown_query(query: str) -> bool:
    q = query.lower()
    region_terms = [
        "every region",
        "for every region",
        "each region",
        "for each region",
        "all regions",
        "by region",
        "region wise",
        "region-wise",
        "regions",
    ]
    return any(term in q for term in region_terms) and any(term in q for term in ["sales", "revenue", "performance", "breakdown"])


def _build_monthly_category_answer(data: pd.DataFrame) -> Dict[str, str]:
    if "Category" not in data.columns:
        return {
            "intent": "category_monthly_breakdown",
            "answer": "I cannot show sales by category for each month because the dataset does not include a Category column.",
            "formatted_answer": format_smart_response(
                "Monthly Category Sales",
                "I could not find a Category column in the current dataset.",
                [
                    "This view needs both Date and Category columns.",
                    "I can still answer overall monthly sales questions from the available data.",
                ],
                "Upload a dataset with category values to see month-by-month category breakdowns."
            ),
        }

    temp = data.copy()
    temp["Date"] = pd.to_datetime(temp["Date"])
    temp["Month"] = temp["Date"].dt.to_period("M").dt.to_timestamp()
    grouped = (
        temp.groupby(["Month", "Category"])["Sales"]
        .sum()
        .reset_index()
        .sort_values(["Month", "Sales"], ascending=[True, False])
    )

    month_blocks: List[str] = []
    for month, month_df in grouped.groupby("Month"):
        item_lines = "\n".join(
            f"- {row['Category']}: {_format_inr(row['Sales'])}"
            for _, row in month_df.iterrows()
        )
        month_blocks.append(f"### {pd.to_datetime(month).strftime('%b %Y')}\n{item_lines}")

    strongest_row = grouped.sort_values("Sales", ascending=False).iloc[0]
    latest_month = grouped["Month"].max()
    latest_df = grouped[grouped["Month"] == latest_month]
    latest_leader = latest_df.iloc[0]

    answer = (
        f"I prepared the month-by-month sales breakdown by category. "
        f"The strongest month-category combination is {strongest_row['Category']} in "
        f"{pd.to_datetime(strongest_row['Month']).strftime('%b %Y')} with {_format_inr(strongest_row['Sales'])}. "
        f"In the latest month, {latest_leader['Category']} leads with {_format_inr(latest_leader['Sales'])}."
    )
    formatted_answer = (
        "**Monthly Category Sales**\n\n"
        "Here is the sales breakdown for each month by category.\n\n"
        + "\n\n".join(month_blocks)
    )

    return {
        "intent": "category_monthly_breakdown",
        "answer": answer,
        "formatted_answer": formatted_answer,
    }


def _build_category_breakdown_answer(data: pd.DataFrame) -> Dict[str, str]:
    if "Category" not in data.columns:
        return {
            "intent": "category_breakdown",
            "answer": "I cannot show a category breakdown because the dataset does not include a Category column.",
            "formatted_answer": format_smart_response(
                "Category Breakdown",
                "I could not find a Category column in the current dataset.",
                [
                    "This answer needs category values in the uploaded data.",
                    "I can still answer overall sales questions from the current dataset.",
                ],
                "Upload a dataset with a Category column to see category-wise sales."
            ),
        }

    cat = data.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    total = float(cat.sum()) if len(cat) else 0.0
    facts = [
        f"{name}: {_format_inr(value)} ({(value / total * 100):.1f}% of category revenue)"
        for name, value in cat.items()
    ]
    answer = "Here is the total sales breakdown for every category: " + "; ".join(
        f"{name} {_format_inr(value)}" for name, value in cat.items()
    ) + "."

    return {
        "intent": "category_breakdown",
        "answer": answer,
        "formatted_answer": format_smart_response(
            "Category Breakdown",
            "Here is the sales contribution for every category in the dataset.",
            facts,
            "Use this view to compare which categories contribute most to total revenue."
        ),
    }


def _build_region_breakdown_answer(data: pd.DataFrame) -> Dict[str, str]:
    if "Region" not in data.columns:
        return {
            "intent": "region_breakdown",
            "answer": "I could not find a Region column in the current dataset.",
            "formatted_answer": format_smart_response(
                "Region Breakdown",
                "I could not find a Region column in the current dataset.",
                [
                    "This answer needs region values in the uploaded data.",
                    "I can still answer overall sales questions from the current dataset.",
                ],
                "Upload a dataset with a Region column to see region-wise sales."
            ),
        }

    reg = data.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    total = float(reg.sum()) if len(reg) else 0.0
    facts = [
        f"{name}: {_format_inr(value)} ({(value / total * 100):.1f}% of regional revenue)"
        for name, value in reg.items()
    ]
    answer = "Here is the total sales breakdown for every region: " + "; ".join(
        f"{name} {_format_inr(value)}" for name, value in reg.items()
    ) + "."

    return {
        "intent": "region_breakdown",
        "answer": answer,
        "formatted_answer": format_smart_response(
            "Region Breakdown",
            "Here is the sales contribution for every region in the dataset.",
            facts,
            "Use this view to compare strong and weak regional performance."
        ),
        "chart": {
            "type": "bar",
            "title": "Sales by Region",
            "data": [
                {"label": str(name), "sales": float(value)}
                for name, value in reg.items()
            ],
        },
    }


def _build_ranked_dimension_answer(data: pd.DataFrame,
                                   group_col: str,
                                   mode: str = "best") -> Dict[str, object]:
    ascending = mode == "weakest"
    grouped = data.groupby(group_col)["Sales"].sum().sort_values(ascending=ascending)
    selected_group = grouped.index[0]
    selected_value = grouped.iloc[0]
    total = grouped.sum()
    pct = selected_value / total * 100 if total else 0
    noun = group_col.lower()
    title_mode = "Weakest" if mode == "weakest" else "Best"
    performance_text = "lowest-performing" if mode == "weakest" else "strongest"
    contribution_text = "lowest sales contribution" if mode == "weakest" else "highest sales contribution"

    answer = (
        f"The {performance_text} {noun} is {selected_group}, contributing "
        f"{_format_inr(selected_value)} ({pct:.1f}% of total {noun} revenue)."
    )
    recommendation = (
        f"Review pricing, stock, campaigns, and demand signals for this {noun}."
        if mode == "weakest" else
        f"Use this {noun} as a benchmark for planning and sales strategy."
    )

    return {
        "intent": f"{mode}_{noun}",
        "answer": answer,
        "formatted_answer": format_smart_response(
            f"{title_mode} {group_col}",
            f"{selected_group} is the {performance_text} {noun}.",
            [
                f"Revenue contribution: {_format_inr(selected_value)}.",
                f"Share of total {noun} sales: {pct:.1f}%.",
                f"This {noun} has the {contribution_text} compared with the other {noun} values.",
            ],
            recommendation,
        ),
        "chart": {
            "type": "bar",
            "title": f"Sales by {group_col}",
            "data": [
                {"label": str(name), "sales": float(value)}
                for name, value in grouped.sort_values(ascending=False).items()
            ],
        },
    }


def _missing_dimension_answer(group_col: str) -> Dict[str, str]:
    return {
        "intent": f"missing_{group_col.lower()}",
        "answer": f"I could not find a {group_col} column in the current dataset.",
        "formatted_answer": format_smart_response(
            f"{group_col} Not Found",
            f"I could not find a {group_col} column in the current dataset.",
            [
                f"This answer needs a {group_col} column in the uploaded data.",
                "I can still answer overall sales, trend, forecast, comparison, and anomaly questions.",
            ],
            f"Upload a dataset with {group_col} values to get {group_col.lower()}-level answers."
        ),
    }


def _build_groupwise_forecast(data: pd.DataFrame,
                              group_col: str,
                              horizon: int = 6,
                              max_groups: int = 8) -> Dict[str, str]:
    grouped_sales = data.groupby(group_col)["Sales"].sum().sort_values(ascending=False)
    selected_groups = grouped_sales.head(max_groups).index.tolist()

    fact_lines: List[str] = []
    summary_parts: List[str] = []
    chart_rows: List[Dict[str, object]] = []
    table_rows: List[Dict[str, object]] = []

    for group_value in selected_groups:
        group_df = data[data[group_col] == group_value].copy()
        monthly_group = _monthly_sales(group_df)
        if len(monthly_group) < 2:
            continue

        forecast_rows, recent_avg, recent_growth, model_key, confidence = _forecast_from_monthly(
            monthly_group,
            horizon,
            include_random_forest=False,
        )
        next_values = [value for _, value, _, _ in forecast_rows]

        total_group_forecast = sum(next_values)
        summary_parts.append(f"{group_value}: {_format_inr(total_group_forecast)}")
        chart_rows.append({"label": str(group_value), "sales": float(total_group_forecast)})

        first_month, first_value, _, _ = forecast_rows[0]
        last_month, last_value, _, _ = forecast_rows[-1]
        fact_lines.append(
            f"{group_value}: total {_format_inr(total_group_forecast)} over {_format_horizon_label(horizon)}; "
            f"{first_month.strftime('%b %Y')} {_format_inr(first_value)} to "
            f"{last_month.strftime('%b %Y')} {_format_inr(last_value)} "
            f"using {get_model_label(model_key)} with {confidence['level'].lower()} confidence."
        )
        for month, value, lower, upper in forecast_rows:
            table_rows.append({
                "Group": str(group_value),
                "Month": month.strftime("%b %Y"),
                "Forecast": round(float(value), 2),
                "Lower": round(float(lower), 2),
                "Upper": round(float(upper), 2),
                "Model": get_model_label(model_key),
                "Confidence": confidence["level"],
            })

    if not fact_lines:
        return {
            "answer": f"I could not build a separate forecast by {group_col.lower()} because there is not enough monthly history in each group.",
            "formatted_answer": format_smart_response(
                "Grouped Forecast",
                f"There is not enough group-level history to forecast by {group_col.lower()}.",
                [
                    "Each group needs at least a small monthly history to estimate recent average and growth.",
                ],
                "Try the main Forecasting page for the full dataset, or upload more history for each product."
            ),
        }

    dimension_label = "product" if "product" in group_col.lower() or group_col.lower() in {"item", "item name", "sku"} else group_col.lower()
    horizon_label = _format_horizon_label(horizon)
    horizon_warning = _long_horizon_warning(horizon)
    headline = f"Here is the quick forecast for each {dimension_label} group for the next {horizon_label}."
    answer = f"Built separate {horizon_label} forecasts for {len(fact_lines)} {dimension_label} groups: " + "; ".join(summary_parts) + "."
    recommendation = (
        f"This chatbot forecast uses the shared forecasting service for each {dimension_label}. "
        "Use the Forecasting page for full model comparison and confidence review."
    )
    if horizon_warning:
        recommendation += f" {horizon_warning}"

    return {
        "intent": "forecast",
        "answer": answer,
        "formatted_answer": format_smart_response(
            f"Forecast by {group_col}",
            headline,
            fact_lines,
            recommendation,
        ),
        "chart": {
            "type": "bar",
            "title": f"{horizon_label.title()} Forecast by {group_col}",
            "data": chart_rows,
        },
        "table": table_rows,
    }


def format_smart_response(title: str,
                          headline: str,
                          facts: List[str],
                          recommendation: str = "") -> str:
    """Create a concise, question-specific chatbot answer."""
    fact_lines = "\n".join(f"- {fact}" for fact in facts if fact)
    response = f"**{title}**\n\n**Answer**\n\n{headline}"
    if fact_lines:
        response += f"\n\n**Key details**\n\n{fact_lines}"
    if recommendation:
        response += f"\n\n**What to do next**\n\n{recommendation}"
    return response


def sanitize_chatbot_answer(text: str) -> str:
    """Collapse any legacy analytical template into a short direct answer."""
    if not text:
        return text

    text = "\n".join(repair_text_encoding(line) for line in text.splitlines())

    legacy_markers = [
        "Data Understanding",
        "Data Preprocessing",
        "Model Insight & Prediction",
    ]
    if not any(marker in text for marker in legacy_markers):
        return text.strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    conclusion = ""
    facts: List[str] = []
    in_model_section = False
    skip_terms = ("missing values", "duplicate rows", "date fields", "numeric sales strings", "outliers are detected", "pipeline helps")

    for line in lines:
        if "### 3. Model Insight & Prediction" in line:
            in_model_section = True
            continue
        if line.startswith("### "):
            continue
        if not line.startswith("- "):
            continue

        item = line[2:].strip()
        if any(term in item.lower() for term in skip_terms):
            continue
        if "grounded in the available data" in item.lower():
            continue

        if in_model_section and not conclusion:
            conclusion = item
            continue

        if len(facts) < 3:
            facts.append(item)

    if not conclusion:
        conclusion = "Here is the direct answer from your sales data."

    return format_smart_response("Direct Answer", conclusion, facts)


def _build_project_info_answer(query: str, data: pd.DataFrame) -> Dict[str, object]:
    q = query.lower()
    total_rows = len(data)
    total_cols = data.shape[1]
    loaded_context = (
        f"The currently loaded dataset has {total_rows:,} rows and {total_cols:,} columns."
        if total_rows else
        "No uploaded dataset is currently available."
    )

    if any(term in q for term in ["module", "workflow", "flow", "pipeline"]):
        headline = "The project workflow moves from upload to cleaning, analysis, forecasting, simulation, and chatbot answers."
        facts = [
            "Data Upload validates CSV/Excel files, standardizes Date and Sales, cleans duplicates/missing values, and stores the cleaned dataset.",
            "Sales Analysis explains trends, categories, regions, anomalies, and correlations.",
            "Forecasting compares models, creates confidence intervals, and exports month-by-month predictions.",
            "Scenario Simulation applies business levers such as growth, seasonality, discounts, marketing, churn, and channel impact.",
            "Chatbot Insights answers data, forecast, model, and project questions using deterministic calculations first.",
        ]
    elif any(term in q for term in ["tech", "technology", "stack", "library", "libraries"]):
        headline = "The project is a Python and Streamlit sales forecasting application."
        facts = [
            "Frontend and app shell: Streamlit.",
            "Data work: Pandas and NumPy.",
            "Charts: Plotly.",
            "Forecasting and ML: custom time-series functions plus scikit-learn Random Forest.",
            "Optional AI explanation layer: Gemini API through REST requests.",
        ]
    else:
        headline = "This project turns sales data into analysis, forecasts, scenarios, and chatbot insights."
        facts = [
            "It accepts sales data with Date and Sales fields, plus optional business dimensions such as Category, Region, Product, SKU, or Brand.",
            "It cleans and prepares the data before analysis so forecasting uses a consistent monthly sales series.",
            "It compares multiple forecasting approaches and recommends the strongest measured option.",
            "It gives business-facing outputs: charts, tables, confidence ranges, model explanations, scenario estimates, and CSV exports.",
            loaded_context,
        ]

    return {
        "intent": "project_info",
        "answer": headline,
        "formatted_answer": format_smart_response(
            "Project Answer",
            headline,
            facts,
            "Use Data Upload first, then Analysis and Forecasting. The chatbot can answer quick questions across those same project areas."
        ),
    }


def _build_data_info_answer(query: str, data: pd.DataFrame, monthly: pd.DataFrame) -> Dict[str, object]:
    q = query.lower()
    missing_total = int(data.isna().sum().sum())
    duplicate_rows = int(data.duplicated().sum())
    date_min = data["Date"].min().strftime("%d %b %Y")
    date_max = data["Date"].max().strftime("%d %b %Y")
    numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
    text_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()

    if any(term in q for term in ["column", "columns", "field", "fields", "schema"]):
        facts = [
            f"Columns detected: {', '.join(map(str, data.columns))}.",
            f"Numeric columns: {', '.join(numeric_cols) if numeric_cols else 'none detected'}.",
            f"Text/category columns: {', '.join(text_cols) if text_cols else 'none detected'}.",
            "Required project columns are Date and Sales; optional dimensions improve category, region, product, and group-wise answers.",
        ]
        headline = f"The dataset currently has {data.shape[1]} columns."
    elif any(term in q for term in ["missing", "duplicate", "quality", "clean"]):
        facts = [
            f"Missing cells currently present: {missing_total:,}.",
            f"Duplicate rows currently present: {duplicate_rows:,}.",
            "The upload workflow standardizes dates, converts sales-like strings to numeric values, removes duplicate rows, fills simple missing values, and clips invalid negative sales to zero.",
            "Forecasting reliability improves when monthly history is complete and outlier months are explained.",
        ]
        headline = "Here is the current data-quality snapshot."
    else:
        facts = [
            f"Rows: {len(data):,}.",
            f"Columns: {data.shape[1]:,}.",
            f"Date range: {date_min} to {date_max}.",
            f"Monthly periods available: {len(monthly):,}.",
            f"Total revenue: {_format_inr(float(data['Sales'].sum()))}.",
            f"Average monthly revenue: {_format_inr(float(monthly['Sales'].mean()))}.",
        ]
        headline = "Here is the current dataset profile."

    return {
        "intent": "data_info",
        "answer": headline,
        "formatted_answer": format_smart_response(
            "Data Answer",
            headline,
            facts,
            "For better forecasts, use at least 12 months of clean history; 24+ months is better for seasonality."
        ),
    }


def _build_recommendations_answer(data: pd.DataFrame, monthly: pd.DataFrame) -> Dict[str, object]:
    total = float(data["Sales"].sum())
    latest = monthly.iloc[-1]
    best_idx = monthly["Sales"].idxmax()
    worst_idx = monthly["Sales"].idxmin()
    best_month = monthly.loc[best_idx]
    worst_month = monthly.loc[worst_idx]
    monthly_avg = float(monthly["Sales"].mean())

    recent_window = monthly["Sales"].iloc[-min(3, len(monthly)):]
    earlier_window = monthly["Sales"].iloc[:min(3, len(monthly))]
    recent_avg = float(recent_window.mean())
    earlier_avg = float(earlier_window.mean())
    trend_pct = ((recent_avg - earlier_avg) / earlier_avg * 100) if earlier_avg else 0.0
    trend_word = "growing" if trend_pct > 5 else ("declining" if trend_pct < -5 else "stable")

    facts = [
        f"Total revenue is {_format_inr(total)} across {len(monthly):,} monthly periods.",
        f"Recent sales look {trend_word}: recent monthly average {_format_inr(recent_avg)} vs early-period average {_format_inr(earlier_avg)} ({trend_pct:+.1f}%).",
        f"Best month was {best_month['Month'].strftime('%b %Y')} with {_format_inr(float(best_month['Sales']))}; weakest month was {worst_month['Month'].strftime('%b %Y')} with {_format_inr(float(worst_month['Sales']))}.",
        f"Latest month is {latest['Month'].strftime('%b %Y')} with {_format_inr(float(latest['Sales']))}, compared with average monthly revenue of {_format_inr(monthly_avg)}.",
    ]

    recommendations = []

    if "Category" in data.columns:
        category_sales = data.groupby("Category")["Sales"].sum().sort_values(ascending=False)
        top_category = category_sales.index[0]
        category_share = float(category_sales.iloc[0] / total * 100) if total else 0.0
        facts.append(f"Top category is {top_category} with {_format_inr(float(category_sales.iloc[0]))} ({category_share:.1f}% of revenue).")
        recommendations.append(
            f"Prioritize inventory, offers, and campaign budget for {top_category}, because it is the strongest revenue category."
        )

        if len(category_sales) > 1:
            weak_category = category_sales.index[-1]
            recommendations.append(
                f"Review pricing, visibility, and demand issues for {weak_category}; decide whether to improve it or reduce focus."
            )

    if "Region" in data.columns:
        region_sales = data.groupby("Region")["Sales"].sum().sort_values(ascending=False)
        top_region = region_sales.index[0]
        region_share = float(region_sales.iloc[0] / total * 100) if total else 0.0
        facts.append(f"Top region is {top_region} with {_format_inr(float(region_sales.iloc[0]))} ({region_share:.1f}% of revenue).")
        recommendations.append(
            f"Use {top_region} as the benchmark region and compare its product mix, pricing, and campaigns with weaker regions."
        )

    if trend_pct > 5:
        recommendations.append("Protect the current growth by planning stock ahead of demand and repeating the campaigns behind recent strong months.")
    elif trend_pct < -5:
        recommendations.append("Investigate the recent decline quickly: check stockouts, pricing changes, campaign gaps, and competitor activity.")
    else:
        recommendations.append("Since sales are stable, use targeted promotions or bundling to create growth instead of relying only on baseline demand.")

    recommendations.append(
        f"Study what worked in {best_month['Month'].strftime('%b %Y')} and what went wrong in {worst_month['Month'].strftime('%b %Y')} before setting the next sales plan."
    )
    recommendations.append("Use the Forecasting page to plan inventory and monthly revenue targets for the next 3-6 months.")

    return {
        "intent": "recommendations",
        "answer": "Here are practical business recommendations based on the loaded sales data.",
        "formatted_answer": format_smart_response(
            "Business Recommendations",
            "Focus on the strongest revenue drivers, investigate weak periods, and use the forecast for planning.",
            facts,
            "\n".join(f"- {item}" for item in recommendations),
        ),
    }


def _build_model_info_answer(query: str, monthly: pd.DataFrame) -> Dict[str, object]:
    q = query.lower()

    metric_facts = {
        "mape": "MAPE is Mean Absolute Percentage Error. Lower is better; 10% means the forecast is off by about 10% on average.",
        "mae": "MAE is Mean Absolute Error. It shows the average rupee difference between actual and predicted sales.",
        "rmse": "RMSE is Root Mean Squared Error. It penalizes large mistakes more strongly than MAE.",
        "r2": "R2 shows how much variation the model explains. Closer to 1.0 is better, but time-series forecasts should still be judged with error metrics.",
        "confidence": "The confidence interval is the lower-to-upper forecast range. Wider ranges mean more uncertainty.",
    }
    metric_hits = []
    if "mape" in q:
        metric_hits.append(metric_facts["mape"])
    if "mae" in q:
        metric_hits.append(metric_facts["mae"])
    if "rmse" in q:
        metric_hits.append(metric_facts["rmse"])
    if any(term in q for term in ["r2", "r-squared", "r squared"]):
        metric_hits.append(metric_facts["r2"])
    if "confidence" in q or "interval" in q:
        metric_hits.append(metric_facts["confidence"])

    if metric_hits:
        headline = "Here is how the model metric works."
        facts = metric_hits
    elif any(term in q for term in ["best", "selected", "recommended", "accuracy", "compare", "comparison", "performance"]):
        try:
            comparison_df = compare_forecast_models(
                monthly.rename(columns={"Month": "Date"})[["Date", "Sales"]],
                include_random_forest=False,
            )
        except Exception:
            comparison_df = pd.DataFrame()

        if comparison_df.empty:
            headline = "There is not enough monthly history to compare models reliably."
            facts = [
                "The app needs a reasonable monthly series to evaluate models on held-out periods.",
                "With more history, it compares forecast error using MAE, RMSE, MAPE, and R2.",
            ]
        else:
            best = comparison_df.iloc[0]
            facts = [
                f"Best measured model right now: {get_model_label(str(best['model_key']))}.",
                f"Validation MAPE: {float(best['MAPE (%)']):.1f}%.",
                f"MAE: {_format_inr(float(best['MAE']))}.",
                f"RMSE: {_format_inr(float(best['RMSE']))}.",
                "Lower MAPE/MAE/RMSE indicates better measured accuracy on recent validation data.",
            ]
            headline = f"{get_model_label(str(best['model_key']))} is currently the strongest measured model."
    else:
        headline = "The forecasting engine supports several model types."
        facts = [
            "Smart Ensemble blends the strongest recent models using inverse-error weights and is the recommended default.",
            "Moving Average predicts from the recent average; it is stable but can be flat.",
            "Linear Trend extends an upward or downward straight-line pattern.",
            "Seasonal Trend keeps yearly seasonal shape while adjusting for recent year-over-year growth.",
            "Seasonal Naive is kept as a lightweight baseline, but it also receives a growth adjustment to avoid copying last year exactly.",
            "Exponential Smoothing emphasizes recent months.",
            "Holt's Double Smoothing tracks level plus trend.",
            "Random Forest uses lag, rolling, and calendar features when enough history exists.",
        ]

    return {
        "intent": "model_info",
        "answer": headline,
        "formatted_answer": format_smart_response(
            "Model Answer",
            headline,
            facts,
            "For planning, use Smart Ensemble first, then check the model comparison and confidence band before making decisions."
        ),
    }


def _build_general_smart_answer(query: str, data: pd.DataFrame, monthly: pd.DataFrame) -> Dict[str, object]:
    """Give a useful grounded answer when the exact question is broad or unclear."""
    total_sales = float(data["Sales"].sum())
    latest = monthly.iloc[-1]
    best_idx = monthly["Sales"].idxmax()
    worst_idx = monthly["Sales"].idxmin()
    category_hint = "Category" if "Category" in data.columns else None
    region_hint = "Region" if "Region" in data.columns else None

    facts = [
        f"Loaded data: {len(data):,} rows, {data.shape[1]:,} columns, {len(monthly):,} monthly periods.",
        f"Total revenue: {_format_inr(total_sales)}.",
        f"Latest month: {latest['Month'].strftime('%b %Y')} with {_format_inr(float(latest['Sales']))}.",
        f"Best month: {monthly.loc[best_idx, 'Month'].strftime('%b %Y')} with {_format_inr(float(monthly.loc[best_idx, 'Sales']))}.",
        f"Weakest month: {monthly.loc[worst_idx, 'Month'].strftime('%b %Y')} with {_format_inr(float(monthly.loc[worst_idx, 'Sales']))}.",
    ]

    if category_hint:
        top_category = data.groupby(category_hint)["Sales"].sum().sort_values(ascending=False).index[0]
        facts.append(f"Top category available to ask about: {top_category}.")
    if region_hint:
        top_region = data.groupby(region_hint)["Sales"].sum().sort_values(ascending=False).index[0]
        facts.append(f"Top region available to ask about: {top_region}.")

    return {
        "intent": "general_help",
        "answer": "I can answer this, but the question is broad, so here is the clearest grounded summary first.",
        "formatted_answer": format_smart_response(
            "Smart Answer",
            "Your data is loaded, and I can answer sales, forecast, model, data-quality, and project questions directly.",
            facts,
            (
                "Ask a specific follow-up like: forecast next 6 months, best category, weakest region, "
                "show data quality, which model is best, explain MAPE, or what does this project do."
            ),
        ),
    }


def answer_from_dataframe(query: str, df: pd.DataFrame) -> Optional[Dict[str, str]]:
    if df is None or df.empty or "Date" not in df.columns or "Sales" not in df.columns:
        return None

    intent = detect_query_intent(query)
    data = normalize_text_columns(df)
    data["Date"] = pd.to_datetime(data["Date"])
    monthly = _monthly_sales(data)
    horizon = _parse_forecast_horizon(query)

    if intent == "project_info":
        return _build_project_info_answer(query, data)

    if intent == "data_info":
        return _build_data_info_answer(query, data, monthly)

    if intent == "model_info":
        return _build_model_info_answer(query, monthly)

    if intent == "recommendations":
        return _build_recommendations_answer(data, monthly)

    if intent == "forecast":
        q = query.lower()
        dimension_match = _find_dimension_filter(query, data)

        if dimension_match and not _is_groupwise_forecast_query(query):
            group_col, group_value = dimension_match
            scoped_data = data[data[group_col].astype(str).str.lower() == group_value.lower()].copy()
            scoped_forecast = _build_single_forecast_answer(
                scoped_data,
                horizon,
                scope_label=f"{group_value} {group_col.lower()}",
            )
            if scoped_forecast:
                return scoped_forecast

            return {
                "intent": "forecast",
                "answer": f"I found {group_value} in {group_col}, but there is not enough monthly history to forecast it.",
                "formatted_answer": format_smart_response(
                    f"{group_value} Forecast",
                    f"I found {group_value} in {group_col}, but there is not enough monthly history to build a forecast.",
                    [
                        "A category or product needs at least 3 monthly observations for the chatbot's quick forecast.",
                        "Upload more history or use the full Forecasting page for broader model options.",
                    ],
                ),
            }

        if _is_groupwise_forecast_query(query):
            if "category" in q and "Category" in data.columns:
                group_col = "Category"
            elif "region" in q and "Region" in data.columns:
                group_col = "Region"
            else:
                group_col = _find_product_dimension(data)

            if group_col:
                grouped = _build_groupwise_forecast(data, group_col=group_col, horizon=horizon)
                return {
                    "intent": "forecast",
                    "answer": grouped["answer"],
                    "formatted_answer": grouped["formatted_answer"],
                    "chart": grouped.get("chart"),
                    "table": grouped.get("table"),
                }

            return {
                "intent": "forecast",
                "answer": "I can forecast by group only if the dataset includes a product-like, category, or region column.",
                "formatted_answer": format_smart_response(
                    "Grouped Forecast",
                    "I could not find a product-like, category, or region column in the current dataset.",
                    [
                        "Expected columns include Category, Product, Product Name, Item, SKU, Brand, or Region.",
                        "Right now I only have total-sales history, so the chatbot can produce an overall forecast but not a group-wise one.",
                    ],
                    "Upload a dataset with a grouping column to get separate forecasts."
                ),
            }

        forecast_answer = _build_single_forecast_answer(data, horizon, scope_label="overall sales")
        if forecast_answer:
            return forecast_answer
        return None

    if _is_monthly_category_query(query):
        return _build_monthly_category_answer(data)

    if _is_category_breakdown_query(query):
        return _build_category_breakdown_answer(data)

    if _is_region_breakdown_query(query):
        return _build_region_breakdown_answer(data)

    if intent == "summary":
        total = data["Sales"].sum()
        avg = data["Sales"].mean()
        best_idx = monthly["Sales"].idxmax()
        worst_idx = monthly["Sales"].idxmin()
        latest = monthly.iloc[-1]
        answer = (
            f"Your dataset covers {len(data):,} records from "
            f"{data['Date'].min().strftime('%d %b %Y')} to {data['Date'].max().strftime('%d %b %Y')}. "
            f"Total revenue is {_format_inr(total)} and average daily revenue is {_format_inr(avg)}. "
            f"Best month is {monthly.loc[best_idx, 'Month'].strftime('%b %Y')} "
            f"with {_format_inr(monthly.loc[best_idx, 'Sales'])}, while the weakest month is "
            f"{monthly.loc[worst_idx, 'Month'].strftime('%b %Y')} "
            f"with {_format_inr(monthly.loc[worst_idx, 'Sales'])}."
        )
        return {
            "intent": intent,
            "answer": answer,
            "formatted_answer": format_smart_response(
                "Overall Sales Summary",
                f"Your sales data covers {len(data):,} records and totals {_format_inr(total)}.",
                [
                    f"Date range: {data['Date'].min().strftime('%d %b %Y')} to {data['Date'].max().strftime('%d %b %Y')}.",
                    f"Average daily revenue: {_format_inr(avg)}.",
                    f"Best month: {monthly.loc[best_idx, 'Month'].strftime('%b %Y')} with {_format_inr(monthly.loc[best_idx, 'Sales'])}.",
                    f"Weakest month: {monthly.loc[worst_idx, 'Month'].strftime('%b %Y')} with {_format_inr(monthly.loc[worst_idx, 'Sales'])}.",
                    f"Latest month: {latest['Month'].strftime('%b %Y')} with {_format_inr(latest['Sales'])}.",
                ],
                "Use the best month as a benchmark and review the weakest month for demand, pricing, or operational issues."
            )
        }

    if intent == "trend":
        if len(monthly) < 2:
            return None
        first_3 = monthly["Sales"].iloc[:min(3, len(monthly))].mean()
        last_3 = monthly["Sales"].iloc[-min(3, len(monthly)):].mean()
        pct = ((last_3 - first_3) / first_3 * 100) if first_3 else 0.0
        direction = "upward" if pct > 0 else ("downward" if pct < 0 else "flat")
        answer = (
            f"The overall revenue trend is {direction}. "
            f"The last 3-month average is {_format_inr(last_3)} versus {_format_inr(first_3)} "
            f"for the first 3-month average, which is a change of {pct:+.1f}%."
        )
        return {
            "intent": intent,
            "answer": answer,
            "formatted_answer": format_smart_response(
                "Sales Trend",
                f"Revenue is currently showing a {direction} direction.",
                [
                    f"First-period average: {_format_inr(first_3)}.",
                    f"Recent-period average: {_format_inr(last_3)}.",
                    f"Overall movement: {pct:+.1f}%.",
                ],
                "Keep monitoring recent months because the latest trend carries more weight for forecasting."
            )
        }

    if intent == "weakest_category" and "Category" in data.columns:
        cat = data.groupby("Category")["Sales"].sum().sort_values(ascending=True)
        weak_cat = cat.index[0]
        val = cat.iloc[0]
        total = cat.sum()
        pct = val / total * 100 if total else 0
        answer = f"The weakest category is {weak_cat}, contributing {_format_inr(val)} ({pct:.1f}% of total category revenue)."
        return {
            "intent": intent,
            "answer": answer,
            "formatted_answer": format_smart_response(
                "Weakest Category",
                f"{weak_cat} is the lowest-performing category.",
                [
                    f"Revenue contribution: {_format_inr(val)}.",
                    f"Share of total category sales: {pct:.1f}%.",
                    "This category has the lowest sales contribution compared with the other categories.",
                ],
                "Review pricing, stock availability, campaign support, and demand patterns for this category."
            ),
            "chart": {
                "type": "bar",
                "title": "Category Sales",
                "data": [
                    {"label": str(name), "sales": float(value)}
                    for name, value in cat.sort_values(ascending=False).items()
                ],
            },
        }

    if intent == "weakest_category":
        return _missing_dimension_answer("Category")

    if intent == "best_category" and "Category" in data.columns:
        cat = data.groupby("Category")["Sales"].sum().sort_values(ascending=False)
        top_cat = cat.index[0]
        val = cat.iloc[0]
        pct = val / cat.sum() * 100 if cat.sum() else 0
        answer = f"The top category is {top_cat}, contributing {_format_inr(val)} ({pct:.1f}% of total revenue)."
        return {
            "intent": intent,
            "answer": answer,
            "formatted_answer": format_smart_response(
                "Best Category",
                f"{top_cat} is the strongest category.",
                [
                    f"Revenue contribution: {_format_inr(val)}.",
                    f"Share of total category sales: {pct:.1f}%.",
                    "This category is currently the clearest demand driver in the dataset.",
                ],
                "Prioritize stock, campaigns, and forecasting attention around this category."
            )
        }

    if intent == "best_category":
        return _missing_dimension_answer("Category")

    if intent in ("best_product", "weakest_product"):
        product_col = _find_product_dimension(data)
        if product_col:
            mode = "weakest" if intent == "weakest_product" else "best"
            return _build_ranked_dimension_answer(data, product_col, mode=mode)
        return {
            "intent": intent,
            "answer": "I could not find a product-like column in the current dataset.",
            "formatted_answer": format_smart_response(
                "Product Not Found",
                "I could not find a product-like column in the current dataset.",
                [
                    "Expected columns include Product, Product Name, Item, SKU, Brand, or Category.",
                    "I can still answer overall sales, category, region, trend, forecast, and anomaly questions if those columns exist.",
                ],
                "Upload product-level data to get product performance answers."
            ),
        }

    if intent == "weakest_region" and "Region" in data.columns:
        reg = data.groupby("Region")["Sales"].sum().sort_values(ascending=True)
        weak_reg = reg.index[0]
        val = reg.iloc[0]
        total = reg.sum()
        pct = val / total * 100 if total else 0
        answer = f"The weakest region is {weak_reg}, contributing {_format_inr(val)} ({pct:.1f}% of total regional revenue)."
        return {
            "intent": intent,
            "answer": answer,
            "formatted_answer": format_smart_response(
                "Weakest Region",
                f"{weak_reg} is the lowest-performing region.",
                [
                    f"Revenue contribution: {_format_inr(val)}.",
                    f"Share of total regional sales: {pct:.1f}%.",
                    "This region has the lowest sales contribution compared with the other regions.",
                ],
                "Review demand, coverage, campaigns, and local operating factors for this region."
            ),
            "chart": {
                "type": "bar",
                "title": "Region Sales",
                "data": [
                    {"label": str(name), "sales": float(value)}
                    for name, value in reg.sort_values(ascending=False).items()
                ],
            },
        }

    if intent == "weakest_region":
        return _missing_dimension_answer("Region")

    if intent == "best_region" and "Region" in data.columns:
        reg = data.groupby("Region")["Sales"].sum().sort_values(ascending=False)
        top_reg = reg.index[0]
        val = reg.iloc[0]
        pct = val / reg.sum() * 100 if reg.sum() else 0
        answer = f"The top region is {top_reg}, contributing {_format_inr(val)} ({pct:.1f}% of total revenue)."
        return {
            "intent": intent,
            "answer": answer,
            "formatted_answer": format_smart_response(
                "Best Region",
                f"{top_reg} is the strongest region.",
                [
                    f"Revenue contribution: {_format_inr(val)}.",
                    f"Share of total regional sales: {pct:.1f}%.",
                    "This region is performing best compared with the other regions in the dataset.",
                ],
                "Use this region as a reference point for sales strategy in weaker regions."
            )
        }

    if intent == "best_region":
        return _missing_dimension_answer("Region")

    if intent == "weakest_month":
        worst_idx = monthly["Sales"].idxmin()
        worst_month = monthly.loc[worst_idx, "Month"].strftime("%b %Y")
        worst_val = monthly.loc[worst_idx, "Sales"]
        answer = f"The weakest month is {worst_month} with total revenue of {_format_inr(worst_val)}."
        return {
            "intent": intent,
            "answer": answer,
            "formatted_answer": format_smart_response(
                "Weakest Month",
                f"{worst_month} is the lowest-performing month.",
                [
                    f"Revenue in that month: {_format_inr(worst_val)}.",
                    "This is the lowest monthly sales point in the available history.",
                ],
                "Review this month for seasonality, stock issues, campaign gaps, pricing changes, or unusual demand drops."
            )
        }

    if intent == "best_month":
        best_idx = monthly["Sales"].idxmax()
        best_month = monthly.loc[best_idx, "Month"].strftime("%b %Y")
        best_val = monthly.loc[best_idx, "Sales"]
        answer = f"The best month is {best_month} with total revenue of {_format_inr(best_val)}."
        return {
            "intent": intent,
            "answer": answer,
            "formatted_answer": format_smart_response(
                "Best Month",
                f"{best_month} is the best-performing month.",
                [
                    f"Revenue in that month: {_format_inr(best_val)}.",
                    "This is the peak monthly sales point in the available history.",
                ],
                "Check what happened in this month, such as campaign timing, seasonal demand, or regional/category performance."
            )
        }

    if intent == "comparison":
        if len(monthly) < 2:
            return None

        q = query.lower()
        if "last 3" in q or "three" in q:
            if len(monthly) < 4:
                return None
            recent = monthly["Sales"].iloc[-3:].mean()
            prev = monthly["Sales"].iloc[-6:-3].mean() if len(monthly) >= 6 else monthly["Sales"].iloc[:-3].mean()
            pct = ((recent - prev) / prev * 100) if prev else 0.0
            direction = "increased" if pct > 0 else ("decreased" if pct < 0 else "stayed flat")
            chart_data = [
                {"label": "Previous period", "sales": float(prev)},
                {"label": "Recent 3 months", "sales": float(recent)},
            ]
            answer = (
                f"The last 3-month average revenue is {_format_inr(recent)} compared with "
                f"{_format_inr(prev)} in the previous comparison period, a change of {pct:+.1f}%."
            )
        else:
            current = monthly["Sales"].iloc[-1]
            previous = monthly["Sales"].iloc[-2]
            current_month = monthly["Month"].iloc[-1].strftime("%b %Y")
            previous_month = monthly["Month"].iloc[-2].strftime("%b %Y")
            pct = ((current - previous) / previous * 100) if previous else 0.0
            direction = "increased" if pct > 0 else ("decreased" if pct < 0 else "stayed flat")
            answer = (
                f"The latest month generated {_format_inr(current)} versus {_format_inr(previous)} "
                f"in the previous month, which is a change of {pct:+.1f}%."
            )
            chart_data = [
                {"label": previous_month, "sales": float(previous)},
                {"label": current_month, "sales": float(current)},
            ]

        return {
            "intent": intent,
            "answer": answer,
            "formatted_answer": format_smart_response(
                "Month Comparison",
                f"Sales {direction} by {pct:+.1f}% compared with the previous month." if "current_month" in locals() else "Here is the comparison for the selected period.",
                [
                    f"{current_month}: {_format_inr(current)}." if "current_month" in locals() else f"Recent 3-month average: {_format_inr(recent)}.",
                    f"{previous_month}: {_format_inr(previous)}." if "previous_month" in locals() else f"Previous comparison average: {_format_inr(prev)}.",
                    f"Difference: {pct:+.1f}%.",
                ]
            ),
            "chart": {
                "type": "bar",
                "title": "Sales Difference",
                "data": chart_data,
            }
        }

    if intent == "anomaly":
        anomaly_df = detect_anomalies(
            monthly.rename(columns={"Month": "Date"}),
            method="rolling_deviation",
            threshold=2.5,
            rolling_window=6,
        )
        if anomaly_df.empty:
            answer = "No major anomalies were detected with the current threshold."
        else:
            top = anomaly_df.sort_values("anomaly_score", ascending=False).iloc[0]
            answer = (
                f"I detected {len(anomaly_df)} anomalies. The strongest anomaly occurred on "
                f"{pd.to_datetime(top['Date']).strftime('%d %b %Y')} with value {_format_inr(top['Sales'])} "
                f"and {top['severity']} severity."
            )

        return {
            "intent": intent,
            "answer": answer,
            "formatted_answer": format_smart_response(
                "Anomaly Check",
                "I checked the monthly sales series for unusual spikes or drops.",
                [
                    answer,
                    "Anomalies are detected from monthly sales movement using rolling-deviation logic.",
                ],
                "Review anomaly months before final forecasting because unusual spikes can distort model accuracy."
            )
        }

    if intent == "scenario":
        total = data["Sales"].sum()
        monthly_avg = monthly["Sales"].mean()
        uplift_total = total * 1.20
        uplift_monthly = monthly_avg * 1.20
        extra_total = uplift_total - total
        answer = (
            f"If sales increase by 20%, total revenue would move from {_format_inr(total)} "
            f"to {_format_inr(uplift_total)}, adding about {_format_inr(extra_total)}."
        )
        return {
            "intent": intent,
            "answer": answer,
            "formatted_answer": format_smart_response(
                "20% Sales Increase Scenario",
                f"A 20% uplift would add roughly {_format_inr(extra_total)} in revenue.",
                [
                    f"Current total revenue: {_format_inr(total)}.",
                    f"Scenario total revenue: {_format_inr(uplift_total)}.",
                    f"Current average monthly revenue: {_format_inr(monthly_avg)}.",
                    f"Scenario average monthly revenue: {_format_inr(uplift_monthly)}.",
                ],
                "Use this scenario to estimate campaign targets, capacity planning, and inventory requirements."
            )
        }

    return _build_general_smart_answer(query, data, monthly)


def build_hybrid_chat_response(query: str,
                               df: pd.DataFrame,
                               api_key: Optional[str] = None,
                               conversation_messages: Optional[List[Dict]] = None,
                               model: str = "gemini-2.5-flash",
                               max_output_tokens: int = 1024) -> Dict[str, str]:
    """
    Hybrid pipeline:
    1. detect intent
    2. compute direct answer from dataframe when possible
    3. format response in a question-specific style
    4. use Gemini only for broader explanation when available
    """
    conversation_messages = conversation_messages or []
    intent = detect_query_intent(query)
    direct = answer_from_dataframe(query, df)

    def _is_gemini_failure(text: str) -> bool:
        if not text:
            return False
        lower = text.lower()
        failure_markers = [
            "gemini api error",
            "connection error",
            "temporarily unavailable",
            "timed out",
            "rate limit",
            "could not be reached",
            "invalid gemini api key",
            "permission denied",
            "status 503",
            "status 429",
            "status 500",
        ]
        return any(marker in lower for marker in failure_markers)

    graceful_fallback = (
        "**Gemini is temporarily unavailable**\n\n"
        "I could not complete the free-form AI explanation just now, but your data is still loaded and ready.\n\n"
        "- Try a direct question about data summary, columns, cleaning, quality, trend, best category, best region, monthly comparison, anomalies, forecast, model metrics, or project workflow.\n"
        "- You can also use the smart prompt buttons above for answers that do not depend on Gemini.\n"
        "- Most data, model, and project questions can now be answered directly without Gemini."
    )

    # Direct dataset questions stay local so answers remain short and question-specific.
    if direct is not None:
        return {
            "intent": direct["intent"],
            "mode": "direct_data",
            "raw_answer": direct["answer"],
            "final_answer": sanitize_chatbot_answer(direct["formatted_answer"]),
            "chart": direct.get("chart"),
            "table": direct.get("table"),
        }

    # No API key and no direct answer
    if not api_key:
        fallback = (
            "**Clear Answer**\n\n"
            "**Answer**\n\n"
            "I need either loaded sales data or a Gemini API key to answer this fully.\n\n"
            "**Key details**\n\n"
            "- With loaded data, I can answer summary, trends, comparisons, categories, regions, forecasts, models, data quality, scenarios, and anomalies.\n"
            "- With Gemini, I can also answer broader free-form business questions.\n\n"
            "**What to do next**\n\n"
            "Upload a dataset or ask a direct project/model question such as: what does this project do, explain MAPE, or how is the data cleaned."
        )
        return {
            "intent": intent,
            "mode": "llm_only",
            "raw_answer": "",
            "final_answer": sanitize_chatbot_answer(fallback),
        }

    data_summary = build_data_summary(df)
    system_prompt = build_system_prompt(data_summary)

    if direct is not None:
        grounding_text = (
            f"User question: {query}\n\n"
            f"Detected intent: {direct['intent']}\n"
            f"Directly computed answer from dataset:\n{direct['answer']}\n\n"
            f"Rewrite this as a concise chatbot answer. Start with the direct conclusion, "
            f"then add 2-4 useful bullets and one practical recommendation if relevant. "
            f"Do not force a fixed three-section format. Do not change the numbers or conclusions."
        )

        messages = trim_history(conversation_messages + [{"role": "user", "content": grounding_text}])
        final_answer = call_gemini_api(
            messages,
            system_prompt,
            api_key,
            model=model,
            max_output_tokens=max_output_tokens
        )

        if final_answer.startswith("❌") or final_answer.startswith("⚠️"):
            return {
                "intent": direct["intent"],
                "mode": "direct_data",
                "raw_answer": direct["answer"],
                "final_answer": sanitize_chatbot_answer(direct["formatted_answer"]),
            }

        return {
            "intent": direct["intent"],
            "mode": "hybrid",
            "raw_answer": direct["answer"],
            "final_answer": sanitize_chatbot_answer(final_answer),
        }

    llm_messages = trim_history(conversation_messages + [{"role": "user", "content": query}])
    final_answer = call_gemini_api(
        llm_messages,
        system_prompt,
        api_key,
        model=model,
        max_output_tokens=max_output_tokens
    )

    if _is_gemini_failure(final_answer):
        return {
            "intent": intent,
            "mode": "llm_only",
            "raw_answer": "",
            "final_answer": graceful_fallback,
        }

    return {
        "intent": intent,
        "mode": "llm_only",
        "raw_answer": "",
        "final_answer": sanitize_chatbot_answer(final_answer),
    }
