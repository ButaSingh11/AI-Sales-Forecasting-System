import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
import textwrap
from services.insight_service import detect_anomalies
from services.forecasting_service import (
    compare_forecast_models,
    get_model_label,
    get_random_forest_feature_importance as service_random_forest_feature_importance,
    prepare_monthly_series,
    rolling_backtest_forecast_models,
    run_forecast,
    run_forecast_with_fallback,
)
from services.evaluation_service import evaluate_model
from utils.app_helpers import format_inr, hex_to_rgba, load_sales_data, render_navigation_link
from utils.ui_theme import apply_theme, render_sidebar_status
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Forecasting",
    page_icon="🔮",
    layout="wide",
)

# ─────────────────────────────────────────────
# CSS — consistent with Analysis page
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0c10;
    color: #e2e8f0;
}
.stApp { background: #0a0c10; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1120 0%, #0f172a 100%);
    border-right: 1px solid #1f2937;
}
[data-testid="stSidebarNav"] {
    padding-top: 1rem;
}
[data-testid="stSidebarNav"] > div:first-child {
    display: none;
}
[data-testid="stSidebarNav"]::before {
    content: "Sales AI";
    display: block;
    padding: 0.2rem 0.9rem 0.1rem 0.9rem;
    font-size: 1.12rem;
    font-weight: 700;
    color: #f8fafc;
}
[data-testid="stSidebarNav"]::after {
    content: "Forecasting Workspace";
    display: block;
    padding: 0 0.9rem 0.75rem 0.9rem;
    margin-bottom: 0.6rem;
    border-bottom: 1px solid #1f2937;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
}
[data-testid="stSidebarNav"] ul {
    padding: 0 0.55rem 0.45rem 0.55rem;
    gap: 0.28rem;
}
[data-testid="stSidebarNavLink"] {
    border-radius: 12px;
    padding: 0.42rem 0.55rem;
    border: 1px solid transparent;
    transition: background 0.2s ease, border-color 0.2s ease;
}
[data-testid="stSidebarNavLink"]:hover {
    background: rgba(148, 163, 184, 0.08);
    border-color: #1f2937;
}
[data-testid="stSidebarNavLink"] span {
    color: #cbd5e1 !important;
    font-size: 0.92rem;
    font-weight: 600;
}
[data-testid="stSidebarNavLink"][aria-current="page"] {
    background: linear-gradient(135deg, rgba(99,102,241,0.16), rgba(59,130,246,0.14));
    border-color: rgba(99,102,241,0.35);
}
[data-testid="stSidebarNavLink"][aria-current="page"] span {
    color: #f8fafc !important;
}

.hero-header {
    background: linear-gradient(135deg, #101827 0%, #17142b 58%, #2e174f 100%);
    border: 1px solid rgba(167,139,250,0.34);
    border-radius: 16px;
    padding: 28px 48px;
    margin-bottom: 28px;
    min-height: 184px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: center;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(167,139,250,0.16) 0%, transparent 72%);
}
.hero-title {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: #f0f9ff;
    margin: 0 0 6px 0;
}
.hero-sub { font-size: 0.82rem; color: #c4b5fd; font-weight: 400; opacity: 0.72; }
.badge {
    display: inline-block;
    background: rgba(167,139,250,0.14);
    border: 1px solid rgba(167,139,250,0.36);
    color: #c4b5fd;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.72rem;
    font-weight: 500;
    margin-bottom: 12px;
}

/* KPI cards */
.kpi-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:24px; }
.kpi-card {
    background:#111827; border:1px solid #1f2937;
    border-radius:12px; padding:18px 20px;
    position:relative; overflow:hidden; transition:border-color .2s;
}
.kpi-card:hover { border-color:#a78bfa; }
.kpi-card::after { content:''; position:absolute; bottom:0; left:0; right:0; height:2px; }
.kpi-card.purple::after { background:linear-gradient(90deg,#a78bfa,#7c3aed); }
.kpi-card.blue::after   { background:linear-gradient(90deg,#38bdf8,#0ea5e9); }
.kpi-card.green::after  { background:linear-gradient(90deg,#34d399,#10b981); }
.kpi-card.amber::after  { background:linear-gradient(90deg,#fbbf24,#f59e0b); }
.kpi-label { font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:0.05em; font-weight:500; margin-bottom:7px; }
.kpi-value { font-size:1.55rem; font-weight:700; color:#f1f5f9; line-height:1; }
.kpi-sub   { font-size:0.73rem; color:#475569; margin-top:5px; }

/* Section */
.section-label {
    font-size:0.95rem; font-weight:600; color:#e2e8f0;
    margin:28px 0 6px 0; display:flex; align-items:center; gap:8px;
}

/* Model card */
.model-card {
    background:#111827; border:2px solid #1f2937;
    border-radius:14px; padding:20px;
    cursor:pointer; transition:all .2s;
    height:100%;
}
.model-card:hover   { border-color:#a78bfa; }
.model-card.active  { border-color:#a78bfa; background:rgba(167,139,250,0.05); }
.model-title  { font-size:0.9rem; font-weight:600; color:#e2e8f0; margin-bottom:6px; }
.model-badge  {
    display:inline-block; border-radius:8px; padding:2px 10px;
    font-size:0.68rem; font-weight:500; margin-bottom:10px;
}
.model-desc   { font-size:0.76rem; color:#64748b; line-height:1.55; }

/* Chart wrap */
.chart-wrap {
    background:#111827; border:1px solid #1f2937;
    border-radius:14px; padding:8px; margin-bottom:18px;
}

/* Confidence band legend */
.legend-row { display:flex; gap:20px; flex-wrap:wrap; margin-bottom:16px; }
.legend-item { display:flex; align-items:center; gap:7px; font-size:0.78rem; color:#94a3b8; }
.legend-dot  { width:12px; height:12px; border-radius:3px; flex-shrink:0; }

/* Info box */
.info-box {
    background:rgba(167,139,250,0.05);
    border:1px solid rgba(167,139,250,0.2);
    border-radius:10px; padding:14px 16px;
    font-size:0.79rem; color:#94a3b8; line-height:1.6;
    margin-bottom:14px;
}
.info-box b { color:#a78bfa; }

/* Metric table */
.metric-table { width:100%; border-collapse:collapse; font-size:0.8rem; }
.metric-table th {
    text-align:left; padding:8px 12px;
    color:#475569; font-weight:600; font-size:0.7rem;
    text-transform:uppercase; letter-spacing:0.05em;
    border-bottom:1px solid #1f2937;
}
.metric-table td { padding:10px 12px; border-bottom:1px solid #1f2937; color:#cbd5e1; }
.metric-table tr:last-child td { border-bottom:none; }
.metric-table tr:hover td { background:rgba(167,139,250,0.03); }

/* Accuracy badge */
.acc-good { color:#34d399; font-weight:600; }
.acc-ok   { color:#fbbf24; font-weight:600; }
.acc-bad  { color:#f87171; font-weight:600; }

/* Model recommendation explanation upgrade */
.reco-grid { display:grid; grid-template-columns:1.2fr 1fr; gap:14px; margin-top:14px; margin-bottom:10px; }
.reco-card { background:#111827; border:1px solid #1f2937; border-radius:14px; padding:18px; position:relative; overflow:hidden; height:100%; }
.reco-card::after { content:""; position:absolute; left:0; right:0; bottom:0; height:2px; background:linear-gradient(90deg,#818cf8,#4f46e5); }
.reco-title { font-size:0.9rem; font-weight:700; color:#e2e8f0; margin-bottom:8px; }
.reco-sub { font-size:0.78rem; color:#64748b; line-height:1.6; margin-bottom:12px; }
.reco-item { background:#0f1623; border:1px solid #1f2937; border-radius:10px; padding:12px 14px; margin-bottom:10px; }
.reco-item-label { font-size:0.68rem; color:#64748b; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px; font-weight:600; }
.reco-item-value { font-size:0.8rem; color:#cbd5e1; line-height:1.65; }
.reco-badge { display:inline-block; border-radius:999px; padding:4px 10px; font-size:0.68rem; font-weight:700; margin-right:8px; margin-bottom:10px; }
.badge-high { color:#34d399; background:rgba(52,211,153,0.10); border:1px solid rgba(52,211,153,0.25); }
.badge-moderate { color:#fbbf24; background:rgba(251,191,36,0.10); border:1px solid rgba(251,191,36,0.25); }
.badge-low { color:#f87171; background:rgba(248,113,113,0.10); border:1px solid rgba(248,113,113,0.25); }
.badge-stable { color:#34d399; background:rgba(52,211,153,0.10); border:1px solid rgba(52,211,153,0.25); }
.badge-mixed { color:#fbbf24; background:rgba(251,191,36,0.10); border:1px solid rgba(251,191,36,0.25); }
.badge-volatile { color:#f87171; background:rgba(248,113,113,0.10); border:1px solid rgba(248,113,113,0.25); }
.reco-note { background:rgba(129,140,248,0.06); border:1px solid rgba(129,140,248,0.2); border-radius:12px; padding:12px 14px; margin-top:6px; font-size:0.79rem; color:#cbd5e1; line-height:1.65; }
.reco-note b { color:#c7d2fe; }
.detail-grid { display:grid; grid-template-columns:1.15fr 1fr; gap:16px; margin:6px 0 24px 0; }
.detail-card { background:#111827; border:1px solid #1f2937; border-radius:14px; padding:18px 18px 16px 18px; }
.detail-title { font-size:0.92rem; font-weight:700; color:#f8fafc; margin-bottom:8px; }
.detail-kicker { font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:0.08em; font-weight:700; margin-bottom:8px; }
.detail-text { font-size:0.8rem; color:#cbd5e1; line-height:1.7; }
.detail-list { margin:10px 0 0 18px; padding:0; color:#cbd5e1; font-size:0.8rem; line-height:1.75; }
.detail-list li { margin-bottom:6px; }
.detail-pill { display:inline-block; border-radius:999px; padding:4px 10px; font-size:0.68rem; font-weight:700; margin:0 8px 10px 0; }
.detail-pill.good { color:#34d399; background:rgba(52,211,153,0.10); border:1px solid rgba(52,211,153,0.25); }
.detail-pill.warn { color:#fbbf24; background:rgba(251,191,36,0.10); border:1px solid rgba(251,191,36,0.25); }
.detail-pill.risk { color:#f87171; background:rgba(248,113,113,0.10); border:1px solid rgba(248,113,113,0.25); }
.decision-panel { background:#111827; border:1px solid #1f2937; border-radius:14px; padding:16px 18px; margin:4px 0 22px 0; }
.decision-grid { display:grid; grid-template-columns:1.2fr 0.9fr 0.9fr; gap:12px; align-items:stretch; }
.decision-block { background:#0f1623; border:1px solid #1f2937; border-radius:10px; padding:12px 14px; min-width:0; }
.decision-label { font-size:0.68rem; color:#64748b; text-transform:uppercase; letter-spacing:0.06em; font-weight:700; margin-bottom:6px; }
.decision-value { font-size:0.88rem; color:#f8fafc; font-weight:700; line-height:1.35; overflow-wrap:anywhere; }
.decision-copy { font-size:0.78rem; color:#94a3b8; line-height:1.6; margin-top:6px; }
.decision-pill { display:inline-block; border-radius:999px; padding:3px 9px; font-size:0.68rem; font-weight:700; margin-right:6px; margin-top:8px; }
.decision-pill.good { color:#34d399; background:rgba(52,211,153,0.10); border:1px solid rgba(52,211,153,0.25); }
.decision-pill.warn { color:#fbbf24; background:rgba(251,191,36,0.10); border:1px solid rgba(251,191,36,0.25); }
.decision-pill.risk { color:#f87171; background:rgba(248,113,113,0.10); border:1px solid rgba(248,113,113,0.25); }
@media (max-width: 900px) {
    .detail-grid { grid-template-columns:1fr; }
    .decision-grid { grid-template-columns:1fr; }
}
</style>
""", unsafe_allow_html=True)

apply_theme("#a78bfa")
render_sidebar_status()


# ─────────────────────────────────────────────
# HELPERS — MODELS
# ─────────────────────────────────────────────
def prepare_series(df: pd.DataFrame) -> pd.DataFrame:
    return prepare_monthly_series(df)


@st.cache_data(show_spinner=False)
def get_random_forest_feature_importance(series: pd.DataFrame) -> pd.DataFrame:
    return service_random_forest_feature_importance(series)


def normalize_metric_keys(metrics):
    r2_value = 0.0
    for key in ("R2", "R2 Score", "R²", "RÂ²"):
        if key in metrics:
            r2_value = metrics[key]
            break

    metrics["R2"] = r2_value
    metrics["R2 Score"] = r2_value
    return metrics


def get_in_sample_fit(series, model_key, model_kwargs=None):
    """Compute in-sample fitted values to get accuracy metrics."""
    model_kwargs = model_kwargs or {}
    metrics = evaluate_model(
        series,
        lambda train, periods, **kwargs: run_forecast(train, model_key, periods, **kwargs),
        model_kwargs,
    )
    return normalize_metric_keys(metrics) if metrics else None


def get_first_existing_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((col for col in candidates if col in df.columns), None)


def apply_damped_trend_if_flat(forecast_df: pd.DataFrame,
                               series: pd.DataFrame,
                               trend_weight: float = 0.45,
                               damping: float = 0.88) -> pd.DataFrame:
    """Keep smoothing forecasts from rendering as a repeated flat line after hot reloads."""
    if forecast_df.empty or forecast_df["Forecast"].nunique() > 1 or len(series) < 2:
        return forecast_df

    values = series["Sales"].to_numpy(dtype=float)
    recent = values[-min(6, len(values)):]
    if len(recent) < 2:
        return forecast_df

    x = np.arange(len(recent), dtype=float)
    recent_trend = float(np.polyfit(x, recent, 1)[0])
    flat_threshold = max(float(np.mean(np.abs(values))) * 0.001, 1e-9)
    if not np.isfinite(recent_trend) or abs(recent_trend) <= flat_threshold:
        return forecast_df

    adjusted = forecast_df.copy()
    steps = np.arange(1, len(adjusted) + 1, dtype=float)
    damped_steps = np.array([
        sum(damping ** step for step in range(int(horizon)))
        for horizon in steps
    ])
    adjustment = trend_weight * recent_trend * damped_steps
    adjusted["Forecast"] = np.maximum(adjusted["Forecast"].to_numpy(dtype=float) + adjustment, 0)
    band_width = np.maximum(
        forecast_df["Upper"].to_numpy(dtype=float) - forecast_df["Forecast"].to_numpy(dtype=float),
        0,
    )
    adjusted["Upper"] = adjusted["Forecast"] + band_width
    adjusted["Lower"] = np.maximum(adjusted["Forecast"] - band_width, 0)
    return adjusted


@st.cache_data(show_spinner=False)
def build_model_comparison_df(series: pd.DataFrame) -> pd.DataFrame:
    comparison_df = compare_forecast_models(series, include_random_forest=False)
    if comparison_df.empty:
        return pd.DataFrame()

    display_df = comparison_df.copy()
    display_df["Model"] = display_df["model_key"].apply(get_model_label)
    display_df["MAPE %"] = display_df["MAPE (%)"].round(1)
    display_df["MAE"] = display_df["MAE"].round()
    r2_col = get_first_existing_column(display_df, ("R2", "R2 Score", "R²", "RÂ²"))
    display_df["R2"] = display_df[r2_col].round(3) if r2_col else 0.0
    return display_df[["Model", "MAPE %", "MAE", "R2"]].sort_values("MAPE %").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def build_rolling_backtest_df(series: pd.DataFrame) -> pd.DataFrame:
    backtest_df = rolling_backtest_forecast_models(
        series,
        min_train_size=6,
        horizon=1,
        max_splits=5,
        include_random_forest=False,
    )
    if backtest_df.empty:
        return backtest_df

    display_df = backtest_df.copy()
    display_df["Model"] = display_df["model_key"].apply(get_model_label)
    r2_col = get_first_existing_column(display_df, ("R2", "R2 Score", "R²", "RÂ²"))
    display_df["R2"] = display_df[r2_col] if r2_col else 0.0
    return display_df[["Model", "MAPE (%)", "MAE", "RMSE", "R2", "Splits", "Horizon"]]


def get_volatility_info(series: pd.DataFrame):
    mean_val = float(series["Sales"].mean()) if len(series) else 0.0
    std_val  = float(series["Sales"].std()) if len(series) > 1 else 0.0
    cv = (std_val / mean_val * 100) if mean_val > 0 else 0.0

    if cv < 10:
        return {
            "label": "Stable",
            "css": "badge-stable",
            "text": f"Monthly sales variability is low (CV {cv:.1f}%), so the series is relatively stable and easier to forecast.",
            "cv": cv,
        }
    elif cv < 20:
        return {
            "label": "Moderately Stable",
            "css": "badge-mixed",
            "text": f"Monthly sales show some movement (CV {cv:.1f}%), but the pattern is still reasonably predictable.",
            "cv": cv,
        }
    return {
        "label": "Volatile",
        "css": "badge-volatile",
        "text": f"Monthly sales fluctuate strongly (CV {cv:.1f}%), which can reduce forecast reliability and widen uncertainty.",
        "cv": cv,
    }


def get_confidence_info(metrics_raw, series: pd.DataFrame, anomaly_count: int):
    if not metrics_raw:
        return {
            "label": "Unavailable",
            "css": "badge-moderate",
            "score": None,
            "reason": "Confidence could not be estimated because model metrics are not available.",
        }

    mape = float(metrics_raw.get("MAPE (%)", 0))
    mean_val = float(series["Sales"].mean()) if len(series) else 0.0
    std_val  = float(series["Sales"].std()) if len(series) > 1 else 0.0
    volatility = (std_val / mean_val * 100) if mean_val > 0 else 0.0
    data_points = len(series)

    score = 100
    if mape < 10:
        score -= 0
    elif mape < 20:
        score -= 10
    elif mape < 35:
        score -= 25
    else:
        score -= 40

    if volatility < 10:
        score -= 0
    elif volatility < 20:
        score -= 8
    elif volatility < 35:
        score -= 18
    else:
        score -= 30

    if data_points >= 24:
        score -= 0
    elif data_points >= 12:
        score -= 10
    elif data_points >= 6:
        score -= 22
    else:
        score -= 35

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
        level, css = "High", "badge-high"
    elif score >= 50:
        level, css = "Moderate", "badge-moderate"
    else:
        level, css = "Low", "badge-low"

    reason_parts = []
    if mape < 10:
        reason_parts.append("low forecast error")
    elif mape < 20:
        reason_parts.append("reasonable forecast error")
    else:
        reason_parts.append("higher forecast error")

    if volatility < 10:
        reason_parts.append("stable sales history")
    elif volatility < 20:
        reason_parts.append("moderate variation")
    else:
        reason_parts.append("volatile sales movement")

    if data_points >= 24:
        reason_parts.append("strong historical depth")
    elif data_points >= 12:
        reason_parts.append("usable historical depth")
    else:
        reason_parts.append("limited history")

    if anomaly_count == 0:
        reason_parts.append("no major anomalies")
    elif anomaly_count <= 2:
        reason_parts.append("a few anomalies")
    else:
        reason_parts.append("multiple anomalies")

    return {
        "label": level,
        "css": css,
        "score": score,
        "reason": "Confidence is based on " + ", ".join(reason_parts) + ".",
    }


def get_model_suitability(model_name: str) -> str:
    name = model_name.lower()
    if "smart ensemble" in name:
        return "Best when you want a balanced forecast because it blends the strongest recent models instead of relying on one method."
    if "moving average" in name:
        return "Best for smooth, stable sales where recent history matters most and trend changes are limited."
    if "linear trend" in name:
        return "Best when sales move in a fairly straight upward or downward direction over time without strong seasonal swings."
    if "seasonal trend" in name:
        return "Best when the same months repeat seasonal peaks and dips, but the business level has also grown or declined versus last year."
    if "seasonal naive" in name:
        return "Best when the business repeats strong seasonal patterns and last year's same period is a strong guide for the next one."
    if "exponential smoothing" in name:
        return "Best for general-purpose forecasting when recent periods should influence predictions more than older history."
    if "holt" in name:
        return "Best when the data has a clear trend and still needs smoothing, which is common in growing or declining sales series."
    return "This model is suitable because it matched the current dataset better than the alternatives based on the evaluation metrics."


def get_accuracy_factors(metrics_raw, series: pd.DataFrame, anomaly_count: int) -> str:
    if not metrics_raw:
        return "Accuracy factors could not be explained because metrics are unavailable."

    parts = []
    mape = float(metrics_raw.get("MAPE (%)", 0))
    if mape < 10:
        parts.append("forecast error is very low")
    elif mape < 20:
        parts.append("forecast error is in a good range")
    else:
        parts.append("forecast error is comparatively higher")

    vol = get_volatility_info(series)["cv"]
    if vol < 10:
        parts.append("sales are relatively stable")
    elif vol < 20:
        parts.append("sales have moderate variation")
    else:
        parts.append("sales are volatile")

    if len(series) >= 24:
        parts.append("historical depth is strong")
    elif len(series) >= 12:
        parts.append("history is reasonably sufficient")
    else:
        parts.append("limited history may reduce reliability")

    if anomaly_count == 0:
        parts.append("no major anomalies were detected")
    elif anomaly_count <= 2:
        parts.append("only a few anomalies were detected")
    else:
        parts.append("multiple anomalies may be affecting accuracy")

    return "Key accuracy drivers: " + ", ".join(parts) + "."


def summarize_forecast_path(forecast_df: pd.DataFrame, last_actual: float) -> dict:
    values = forecast_df["Forecast"].to_numpy(dtype=float)
    dates = forecast_df["Date"].dt.strftime("%b %Y").tolist()
    if len(values) == 0:
        return {
            "direction": "No forecast available",
            "summary": "No forecast values were generated.",
            "highlights": [],
        }

    first_val = float(values[0])
    last_val = float(values[-1])
    total_change_pct = ((last_val - first_val) / max(first_val, 1)) * 100
    launch_change_pct = ((first_val - last_actual) / max(last_actual, 1)) * 100

    if total_change_pct > 8:
        direction = "Upward outlook"
    elif total_change_pct < -8:
        direction = "Softening outlook"
    else:
        direction = "Mostly steady outlook"

    summary = (
        f"The forecast starts at {format_inr(first_val)} in {dates[0]} and ends at "
        f"{format_inr(last_val)} by {dates[-1]}. Compared with the last actual month, "
        f"the first forecasted month is {'up' if launch_change_pct >= 0 else 'down'} "
        f"{abs(launch_change_pct):.1f}%."
    )

    diffs = np.diff(values) if len(values) > 1 else np.array([])
    if len(diffs) > 0:
        biggest_rise_idx = int(np.argmax(diffs))
        biggest_drop_idx = int(np.argmin(diffs))
        rise_text = (
            f"Biggest increase: {dates[biggest_rise_idx + 1]} "
            f"({abs(diffs[biggest_rise_idx]) / max(values[biggest_rise_idx], 1) * 100:.1f}% vs previous month)."
        )
        drop_text = (
            f"Biggest slowdown: {dates[biggest_drop_idx + 1]} "
            f"({abs(diffs[biggest_drop_idx]) / max(values[biggest_drop_idx], 1) * 100:.1f}% vs previous month)."
        )
    else:
        rise_text = "Only one forecasted month is selected, so there is no month-to-month trend yet."
        drop_text = "Only one forecasted month is selected, so there is no month-to-month slowdown yet."

    return {
        "direction": direction,
        "summary": summary,
        "highlights": [
            rise_text,
            drop_text,
            f"Overall change across the forecast horizon: {'+' if total_change_pct >= 0 else '-'}{abs(total_change_pct):.1f}%.",
        ],
    }


# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────
CHART_BG = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font         =dict(family="Inter", color="#94a3b8", size=11),
    title_font   =dict(family="Inter", color="#e2e8f0", size=13, weight="bold"),
    legend       =dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1f2937",
                       borderwidth=1, font=dict(color="#94a3b8", size=10)),
    margin       =dict(l=16, r=16, t=44, b=16),
    xaxis        =dict(gridcolor="#1f2937", linecolor="#1f2937",
                       tickcolor="#1f2937", tickfont=dict(size=10)),
    yaxis        =dict(gridcolor="#1f2937", linecolor="#1f2937",
                       tickcolor="#1f2937", tickfont=dict(size=10)),
)


raw_df, is_sample = load_sales_data(include_segments=False, copy_uploaded=False)
raw_df["Date"] = pd.to_datetime(raw_df["Date"])
monthly = prepare_series(raw_df)

if monthly.empty:
    st.error("No monthly sales history is available yet. Upload data with Date and Sales columns before running forecasts.")
    render_navigation_link("pages/1_Data_Upload.py", "/1_Data_Upload", "Go to Data Upload")
    st.stop()


# ═════════════════════════════════════════════
# HERO
# ═════════════════════════════════════════════
st.markdown("""
<div class="hero-header">
  <div class="badge">🔮 MODULE 03 — FORECASTING</div>
  <div class="hero-title">AI Sales Forecasting Engine</div>
  <div class="hero-sub">Choose a model · Set horizon · Explore predictions with confidence intervals</div>
</div>
""", unsafe_allow_html=True)

# sample data warning
if is_sample:
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(251,191,36,0.08),rgba(245,158,11,0.04));
                border:1px solid rgba(251,191,36,0.3);border-radius:12px;
                padding:14px 20px;margin-bottom:20px;display:flex;align-items:center;gap:12px;">
      <span style="font-size:1.4rem;">⚠️</span>
      <div>
        <div style="font-size:0.88rem;font-weight:600;color:#fbbf24;">Using Sample Data</div>
        <div style="font-size:0.76rem;color:#92400e;margin-top:2px;">
          Go to <b style="color:#fbbf24;">📁 Data Upload</b> to use your real sales data.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── What is forecasting? ───────────────────
with st.expander("📖 What is Sales Forecasting? — Click to learn", expanded=False):
    st.markdown("""
    **Sales Forecasting** means using past sales data to predict future revenue.

    Think of it like weather forecasting — you can't be 100% certain, but using patterns
    from the past (seasonality, trends, momentum) you can make educated predictions.

    **Why does it matter?**
    - 📦 Plan inventory — avoid stockouts or overstocking
    - 👥 Staff planning — hire or schedule based on expected demand
    - 💰 Budget setting — realistic revenue targets
    - 📈 Investor reporting — show growth trajectory

    **What is a Confidence Interval?**
    The shaded band around the forecast line is the **confidence interval** (95% by default).
    It means: *"We are 95% sure the actual value will fall inside this range."*
    A wider band = more uncertainty. A narrow band = model is more confident.

    **What is MAPE?**
    Mean Absolute Percentage Error — measures how far off the forecast is on average, as a %.
    - Under 10% → Excellent
    - 10–20% → Good
    - 20–50% → Acceptable
    - Above 50% → Poor
    """)


# ═════════════════════════════════════════════
# STEP 1 — MODEL SELECTION
# ═════════════════════════════════════════════
st.markdown('<div class="section-label">🧠 Step 1 — Choose a Forecasting Model</div>',
            unsafe_allow_html=True)
st.markdown("""
<style>
div[data-testid="stButton"] > button {
    min-height: 38px;
    padding: 0.35rem 0.75rem;
    border-radius: 10px;
    font-size: 0.84rem;
}
</style>
""", unsafe_allow_html=True)

models_info = [
    ("Smart Ensemble",         "Best Auto",  "#22c55e", "green",
     "Blends the strongest recent models using validation error. "
     "Best when you want the app to pick a balanced forecast path automatically. "
     "✅ Recommended default for planning.",
     "smart_ensemble"),
    ("Moving Average",         "Simple",     "#34d399", "green",
     "Averages recent months and adds a small damped recent trend. "
     "Stable for noisy data, but no longer repeats one exact value across the horizon.",
     "moving_average"),
    ("Linear Trend",           "Trending",   "#38bdf8", "sky blue",
     "Fits a straight line through your historical data and extends it forward. "
     "Great if sales are steadily growing or declining. "
     "❌ Assumes growth stays constant — misses seasonal peaks.",
     "linear_trend"),
    ("Seasonal Trend",         "Seasonal+",   "#fbbf24", "amber",
     "Uses last year's same month as the seasonal shape, then adjusts it using recent year-over-year growth. "
     "Better when seasonal peaks repeat but the business has moved up or down.",
     "seasonal_trend"),
    ("Seasonal Naive",         "Seasonal",   "#eab308", "amber",
     "Seasonal baseline with a light growth adjustment. "
     "Use Seasonal Trend when yearly seasonality and business growth both matter.",
     "seasonal_naive"),
    ("Exponential Smoothing",  "Balanced",   "#a78bfa", "violet",
     "Gives more weight to recent data and carries a damped recent trend into the future. "
     "Good all-purpose model for most businesses. "
     "Smoother than Holt's, but avoids repeating the exact same value every month.",
     "exp_smoothing"),
    ("Holt's Double Smoothing","Trend+Smooth","#fb923c", "orange",
     "Like Exponential Smoothing but also tracks the trend direction separately. "
     "Best of both worlds — adapts to recent changes AND captures growth direction. "
     "✅ Recommended for most use cases.",
     "holts"),
]
model_lookup = {
    key: {
        "name": name,
        "tag": tag,
        "color": color,
        "color_name": color_name,
        "description": desc,
    }
    for name, tag, color, color_name, desc, key in models_info
}

# render model cards
col_counts = st.columns(4)
selected_model = st.session_state.get("selected_model", "smart_ensemble")

for idx, (name, tag, color, color_name, desc, key) in enumerate(models_info):
    with col_counts[idx % len(col_counts)]:
        is_active  = selected_model == key
        border     = color if is_active else "#1f2937"
        bg         = hex_to_rgba(color, 0.05) if is_active else "#111827"
        tag_bg     = hex_to_rgba(color, 0.13)
        tag_border = hex_to_rgba(color, 0.35)
        st.markdown(textwrap.dedent(f"""
        <div style="background:{bg};border:1.5px solid {border};border-radius:10px;
                    padding:12px 12px;min-height:156px;height:100%;margin-bottom:6px;
                    display:flex;flex-direction:column;">
          <div style="font-size:0.8rem;font-weight:650;color:#e2e8f0;margin-bottom:5px;line-height:1.2;">{name}</div>
          <div style="display:inline-block;background:{tag_bg};color:{color};
                      border:1px solid {tag_border};border-radius:7px;
                      padding:1px 8px;font-size:0.6rem;font-weight:600;margin-bottom:8px;line-height:1.4;">{tag}</div>
          <div style="font-size:0.68rem;color:#7890b5;line-height:1.42;flex:1;">{desc}</div>
        </div>
        """), unsafe_allow_html=True)
        if st.button(f"{'✅ Selected' if is_active else 'Select'}", key=f"btn_{key}",
                     width="stretch"):
            st.session_state["selected_model"] = key
            st.rerun()

selected_model      = st.session_state.get("selected_model", "smart_ensemble")
selected_meta       = model_lookup.get(selected_model, model_lookup["smart_ensemble"])
selected_name       = selected_meta["name"]
selected_color      = selected_meta["color"]
selected_color_name = selected_meta["color_name"]

# pre-compute rgba fill string safely
r = int(selected_color[1:3], 16)
g = int(selected_color[3:5], 16)
b = int(selected_color[5:7], 16)
selected_color_rgba = f"rgba({r},{g},{b},0.12)"
selected_color_bar  = f"rgba({r},{g},{b},0.6)"

top_reco_df = build_model_comparison_df(monthly)
if not top_reco_df.empty:
    top_best_row = top_reco_df.iloc[0]
    top_best = top_best_row["Model"]
    top_best_mape = float(top_best_row["MAPE %"])
    top_selected_match = top_reco_df[top_reco_df["Model"] == selected_name]
    top_selected_mape = (
        float(top_selected_match.iloc[0]["MAPE %"])
        if not top_selected_match.empty else None
    )
    top_status = (
        "You are already using the recommended model."
        if top_best == selected_name else
        f"Current selection is {selected_name}; recommended is {top_best}."
    )
    if top_selected_mape is not None and top_best != selected_name:
        top_status += f" Current MAPE: {top_selected_mape:.1f}%; recommended MAPE: {top_best_mape:.1f}%."

    st.markdown(f"""
    <div style="background:rgba(52,211,153,0.06);border:1px solid rgba(52,211,153,0.22);
                border-radius:10px;padding:10px 14px;margin:8px 0 14px 0;
                display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap;">
      <div style="font-size:0.78rem;color:#94a3b8;line-height:1.55;">
        <b style="color:#34d399;">Recommended model:</b>
        <span style="color:#e2e8f0;font-weight:700;">{top_best}</span>
        <span style="color:#64748b;"> · MAPE {top_best_mape:.1f}%</span><br>
        {top_status}
      </div>
      <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:0.06em;font-weight:700;">
        Based on model comparison
      </div>
    </div>
    """, unsafe_allow_html=True)
    if top_best != selected_name:
        top_best_key = next(
            (key for name, tag, color, color_name, desc, key in models_info if name == top_best),
            None,
        )
        if top_best_key and st.button(
            f"Use recommended model: {top_best}",
            key="use_recommended_model_top",
            width="stretch",
        ):
            st.session_state["selected_model"] = top_best_key
            st.rerun()


# ═════════════════════════════════════════════
# STEP 2 — PARAMETERS
# ═════════════════════════════════════════════
st.markdown('<div class="section-label">⚙️ Step 2 — Set Forecast Parameters</div>',
            unsafe_allow_html=True)

p1, p2, p3, p4 = st.columns([1.5, 1.5, 1.5, 1.5])

with p1:
    forecast_periods = st.slider(
        "Forecast Horizon (months)", min_value=1, max_value=24, value=6, step=1,
        help="How many months into the future to predict."
    )
with p2:
    conf_level = st.selectbox(
        "Confidence Interval",
        ["95% (recommended)", "80%", "99%"],
        help="The shaded band around the forecast. 95% means we're 95% sure the real value falls inside."
    )
    ci_z = {"95% (recommended)": 1.96, "80%": 1.28, "99%": 2.576}[conf_level]

with p3:
    show_history = st.selectbox(
        "Show History",
        ["Last 12 months", "Last 24 months", "All data"],
        help="How much past data to display alongside the forecast."
    )
    history_map = {"Last 12 months": 12, "Last 24 months": 24, "All data": 9999}
    history_n   = history_map[show_history]

with p4:
    # model-specific params
    if selected_model == "moving_average":
        window = st.slider("MA Window (months)", 2, 12, 3,
                           help="Number of past months to average. Larger = smoother but slower to react.")
    elif selected_model == "exp_smoothing":
        alpha = st.slider("Smoothing Factor α", 0.05, 0.95, 0.30, 0.05,
                          help="How much weight to give recent data. Higher α = reacts faster to changes.")
        trend_weight = st.slider("Trend Blend", 0.00, 1.00, 0.45, 0.05,
                                 help="How strongly the smoothed forecast follows the recent month-to-month direction.")
    elif selected_model == "holts":
        alpha_h = st.slider("Level α", 0.05, 0.95, 0.30, 0.05,
                            help="Controls how fast the level (base value) adapts.")
        beta_h  = st.slider("Trend β", 0.01, 0.50, 0.10, 0.01,
                            help="Controls how fast the trend direction adapts.")
    else:
        st.markdown("<span style='font-size:0.78rem;color:#475569;'>No extra parameters needed for this model.</span>",
                    unsafe_allow_html=True)


# ═════════════════════════════════════════════
# RUN FORECAST
# ═════════════════════════════════════════════
history_df = monthly.tail(history_n).copy()

model_kwargs = {}
if selected_model == "moving_average":
    model_kwargs = {"window": window}
elif selected_model == "exp_smoothing":
    model_kwargs = {"alpha": alpha}
elif selected_model == "holts":
    model_kwargs = {"alpha": alpha_h, "beta": beta_h}

try:
    forecast_df = run_forecast(monthly, selected_model, forecast_periods, **model_kwargs)
except Exception as exc:
    try:
        forecast_df, fallback_model = run_forecast_with_fallback(
            monthly,
            forecast_periods,
            preferred_model=selected_model,
            model_kwargs=model_kwargs,
        )
    except Exception as fallback_exc:
        st.error(f"Forecasting could not run on the current dataset: {fallback_exc}")
        render_navigation_link("pages/1_Data_Upload.py", "/1_Data_Upload", "Review uploaded data")
        st.stop()

    selected_model = fallback_model
    selected_meta = model_lookup.get(selected_model, model_lookup["smart_ensemble"])
    selected_name = selected_meta["name"]
    selected_color = selected_meta["color"]
    selected_color_name = selected_meta["color_name"]
    r = int(selected_color[1:3], 16)
    g = int(selected_color[3:5], 16)
    b = int(selected_color[5:7], 16)
    selected_color_rgba = f"rgba({r},{g},{b},0.12)"
    selected_color_bar = f"rgba({r},{g},{b},0.6)"
    model_kwargs = {}
    st.warning(
        f"{selected_meta['name']} was used because the selected model could not run on this dataset. "
        f"Original issue: {exc}"
    )

if selected_model == "exp_smoothing":
    forecast_df = apply_damped_trend_if_flat(forecast_df, monthly, trend_weight=trend_weight)

metrics_raw = get_in_sample_fit(monthly, selected_model, model_kwargs)

# apply chosen CI z-score
if ci_z != 1.96:
    diff = forecast_df["Upper"] - forecast_df["Forecast"]
    forecast_df["Upper"] = forecast_df["Forecast"] + (ci_z / 1.96) * diff
    forecast_df["Lower"] = np.maximum(forecast_df["Forecast"] - (ci_z / 1.96) * diff, 0)


# ═════════════════════════════════════════════
# FORECAST KPI CARDS
# ═════════════════════════════════════════════
total_fcst  = forecast_df["Forecast"].sum()
avg_fcst    = forecast_df["Forecast"].mean()
peak_month  = forecast_df.loc[forecast_df["Forecast"].idxmax(), "Date"].strftime("%b %Y")
last_actual = monthly["Sales"].iloc[-1]
next_pred   = forecast_df["Forecast"].iloc[0]
mom_change  = (next_pred - last_actual) / last_actual * 100
anomalies_monthly = detect_anomalies(
    monthly.copy(),
    method="rolling_deviation",
    threshold=2.5,
    rolling_window=6,
)
forecast_anomaly_count = len(anomalies_monthly)
forecast_confidence_info = get_confidence_info(metrics_raw, monthly, forecast_anomaly_count)
forecast_volatility_info = get_volatility_info(monthly)
forecast_story = summarize_forecast_path(forecast_df, last_actual)

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card purple">
    <div class="kpi-label">Forecasted Revenue ({forecast_periods}mo)</div>
    <div class="kpi-value">₹{total_fcst/1e6:.2f}M</div>
    <div class="kpi-sub">total predicted revenue</div>
  </div>
  <div class="kpi-card blue">
    <div class="kpi-label">Avg Monthly Forecast</div>
    <div class="kpi-value">₹{avg_fcst:,.0f}</div>
    <div class="kpi-sub">per month average</div>
  </div>
  <div class="kpi-card green">
    <div class="kpi-label">Peak Forecast Month</div>
    <div class="kpi-value" style="font-size:1.2rem;">{peak_month}</div>
    <div class="kpi-sub">highest predicted month</div>
  </div>
  <div class="kpi-card amber">
    <div class="kpi-label">Next Month Change</div>
    <div class="kpi-value" style="color:{'#34d399' if mom_change>=0 else '#f87171'};">
      {'▲' if mom_change>=0 else '▼'} {abs(mom_change):.1f}%
    </div>
    <div class="kpi-sub">vs last actual month</div>
  </div>
</div>
""", unsafe_allow_html=True)

next_lower = float(forecast_df["Lower"].iloc[0])
next_upper = float(forecast_df["Upper"].iloc[0])
range_width_pct = ((next_upper - next_lower) / max(next_pred, 1)) * 100
confidence_class = (
    "good" if forecast_confidence_info["label"] == "High"
    else ("warn" if forecast_confidence_info["label"] == "Moderate" else "risk")
)
volatility_class = (
    "good" if forecast_volatility_info["label"] == "Stable"
    else ("warn" if forecast_volatility_info["label"] == "Moderately Stable" else "risk")
)
decision_action = (
    "Good for planning. Use the scenario page to stress-test growth, discount, and cost assumptions."
    if forecast_confidence_info["label"] == "High" else
    "Use with caution. Review anomaly months and compare scenarios before committing targets."
    if forecast_confidence_info["label"] == "Moderate" else
    "Treat as directional. Add more history or clean anomaly periods before using it for firm targets."
)

st.markdown(f"""
<div class="decision-panel">
  <div class="decision-grid">
    <div class="decision-block">
      <div class="decision-label">Recommended next action</div>
      <div class="decision-value">{decision_action}</div>
      <div class="decision-copy">The first forecast month is expected at {format_inr(next_pred)}.</div>
    </div>
    <div class="decision-block">
      <div class="decision-label">Confidence check</div>
      <div class="decision-value">{forecast_confidence_info['label']} confidence</div>
      <span class="decision-pill {confidence_class}">Score {forecast_confidence_info['score'] if forecast_confidence_info['score'] is not None else 'N/A'}/100</span>
      <span class="decision-pill {volatility_class}">{forecast_volatility_info['label']}</span>
    </div>
    <div class="decision-block">
      <div class="decision-label">Next month range</div>
      <div class="decision-value">{format_inr(next_lower)} - {format_inr(next_upper)}</div>
      <div class="decision-copy">Range width is {range_width_pct:.1f}% of the forecast value.</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-label">🧭 Detailed Forecast Explanation</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="detail-grid">
  <div class="detail-card">
    <div class="detail-kicker">What the forecast is saying</div>
    <div class="detail-title">{forecast_story['direction']}</div>
    <div class="detail-text">
      {forecast_story['summary']}
      This forecast is being generated with <b style="color:#f8fafc;">{selected_name}</b> over a
      <b style="color:#f8fafc;">{forecast_periods}-month</b> horizon.
    </div>
    <ul class="detail-list">
      <li>{forecast_story['highlights'][0]}</li>
      <li>{forecast_story['highlights'][1]}</li>
      <li>{forecast_story['highlights'][2]}</li>
    </ul>
  </div>

  <div class="detail-card">
    <div class="detail-kicker">How to understand it</div>
    <span class="detail-pill {'good' if forecast_confidence_info['label'] == 'High' else ('warn' if forecast_confidence_info['label'] == 'Moderate' else 'risk')}">
      Confidence: {forecast_confidence_info['label']}
    </span>
    <span class="detail-pill {'good' if forecast_volatility_info['label'] == 'Stable' else ('warn' if forecast_volatility_info['label'] == 'Moderately Stable' else 'risk')}">
      Data pattern: {forecast_volatility_info['label']}
    </span>
    <div class="detail-text">
      <b style="color:#f8fafc;">Confidence band:</b> the lower and upper bounds show a likely range around each forecasted month.
      Narrower ranges usually mean the data pattern is easier to predict, while wider ranges suggest more uncertainty.
      <br><br>
      <b style="color:#f8fafc;">Why confidence looks this way:</b> {forecast_confidence_info['reason']}
      {f" Overall score: {forecast_confidence_info['score']}/100." if forecast_confidence_info['score'] is not None else ""}
    </div>
    <ul class="detail-list">
      <li>Use the line chart to understand direction over time.</li>
      <li>Use the month-by-month table to check exact forecast values.</li>
      <li>Use the confidence range before making decisions on uncertain months.</li>
    </ul>
  </div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════
# STEP 3 — MAIN FORECAST CHART
# ═════════════════════════════════════════════
st.markdown('<div class="section-label">📈 Step 3 — Forecast vs Historical Continuity</div>', unsafe_allow_html=True)

with st.expander("ℹ️ How to read this chart?", expanded=False):
    st.markdown(f"""
    - **Blue line** = actual historical sales
    - **{selected_color_name} line** = forecast continuation from the last real month
    - **Shaded band** = {conf_level} confidence interval around the forecast path
    - **Dashed vertical line** = where history ends and forecast begins
    - **Wider band** = higher uncertainty; **narrower band** = stronger confidence
    """)

fig = go.Figure()
last_actual_date = monthly["Date"].iloc[-1]

continuity_x = [last_actual_date] + forecast_df["Date"].tolist()
continuity_y = [last_actual] + forecast_df["Forecast"].tolist()
band_upper = [last_actual] + forecast_df["Upper"].tolist()
band_lower = [last_actual] + forecast_df["Lower"].tolist()

# full historical line
fig.add_trace(go.Scatter(
    x=monthly["Date"], y=monthly["Sales"],
    mode="lines", name="Actual Historical Sales",
    line=dict(color="rgba(56,189,248,0.35)", width=1.6),
    hovertemplate="<b>%{x|%b %Y}</b><br>Actual: ₹%{y:,.0f}<extra></extra>",
))

# recent history highlight
fig.add_trace(go.Scatter(
    x=history_df["Date"], y=history_df["Sales"],
    mode="lines+markers", name="Recent Actuals",
    line=dict(color="#38bdf8", width=3),
    marker=dict(size=5, color="#38bdf8"),
    hovertemplate="<b>%{x|%b %Y}</b><br>Actual: ₹%{y:,.0f}<extra></extra>",
))

# confidence band continuity (upper)
fig.add_trace(go.Scatter(
    x=continuity_x, y=band_upper,
    mode="lines", name=f"Upper {conf_level}",
    line=dict(width=0), showlegend=False,
    hoverinfo="skip",
))

# confidence band continuity (lower — fills to upper)
fig.add_trace(go.Scatter(
    x=continuity_x, y=band_lower,
    mode="lines", name=f"{conf_level} Confidence Band",
    line=dict(width=0),
    fill="tonexty",
    fillcolor=selected_color_rgba,
    hoverinfo="skip",
))

# forecast continuation line
fig.add_trace(go.Scatter(
    x=continuity_x, y=continuity_y,
    mode="lines+markers", name=f"{selected_name} Forecast",
    line=dict(color=selected_color, width=2.8, dash="dot"),
    marker=dict(
        size=[0] + [6] * len(forecast_df),
        color=[selected_color] * (len(forecast_df) + 1),
        symbol=["circle"] * (len(forecast_df) + 1),
    ),
    hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: ₹%{y:,.0f}<extra></extra>",
))

# transition marker
fig.add_trace(go.Scatter(
    x=[last_actual_date], y=[last_actual],
    mode="markers",
    name="Transition Point",
    marker=dict(size=9, color="#f8fafc", line=dict(color=selected_color, width=2)),
    hovertemplate="<b>%{x|%b %Y}</b><br>Last Actual: ₹%{y:,.0f}<extra></extra>",
))

# vertical divider — drawn as a scatter trace (add_vline has a Timestamp bug in older Plotly)
divider_x = [last_actual_date, last_actual_date]
divider_y = [0, max(history_df["Sales"].max(), forecast_df["Upper"].max()) * 1.05]
fig.add_trace(go.Scatter(
    x=divider_x, y=divider_y,
    mode="lines",
    line=dict(color="#475569", width=1.5, dash="dash"),
    name="Forecast Start",
    hoverinfo="skip",
    showlegend=True,
))

fig.update_layout(
    **CHART_BG, height=380,
    title=f"{selected_name} — Historical to Forecast Continuation",
    yaxis_title="Monthly Revenue (₹)", xaxis_title="",
    hovermode="x unified",
)

fig.add_annotation(
    x=last_actual_date,
    y=divider_y[1],
    text="Forecast starts here",
    showarrow=False,
    yshift=12,
    font=dict(size=11, color="#94a3b8"),
    bgcolor="rgba(15,22,35,0.85)",
    bordercolor="#1f2937",
    borderwidth=1,
)

st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════
# FORECAST TABLE + GROWTH CHART — side by side
# ═════════════════════════════════════════════
col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.markdown('<div class="section-label">📋 Month-by-Month Forecast Table</div>',
                unsafe_allow_html=True)
    with st.expander("ℹ️ What is this table?", expanded=False):
        st.markdown("""
        Each row is one forecast month. Columns:
        - **Forecast** — the model's best single estimate
        - **Lower / Upper** — the confidence interval range
        - **MoM Change** — Month-over-Month % change vs the previous forecasted month
        A positive MoM = predicted growth that month. Negative = predicted dip.
        """)

    table_df = forecast_df.copy()
    table_df["Month"]      = table_df["Date"].dt.strftime("%b %Y")
    table_df["Forecast ₹"] = table_df["Forecast"].apply(lambda x: f"₹{x:,.0f}")
    table_df["Lower ₹"]   = table_df["Lower"].apply(lambda x: f"₹{x:,.0f}")
    table_df["Upper ₹"]   = table_df["Upper"].apply(lambda x: f"₹{x:,.0f}")

    prev_vals = [last_actual] + list(forecast_df["Forecast"].values[:-1])
    table_df["MoM %"] = [
        f"{'▲' if (f - p) >= 0 else '▼'} {abs((f - p)/max(p,1)*100):.1f}%"
        for f, p in zip(forecast_df["Forecast"], prev_vals)
    ]

    st.dataframe(
        table_df[["Month","Forecast ₹","Lower ₹","Upper ₹","MoM %"]],
        width="stretch", hide_index=True, height=min(380, 55 + forecast_periods * 35),
    )

with col_right:
    st.markdown('<div class="section-label">📊 Monthly Forecast Bar Chart</div>',
                unsafe_allow_html=True)
    with st.expander("ℹ️ What does this chart show?", expanded=False):
        st.markdown("""
        Each bar = one forecasted month's predicted revenue.
        Error bars show the confidence interval (how wide the uncertainty is).
        Taller bars with narrow error bars = confident high-revenue months.
        Short bars with wide error bars = uncertain, lower predicted months.
        """)

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=forecast_df["Date"].dt.strftime("%b %Y"),
        y=forecast_df["Forecast"],
        error_y=dict(
            type="data",
            array=forecast_df["Upper"] - forecast_df["Forecast"],
            arrayminus=forecast_df["Forecast"] - forecast_df["Lower"],
            color=selected_color, thickness=1.5, width=4,
        ),
        marker=dict(
            color=forecast_df["Forecast"],
            colorscale=[[0, "#1e1b3a"], [0.5, selected_color_bar], [1, selected_color]],
            line=dict(width=0),
        ),
        hovertemplate="<b>%{x}</b><br>₹%{y:,.0f}<extra></extra>",
    ))
    fig_bar.update_layout(
        **CHART_BG, height=340,
        title="Forecasted Revenue by Month",
        yaxis_title="Revenue (₹)", xaxis_title="", showlegend=False,
    )
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig_bar, width="stretch", config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════
# MODEL ACCURACY + MODEL COMPARISON
# ═════════════════════════════════════════════
st.markdown('<div class="section-label">📐 Step 4 — Model Accuracy & Comparison</div>',
            unsafe_allow_html=True)

with st.expander("ℹ️ What do these accuracy metrics mean?", expanded=False):
    st.markdown("""
    | Metric | Full Name | What it means |
    |--------|-----------|---------------|
    | **MAE** | Mean Absolute Error | Average ₹ difference between predicted and actual. Lower = better. |
    | **RMSE** | Root Mean Squared Error | Like MAE but penalises large errors more. Lower = better. |
    | **MAPE** | Mean Absolute % Error | Average % error. Under 10% = excellent, 10–20% = good. |
    | **R2** | R-Squared | How much of the data variance the model explains. Closer to 1.0 = better. |

    These metrics are computed on a **test set** — the last 20% of your data that the model
    was NOT trained on. This gives a realistic picture of real-world accuracy.
    """)

# run all models for comparison
acc_col1, acc_col2 = st.columns([1, 1.4])
recommendation_html = None

with acc_col1:
    st.markdown("**Selected Model Accuracy**")
    if metrics_raw:
        mape_val = metrics_raw["MAPE (%)"]
        r2_val = metrics_raw.get("R2", metrics_raw.get("R2 Score", 0.0))
        acc_class = "acc-good" if mape_val < 10 else ("acc-ok" if mape_val < 20 else "acc-bad")
        acc_label = "Excellent" if mape_val < 10 else ("Good" if mape_val < 20 else "Needs improvement")
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1f2937;border-radius:12px;
                    padding:20px;margin-bottom:12px;">
          <div style="font-size:0.78rem;color:#475569;margin-bottom:14px;">
            Model: <b style="color:#e2e8f0;">{selected_name}</b> &nbsp;|&nbsp;
            Accuracy: <span class="{acc_class}">{acc_label}</span>
          </div>
          <table class="metric-table">
            <tr><th>Metric</th><th>Value</th><th>Rating</th></tr>
            <tr>
              <td>MAE</td>
              <td>₹{metrics_raw['MAE']:,.0f}</td>
              <td><span style="color:#64748b;font-size:0.72rem;">avg ₹ error</span></td>
            </tr>
            <tr>
              <td>RMSE</td>
              <td>₹{metrics_raw['RMSE']:,.0f}</td>
              <td><span style="color:#64748b;font-size:0.72rem;">error magnitude</span></td>
            </tr>
            <tr>
              <td>MAPE</td>
              <td><span class="{acc_class}">{metrics_raw['MAPE (%)']:.1f}%</span></td>
              <td><span class="{acc_class}">{acc_label}</span></td>
            </tr>
            <tr>
              <td>R2</td>
              <td>{r2_val:.3f}</td>
              <td><span style="color:#64748b;font-size:0.72rem;">{'Good fit' if r2_val > 0.7 else 'Weak fit'}</span></td>
            </tr>
          </table>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Not enough data to compute accuracy metrics (need at least 10 months).")

with acc_col2:
    st.markdown("**All Models Comparison**")
    comp_df = top_reco_df.copy()

    if not comp_df.empty:
        fig_comp = go.Figure(go.Bar(
            x=comp_df["Model"], y=comp_df["MAPE %"],
            marker=dict(
                color=comp_df["MAPE %"],
                colorscale=[[0,"#34d399"],[0.3,"#fbbf24"],[1,"#f87171"]],
                line=dict(width=0),
            ),
            text=comp_df["MAPE %"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
            textfont=dict(color="#94a3b8", size=10),
            hovertemplate="<b>%{x}</b><br>MAPE: %{y:.1f}%<extra></extra>",
        ))
        fig_comp.update_layout(
            **CHART_BG, height=280,
            title="MAPE Comparison — Lower is Better",
            yaxis_title="MAPE (%)", xaxis_title="", showlegend=False,
        )
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig_comp, width="stretch", config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

        st.download_button(
            label="Download Model Comparison CSV",
            data=comp_df.to_csv(index=False).encode("utf-8"),
            file_name="model_comparison.csv",
            mime="text/csv",
            width="stretch",
        )

        # best model tip
        best_row = comp_df.iloc[0]
        best = best_row["Model"]
        best_mape = float(best_row["MAPE %"])
        selected_match = comp_df[comp_df["Model"] == selected_name]
        selected_mape = (
            float(selected_match.iloc[0]["MAPE %"])
            if not selected_match.empty else None
        )
        st.markdown(f"""
        <div style="background:rgba(52,211,153,0.06);border:1px solid rgba(52,211,153,0.2);
                    border-radius:10px;padding:14px 18px;font-size:0.8rem;color:#94a3b8;
                    margin:4px 0 26px 0;line-height:1.65;">
          💡 <b style="color:#34d399;">Best model for your data:</b> {best}
          with a MAPE of {best_mape}% — {"excellent accuracy" if best_mape < 10 else "good accuracy"}.
          {'You are already using the best model! ✅' if best == selected_name else
           f'Consider switching to <b style="color:#e2e8f0;">{best}</b> for better accuracy.'}
        </div>
        """, unsafe_allow_html=True)

        # model recommendation explanation upgrade
        anomalies_monthly = detect_anomalies(
            monthly.copy(),
            method="rolling_deviation",
            threshold=2.5,
            rolling_window=6,
        )
        anomaly_count = len(anomalies_monthly)
        volatility_info = get_volatility_info(monthly)
        best_metrics_raw = {"MAPE (%)": best_mape}
        confidence_info = get_confidence_info(best_metrics_raw, monthly, anomaly_count)
        confidence_reason_short = (
            confidence_info["reason"]
            .replace("Confidence is based on ", "")
            .rstrip(".")
            + "."
        )
        suitable_text = get_model_suitability(best)
        factors_text = get_accuracy_factors(best_metrics_raw, monthly, anomaly_count)
        selected_note = (
            "You are currently using the recommended model, so the forecast is aligned with the strongest measured accuracy."
            if best == selected_name else
            f"You are currently using {selected_name}, but the comparison suggests {best} may produce more reliable results on this dataset."
        )
        if selected_mape is not None and best != selected_name:
            selected_note += f" Current selection MAPE: {selected_mape:.2f}%; recommended MAPE: {best_mape:.2f}%."

        recommendation_html = textwrap.dedent(f"""
        <style>
            .reco-grid {{ display:grid; grid-template-columns:minmax(0,1.15fr) minmax(0,1fr); gap:18px; width:100%; align-items:stretch; box-sizing:border-box; margin:8px 0 26px 0; }}
            .reco-card {{ background:#111827; border:1px solid #1f2937; border-radius:14px; padding:18px; position:relative; overflow:hidden; min-width:0; box-sizing:border-box; }}
            .reco-card::after {{ content:""; position:absolute; left:0; right:0; bottom:0; height:2px; background:linear-gradient(90deg,#818cf8,#4f46e5); }}
            .reco-title {{ font-size:0.9rem; font-weight:700; color:#e2e8f0; margin-bottom:8px; line-height:1.35; }}
            .reco-sub {{ font-size:0.78rem; color:#94a3b8; line-height:1.6; margin-bottom:12px; }}
            .reco-item {{ background:#0f1623; border:1px solid #1f2937; border-radius:10px; padding:12px 14px; margin-bottom:10px; box-sizing:border-box; }}
            .reco-item-label {{ font-size:0.68rem; color:#64748b; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px; font-weight:600; }}
            .reco-item-value {{ font-size:0.8rem; color:#cbd5e1; line-height:1.65; overflow-wrap:anywhere; }}
            .reco-badge {{ display:inline-block; border-radius:999px; padding:4px 10px; font-size:0.68rem; font-weight:700; margin-right:8px; margin-bottom:10px; }}
            .confidence-panel {{ background:rgba(52,211,153,0.06); border:1px solid rgba(52,211,153,0.18); border-radius:12px; padding:14px 16px; margin:4px 0 14px 0; }}
            .confidence-label {{ color:#f8fafc; font-size:1.05rem; font-weight:700; line-height:1.25; margin-bottom:6px; }}
            .confidence-reason {{ color:#94a3b8; font-size:0.8rem; line-height:1.6; }}
            .badge-high,.badge-stable {{ color:#34d399; background:rgba(52,211,153,0.10); border:1px solid rgba(52,211,153,0.25); }}
            .badge-moderate,.badge-mixed {{ color:#fbbf24; background:rgba(251,191,36,0.10); border:1px solid rgba(251,191,36,0.25); }}
            .badge-low,.badge-volatile {{ color:#f87171; background:rgba(248,113,113,0.10); border:1px solid rgba(248,113,113,0.25); }}
            .reco-note {{ background:rgba(129,140,248,0.06); border:1px solid rgba(129,140,248,0.2); border-radius:12px; padding:12px 14px; margin-top:6px; font-size:0.79rem; color:#cbd5e1; line-height:1.65; overflow-wrap:anywhere; }}
            .reco-note b {{ color:#c7d2fe; }}
            @media (max-width: 900px) {{ .reco-grid {{ grid-template-columns:1fr; }} }}
        </style>
        <div class="reco-grid">
            <div class="reco-card">
                <div class="reco-title">🧠 Model Recommendation Explanation</div>
                <div class="reco-sub">
                    This section explains why the recommended model looks strongest for the current dataset
                    and what influences its forecast quality.
                </div>

                <div class="reco-item">
                    <div class="reco-item-label">Why this model was selected</div>
                    <div class="reco-item-value">
                        {best} was selected because it produced the lowest MAPE in the comparison table
                        ({best_mape:.2f}%), which means it had the lowest average percentage error on the
                        test portion of your data.
                    </div>
                </div>

                <div class="reco-item">
                    <div class="reco-item-label">When this model is suitable</div>
                    <div class="reco-item-value">{suitable_text}</div>
                </div>

                <div class="reco-item">
                    <div class="reco-item-label">What affected accuracy</div>
                    <div class="reco-item-value">{factors_text}</div>
                </div>

                <div class="reco-note">
                    <b>Selection status:</b> {selected_note}
                </div>
            </div>

            <div class="reco-card">
                <div class="reco-title">📊 Forecast Reliability Context</div>
                <div class="reco-sub">
                    These indicators help explain whether your data pattern is stable enough
                    for more confident forecasting.
                </div>

                <div class="confidence-panel">
                    <div class="confidence-label">{confidence_info['label']} confidence</div>
                    <div class="confidence-reason">{confidence_reason_short}</div>
                </div>

                <span class="reco-badge {confidence_info['css']}">Confidence: {confidence_info['label']}</span>
                <span class="reco-badge {volatility_info['css']}">Data Pattern: {volatility_info['label']}</span>

                <div class="reco-item">
                    <div class="reco-item-label">Confidence explanation</div>
                    <div class="reco-item-value">
                        {confidence_info['reason']}
                        {f' Overall score: {confidence_info["score"]}/100.' if confidence_info['score'] is not None else ''}
                    </div>
                </div>

                <div class="reco-item">
                    <div class="reco-item-label">Stability vs volatility</div>
                    <div class="reco-item-value">{volatility_info['text']}</div>
                </div>

                <div class="reco-item">
                    <div class="reco-item-label">Anomaly impact</div>
                    <div class="reco-item-value">
                        {"No anomalies were detected in the monthly series, so model stability is less likely to be disturbed by unusual spikes or drops." if anomaly_count == 0 else f"{anomaly_count} anomaly point(s) were detected in the monthly series. These unusual movements can reduce model consistency and make forecast intervals less certain."}
                    </div>
                </div>
            </div>
        </div>
        """)
    else:
        st.info("Model comparison is available after at least six monthly data points.")


# ═════════════════════════════════════════════
with st.expander("Advanced diagnostics: backtest and Random Forest feature importance", expanded=False):
    st.markdown("""
    <div style="color:#94a3b8;font-size:0.8rem;line-height:1.6;margin:0 0 12px 0;">
      These checks are useful for model review, but they train extra models. Run them only when you need
      deeper validation so the page can load faster during normal use.
    </div>
    """, unsafe_allow_html=True)

    if st.button("Run advanced diagnostics", width="stretch"):
        backtest_df = build_rolling_backtest_df(monthly)
        if not backtest_df.empty:
            st.markdown('<div class="section-label">Rolling Backtest Reliability</div>', unsafe_allow_html=True)
            st.dataframe(
                backtest_df,
                width="stretch",
                hide_index=True,
            )
            st.download_button(
                label="Download Rolling Backtest CSV",
                data=backtest_df.to_csv(index=False).encode("utf-8"),
                file_name="rolling_backtest.csv",
                mime="text/csv",
                width="stretch",
            )

        rf_importance_df = get_random_forest_feature_importance(monthly)
        if not rf_importance_df.empty:
            st.markdown('<div class="section-label">Random Forest Feature Importance</div>', unsafe_allow_html=True)
            rf_col1, rf_col2 = st.columns([1.35, 1])

            with rf_col1:
                fig_rf = px.bar(
                    rf_importance_df.sort_values("Importance", ascending=True),
                    x="Importance",
                    y="Feature",
                    color="Feature Type",
                    orientation="h",
                    color_discrete_map={
                        "Lag feature": "#38bdf8",
                        "Calendar feature": "#a78bfa",
                    },
                    text="Importance",
                )
                fig_rf.update_traces(
                    texttemplate="%{text:.1f}%",
                    textposition="outside",
                    cliponaxis=False,
                )
                fig_rf.update_layout(
                    **CHART_BG,
                    height=340,
                    title="Random Forest Feature Influence",
                    xaxis_title="Importance (%)",
                    yaxis_title="",
                )
                fig_rf.update_layout(
                    margin=dict(l=16, r=24, t=86, b=20),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.08,
                        xanchor="right",
                        x=1,
                        bgcolor="rgba(0,0,0,0)",
                        borderwidth=0,
                        font=dict(color="#94a3b8", size=10),
                        title=dict(text="Feature Type", font=dict(color="#e2e8f0", size=11)),
                    ),
                )
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                st.plotly_chart(fig_rf, width="stretch", config={"displayModeBar": False})
                st.markdown('</div>', unsafe_allow_html=True)

            with rf_col2:
                top_features = rf_importance_df.head(3).copy()
                top_feature_html = "".join(
                    f"<li><b>{row['Feature']}</b> - {row['Importance']:.1f}% ({row['Feature Type']})</li>"
                    for _, row in top_features.iterrows()
                )
                st.markdown(f"""
                <div style="background:#111827;border:1px solid #1f2937;border-radius:14px;
                            padding:18px 20px;margin-bottom:18px;">
                  <div style="font-size:0.92rem;font-weight:700;color:#e2e8f0;margin-bottom:10px;">
                    Top 3 influential features
                  </div>
                  <ol style="margin:0 0 14px 18px;padding:0;color:#cbd5e1;font-size:0.82rem;line-height:1.75;">
                    {top_feature_html}
                  </ol>
                  <div style="border-top:1px solid #1f2937;padding-top:12px;color:#94a3b8;
                              font-size:0.8rem;line-height:1.65;">
                    <b style="color:#c7d2fe;">Lag features</b> use recent sales history, such as last month
                    or the 3-month average, to explain momentum. <b style="color:#c7d2fe;">Calendar features</b>
                    use month, quarter, and year so the model can learn seasonality and long-term timing patterns.
                  </div>
                </div>
                """, unsafe_allow_html=True)


# Render the recommendation explanation after the accuracy columns so it can use the full page width.
if recommendation_html:
    st.html(recommendation_html)


# DOWNLOAD FORECAST
# ═════════════════════════════════════════════
st.markdown('<div class="section-label">⬇️ Export Forecast</div>', unsafe_allow_html=True)

export_df = forecast_df.copy()
export_df["Month"]    = export_df["Date"].dt.strftime("%b %Y")
export_df["Model"]    = selected_name
export_df["Forecast"] = export_df["Forecast"].round(2)
export_df["Lower"]    = export_df["Lower"].round(2)
export_df["Upper"]    = export_df["Upper"].round(2)
export_df = export_df[["Month","Forecast","Lower","Upper","Model"]]

csv_data = export_df.to_csv(index=False).encode("utf-8")

dl1, dl2, _ = st.columns([1.2, 1.2, 3])
with dl1:
    st.download_button(
        label="⬇️ Download Forecast CSV",
        data=csv_data,
        file_name=f"sales_forecast_{selected_model}_{forecast_periods}mo.csv",
        mime="text/csv",
        width="stretch",
    )
with dl2:
    render_navigation_link(
        "pages/4_Scenario_Simulation.py",
        "/4_Scenario_Simulation",
        label="📊 Go to Scenario Simulation →",
        help_text="Use the Scenario Simulation page to test what-if situations on top of this forecast.",
    )


# ═════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;margin-top:40px;padding:16px;border-top:1px solid #1f2937;
            color:#334155;font-size:0.72rem;font-family:'Inter',sans-serif;">
  AI Sales Forecasting System · Forecasting Module · 5 models · Confidence intervals included
</div>
""", unsafe_allow_html=True)
