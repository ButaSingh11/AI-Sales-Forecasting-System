import streamlit as st
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.insight_service import detect_anomalies
from app.services.forecasting_service import (
    prepare_monthly_series,
    moving_average,
    linear_trend,
    seasonal_trend,
    exp_smoothing,
    holts_double,
    smart_ensemble_forecast,
)
from app.services.evaluation_service import compare_all_models, get_best_model, forecast_confidence_score
from app.utils.app_helpers import format_inr, load_sales_data
from app.utils.ui_theme import apply_theme, render_sidebar_status

st.set_page_config(
    page_title="AI Sales Forecasting System",
    page_icon="📊",
    layout="wide"
)

def safe_pct(new, old):
    return ((new - old) / old * 100) if old not in [0, None] else 0.0


@st.cache_data(show_spinner=False)
def get_homepage_model_comparison(monthly_df: pd.DataFrame) -> pd.DataFrame:
    # Keep the landing page fast; the Forecasting page runs the heavier ML diagnostics.
    model_registry = {
        "smart_ensemble": (smart_ensemble_forecast, {}),
        "moving_average": (moving_average, {"window": 3}),
        "linear_trend": (linear_trend, {}),
        "seasonal_trend": (seasonal_trend, {"season": 12}),
        "exp_smoothing": (exp_smoothing, {"alpha": 0.3}),
        "holts": (holts_double, {"alpha": 0.3, "beta": 0.1}),
    }
    return compare_all_models(monthly_df, model_registry)


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

# ─────────────────────────────────────────────
# CSS
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

.hero-box {
    background: linear-gradient(135deg, #0f1623 0%, #151b35 58%, #1e1b4b 100%);
    border: 1px solid rgba(129,140,248,0.34);
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
.hero-box::before {
    content: "";
    position: absolute;
    top: -55px;
    right: -55px;
    width: 180px;
    height: 180px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(129,140,248,0.16) 0%, transparent 72%);
}
.hero-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 6px;
    letter-spacing: -0.01em;
}
.hero-sub {
    font-size: 0.82rem;
    color: #c7d2fe;
    line-height: 1.6;
    opacity: 0.76;
}

.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 12px 0 12px 0;
}

.summary-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 10px 0 22px 0;
}
.summary-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 18px 18px 16px 18px;
    position: relative;
    min-height: 118px;
}
.summary-card::after {
    content: "";
    position: absolute;
    left: 0; right: 0; bottom: 0;
    height: 2px;
}
.summary-card.blue::after   { background: linear-gradient(90deg,#38bdf8,#0ea5e9); }
.summary-card.green::after  { background: linear-gradient(90deg,#34d399,#10b981); }
.summary-card.amber::after  { background: linear-gradient(90deg,#fbbf24,#f59e0b); }
.summary-card.rose::after   { background: linear-gradient(90deg,#f87171,#ef4444); }
.summary-card.purple::after { background: linear-gradient(90deg,#a78bfa,#7c3aed); }
.summary-card.indigo::after { background: linear-gradient(90deg,#818cf8,#4f46e5); }

.summary-label {
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 10px;
    font-weight: 600;
}
.summary-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1.2;
}
.summary-sub {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 8px;
    line-height: 1.45;
}

.callout {
    background: rgba(129,140,248,0.08);
    border: 1px solid rgba(129,140,248,0.24);
    border-radius: 14px;
    padding: 16px 18px;
    margin: 8px 0 24px 0;
}
.callout-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: #c7d2fe;
    margin-bottom: 6px;
}
.callout-text {
    font-size: 0.82rem;
    color: #cbd5e1;
    line-height: 1.6;
}

.workflow-grid {
    display:grid;
    grid-template-columns:repeat(3, minmax(0, 1fr));
    gap:14px;
    margin:12px 0 8px 0;
}
.workflow-step {
    background:#111827;
    border:1px solid #1f2937;
    border-radius:14px;
    padding:16px 18px;
    min-height:116px;
}
.workflow-step-label {
    font-size:0.7rem;
    color:#64748b;
    text-transform:uppercase;
    letter-spacing:0.08em;
    font-weight:600;
    margin-bottom:8px;
}
.workflow-step-title {
    font-size:0.92rem;
    color:#f8fafc;
    font-weight:700;
    margin-bottom:6px;
}
.workflow-step-desc {
    font-size:0.78rem;
    color:#94a3b8;
    line-height:1.6;
}

.info-chip {
    display: inline-block;
    background: rgba(52,211,153,0.10);
    border: 1px solid rgba(52,211,153,0.25);
    color: #34d399;
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-bottom: 10px;
}
.warn-chip {
    display: inline-block;
    background: rgba(251,191,36,0.10);
    border: 1px solid rgba(251,191,36,0.25);
    color: #fbbf24;
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-bottom: 10px;
}

@media (max-width: 900px) {
    .summary-grid { grid-template-columns: repeat(2, 1fr); }
    .workflow-grid { grid-template-columns: 1fr; }
}
</style>
""", unsafe_allow_html=True)

apply_theme("#818cf8")
render_sidebar_status()

# ─────────────────────────────────────────────
# LOAD + PREP DATA
# ─────────────────────────────────────────────
df, is_sample = load_sales_data()
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

monthly = prepare_monthly_series(df)

# Core summary stats
total_sales = float(df["Sales"].sum())
best_idx = monthly["Sales"].idxmax()
worst_idx = monthly["Sales"].idxmin()

best_month = monthly.loc[best_idx, "Date"].strftime("%b %Y")
best_month_value = float(monthly.loc[best_idx, "Sales"])

weakest_month = monthly.loc[worst_idx, "Date"].strftime("%b %Y")
weakest_month_value = float(monthly.loc[worst_idx, "Sales"])

if len(monthly) >= 6:
    first_half_avg = float(monthly["Sales"].iloc[:len(monthly)//2].mean())
    second_half_avg = float(monthly["Sales"].iloc[len(monthly)//2:].mean())
    growth_pct = safe_pct(second_half_avg, first_half_avg)
else:
    first_3 = float(monthly["Sales"].iloc[:min(3, len(monthly))].mean())
    last_3 = float(monthly["Sales"].iloc[-min(3, len(monthly)):].mean())
    growth_pct = safe_pct(last_3, first_3)

growth_label = (
    "Strong Upward Trend" if growth_pct > 10 else
    "Declining Trend" if growth_pct < -10 else
    "Stable Trend"
)

# Anomalies
anomalies = detect_anomalies(
    monthly.rename(columns={"Date": "Date"}),
    method="rolling_deviation",
    threshold=2.5,
    rolling_window=6,
)
anomaly_count = len(anomalies)

best_model_name = "Not enough data"
best_model_mape = None
confidence_level = "N/A"
confidence_score = None
confidence_reason = "Need more data to estimate confidence."
comparison_df = pd.DataFrame()

try:
    comparison_df = get_homepage_model_comparison(monthly)
    best_model = get_best_model(comparison_df)

    if best_model:
        best_model_name = MODEL_LABELS.get(best_model["model_key"], best_model["model_key"])
        best_model_mape = best_model["mape"]

        volatility = float((monthly["Sales"].std() / monthly["Sales"].mean()) * 100) if monthly["Sales"].mean() else 0.0
        confidence = forecast_confidence_score(
            mape=best_model_mape,
            volatility=volatility,
            data_points=len(monthly),
            anomaly_count=anomaly_count,
        )
        confidence_level = confidence["level"]
        confidence_score = confidence["score"]
        confidence_reason = confidence["reason"]
except Exception:
    pass

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-box">
  <div class="hero-title">📊 AI-Powered Sales Forecasting System</div>
  <div class="hero-sub">
    Smarter forecasting. Real insights. Explainable AI.<br>
    Analyze your sales data, detect anomalies, and generate intelligent forecasts using a hybrid AI + machine learning pipeline.
  </div>
</div>
""", unsafe_allow_html=True)

if is_sample:
    st.markdown('<div class="warn-chip">Using sample data</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="info-chip">Using uploaded dataset</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# EXECUTIVE SUMMARY
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">🧾 Executive Summary</div>', unsafe_allow_html=True)

summary_html = f"""
<div class="summary-grid">
  <div class="summary-card blue">
    <div class="summary-label">Total Sales</div>
    <div class="summary-value">{format_inr(total_sales)}</div>
    <div class="summary-sub">Across {len(df):,} records in the active dataset</div>
  </div>

  <div class="summary-card green">
    <div class="summary-label">Growth Trend</div>
    <div class="summary-value">{growth_pct:+.1f}%</div>
    <div class="summary-sub">{growth_label}</div>
  </div>

  <div class="summary-card amber">
    <div class="summary-label">Best Month</div>
    <div class="summary-value">{best_month}</div>
    <div class="summary-sub">{format_inr(best_month_value)}</div>
  </div>

  <div class="summary-card rose">
    <div class="summary-label">Weakest Month</div>
    <div class="summary-value">{weakest_month}</div>
    <div class="summary-sub">{format_inr(weakest_month_value)}</div>
  </div>

  <div class="summary-card purple">
    <div class="summary-label">Anomaly Count</div>
    <div class="summary-value">{anomaly_count}</div>
    <div class="summary-sub">Detected using rolling deviation on monthly sales</div>
  </div>

  <div class="summary-card indigo">
    <div class="summary-label">Best Forecast Model</div>
    <div class="summary-value">{best_model_name}</div>
    <div class="summary-sub">{f"MAPE: {best_model_mape:.2f}%" if best_model_mape is not None else "Model comparison unavailable"}</div>
  </div>

  <div class="summary-card green">
    <div class="summary-label">Forecast Confidence</div>
    <div class="summary-value">{confidence_level}</div>
    <div class="summary-sub">{f"Confidence score: {confidence_score}/100" if confidence_score is not None else "Confidence not available"}</div>
  </div>

  <div class="summary-card blue">
    <div class="summary-label">Confidence Reason</div>
    <div class="summary-value" style="font-size:1rem;">{confidence_reason}</div>
    <div class="summary-sub">Based on error, volatility, history, and anomalies</div>
  </div>
</div>
"""
st.markdown(summary_html, unsafe_allow_html=True)

st.markdown(f"""
<div class="callout">
  <div class="callout-title">💡 Executive Insight</div>
  <div class="callout-text">
    The system sees an overall <b>{growth_label.lower()}</b> with total sales of
    <b>{format_inr(total_sales)}</b>. The strongest month was <b>{best_month}</b>,
    while the weakest was <b>{weakest_month}</b>. It detected <b>{anomaly_count}</b>
    anomaly point(s), and the current best forecasting approach is
    <b>{best_model_name}</b>{f" with MAPE {best_model_mape:.2f}%" if best_model_mape is not None else ""}.
  </div>
</div>
""", unsafe_allow_html=True)

with st.expander("Project Transparency", expanded=False):
    st.markdown("""
    - Data is cleaned automatically after upload so date fields, numeric sales values, duplicates, and missing values are standardized before analysis.
    - Model selection is metric-based, using the comparison table to choose the strongest-performing forecast model rather than a manual guess.
    - Forecast confidence depends on measured error, sales volatility, anomaly presence, and available history depth.
    - Chatbot answers are grounded in the uploaded dataset whenever a direct data-backed answer is possible.
    """)

st.divider()

# ─────────────────────────────────────────────
# CORE FEATURES
# ─────────────────────────────────────────────
st.subheader("🚀 What this system actually does")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
### 🧠 AI Data Understanding
- Automatic data cleaning & preprocessing
- Smart date parsing & currency handling
- Schema detection & validation

### 📈 Forecasting Engine
- Statistical models (Holt’s, Moving Avg, etc.)
- ML models (Random Forest forecasting)
- Feature engineering (lags, rolling stats, seasonality)

### 🏆 Best Model Recommendation
- Automatically evaluates multiple models
- Selects best model using **lowest MAPE**
- Confidence-based reliability scoring
""")

with col2:
    st.markdown("""
### 🚨 AI Anomaly Detection
- Detect unusual spikes & drops
- IQR + rolling deviation methods
- Severity-based anomaly tagging

### 🤖 Grounded AI Chatbot
- Answers directly from your dataset
- Supports:
  - trends
  - comparisons
  - anomalies
  - forecasting queries
- Uses hybrid pipeline (data + AI explanation)

### 🔍 Explainable Forecasts
- Why a model was selected
- Confidence level for predictions
- Transparent decision logic
""")

st.divider()

# ─────────────────────────────────────────────
# WORKFLOW
# ─────────────────────────────────────────────
st.subheader("⚙️ How it works")

st.markdown("""
<div class="workflow-grid">
  <div class="workflow-step">
    <div class="workflow-step-label">Step 1</div>
    <div class="workflow-step-title">Upload Data</div>
    <div class="workflow-step-desc">Bring in CSV or Excel sales data and let the app detect the working schema.</div>
  </div>
  <div class="workflow-step">
    <div class="workflow-step-label">Step 2</div>
    <div class="workflow-step-title">Clean and Validate</div>
    <div class="workflow-step-desc">The preprocessing layer standardizes dates, fixes numeric fields, removes duplicates, and checks data quality.</div>
  </div>
  <div class="workflow-step">
    <div class="workflow-step-label">Step 3</div>
    <div class="workflow-step-title">Analyze Patterns</div>
    <div class="workflow-step-desc">Trend, category, region, anomaly, and correlation views expose the structure of the dataset.</div>
  </div>
  <div class="workflow-step">
    <div class="workflow-step-label">Step 4</div>
    <div class="workflow-step-title">Compare Models</div>
    <div class="workflow-step-desc">Forecast models are evaluated side by side so the best option is selected using metrics, not assumption.</div>
  </div>
  <div class="workflow-step">
    <div class="workflow-step-label">Step 5</div>
    <div class="workflow-step-title">Generate Forecasts</div>
    <div class="workflow-step-desc">Historical continuity, confidence bands, and scenario simulation turn the prediction into a usable planning view.</div>
  </div>
  <div class="workflow-step">
    <div class="workflow-step-label">Step 6</div>
    <div class="workflow-step-title">Ask Grounded Questions</div>
    <div class="workflow-step-desc">The chatbot answers from the uploaded data first, then uses AI explanation only when it adds value.</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────
# KEY HIGHLIGHTS
# ─────────────────────────────────────────────
st.subheader("🔥 Why this is different")

st.markdown("""
- Not just charts → **real analysis engine**
- Not just ML → **hybrid AI + data reasoning**
- Not just predictions → **explainable forecasts**
- Not just chatbot → **grounded dataset answers**
""")

st.divider()

# ─────────────────────────────────────────────
# CTA
# ─────────────────────────────────────────────
st.success("👉 Start by uploading your dataset from the sidebar to unlock AI analysis.")

st.caption("Built for intelligent sales analytics and forecasting.")
