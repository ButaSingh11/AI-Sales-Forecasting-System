import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from app.services.forecasting_service import prepare_monthly_series, run_forecast, run_forecast_with_fallback
from app.services.simulation_service import apply_levers as service_apply_levers
from app.utils.app_helpers import format_inr, hex_to_rgba, load_sales_data, render_navigation_link
from app.utils.ui_theme import apply_theme, render_sidebar_status
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Scenario Simulation",
    page_icon="🎮",
    layout="wide",
)

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

.hero-header {
    background: linear-gradient(135deg, #0f1623 0%, #102420 58%, #064e3b 100%);
    border: 1px solid rgba(45,212,191,0.32);
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
    background: radial-gradient(circle, rgba(45,212,191,0.15) 0%, transparent 72%);
}
.hero-title { font-size:1.6rem; font-weight:700; letter-spacing:-0.01em; color:#f0f9ff; margin:0 0 6px 0; }
.hero-sub   { font-size:0.82rem; color:#99f6e4; font-weight:400; opacity:0.72; }
.badge {
    display:inline-block;
    background:rgba(45,212,191,0.13);
    border:1px solid rgba(45,212,191,0.34);
    color:#5eead4; border-radius:20px;
    padding:3px 12px; font-size:0.72rem;
    font-weight:500; margin-bottom:12px;
}

/* KPI grid */
.kpi-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:24px; }
.kpi-card {
    background:#111827; border:1px solid #1f2937;
    border-radius:12px; padding:18px 20px;
    position:relative; overflow:hidden; transition:border-color .2s;
}
.kpi-card::after { content:''; position:absolute; bottom:0; left:0; right:0; height:2px; }
.kpi-card.green::after  { background:linear-gradient(90deg,#34d399,#10b981); }
.kpi-card.red::after    { background:linear-gradient(90deg,#f87171,#ef4444); }
.kpi-card.blue::after   { background:linear-gradient(90deg,#38bdf8,#0ea5e9); }
.kpi-card.amber::after  { background:linear-gradient(90deg,#fbbf24,#f59e0b); }
.kpi-label { font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:0.05em; font-weight:500; margin-bottom:7px; }
.kpi-value { font-size:1.5rem; font-weight:700; color:#f1f5f9; line-height:1; }
.kpi-sub   { font-size:0.73rem; color:#475569; margin-top:5px; }

/* Scenario cards */
.scenario-card {
    background:#111827; border:2px solid #1f2937;
    border-radius:14px; padding:18px;
    margin-bottom:10px; transition:border-color .2s;
}
.scenario-card.best   { border-color:rgba(52,211,153,0.4); background:rgba(52,211,153,0.03); }
.scenario-card.base   { border-color:rgba(56,189,248,0.4); background:rgba(56,189,248,0.03); }
.scenario-card.worst  { border-color:rgba(248,113,113,0.4); background:rgba(248,113,113,0.03); }

/* Section label */
.section-label {
    font-size:0.95rem; font-weight:600; color:#e2e8f0;
    margin:28px 0 6px 0; display:flex; align-items:center; gap:8px;
}

/* Chart wrap */
.chart-wrap {
    background:#111827; border:1px solid #1f2937;
    border-radius:14px; padding:8px; margin-bottom:18px;
}

/* Lever card */
.lever-group {
    background:#111827; border:1px solid #1f2937;
    border-radius:12px; padding:18px 20px; margin-bottom:12px;
}
.lever-title { font-size:0.85rem; font-weight:600; color:#e2e8f0; margin-bottom:4px; }
.lever-desc  { font-size:0.74rem; color:#475569; line-height:1.5; margin-bottom:12px; }

/* Impact tag */
.tag-positive { color:#34d399; background:rgba(52,211,153,0.1); border:1px solid rgba(52,211,153,0.25);
                border-radius:8px; padding:2px 9px; font-size:0.7rem; font-weight:500; }
.tag-negative { color:#f87171; background:rgba(248,113,113,0.1); border:1px solid rgba(248,113,113,0.25);
                border-radius:8px; padding:2px 9px; font-size:0.7rem; font-weight:500; }
.tag-neutral  { color:#94a3b8; background:rgba(148,163,184,0.1); border:1px solid rgba(148,163,184,0.2);
                border-radius:8px; padding:2px 9px; font-size:0.7rem; font-weight:500; }

.summary-panel {
    background:rgba(56,189,248,0.06);
    border:1px solid rgba(56,189,248,0.18);
    border-radius:14px;
    padding:16px 18px;
    margin:4px 0 18px 0;
}
.summary-title {
    font-size:0.9rem;
    font-weight:700;
    color:#e2e8f0;
    margin-bottom:6px;
}
.summary-text {
    font-size:0.8rem;
    color:#cbd5e1;
    line-height:1.65;
}
.driver-grid {
    display:grid;
    grid-template-columns:repeat(2, minmax(0, 1fr));
    gap:14px;
    margin:8px 0 18px 0;
}
.driver-card {
    background:#111827;
    border:1px solid #1f2937;
    border-radius:12px;
    padding:16px 18px;
}
.driver-label {
    font-size:0.72rem;
    color:#64748b;
    text-transform:uppercase;
    letter-spacing:0.06em;
    font-weight:600;
    margin-bottom:8px;
}
.driver-value {
    font-size:0.92rem;
    color:#f8fafc;
    font-weight:700;
    margin-bottom:6px;
}
.driver-sub {
    font-size:0.78rem;
    color:#94a3b8;
    line-height:1.6;
}

@media (max-width: 900px) {
    .driver-grid { grid-template-columns:1fr; }
}

/* Comparison table */
.comp-table { width:100%; border-collapse:collapse; font-size:0.8rem; }
.comp-table th {
    text-align:left; padding:9px 14px; color:#475569;
    font-weight:600; font-size:0.7rem; text-transform:uppercase;
    letter-spacing:0.05em; border-bottom:1px solid #1f2937;
    background:#0f1623;
}
.comp-table td { padding:10px 14px; border-bottom:1px solid #1f2937; color:#cbd5e1; }
.comp-table tr:last-child td { border-bottom:none; }
.comp-table tr:hover td { background:rgba(52,211,153,0.02); }
</style>
""", unsafe_allow_html=True)

apply_theme("#2dd4bf")
render_sidebar_status()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def prepare_series(df):
    return prepare_monthly_series(df)


def holts_forecast(series, periods, alpha=0.3, beta=0.1):
    return run_forecast(series, "holts", periods, alpha=alpha, beta=beta)


def apply_levers(forecast_df, growth_pct, seasonality_boost, discount_impact,
                 marketing_boost, churn_impact, new_channel_pct):
    """Apply all scenario levers to a base forecast and return adjusted values."""
    return service_apply_levers(
        forecast_df,
        growth_pct=growth_pct,
        seasonality_boost=seasonality_boost,
        discount_impact=discount_impact,
        marketing_boost=marketing_boost,
        churn_impact=churn_impact,
        new_channel_pct=new_channel_pct,
    )


# ─────────────────────────────────────────────
# CHART THEME
# ─────────────────────────────────────────────
CHART_BG = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font         =dict(family="Inter", color="#94a3b8", size=11),
    title_font   =dict(family="Inter", color="#e2e8f0", size=13, weight="bold"),
    legend       =dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1f2937", borderwidth=1,
                       font=dict(color="#94a3b8", size=10)),
    margin       =dict(l=16, r=16, t=44, b=16),
    xaxis        =dict(gridcolor="#1f2937", linecolor="#1f2937",
                       tickcolor="#1f2937", tickfont=dict(size=10)),
    yaxis        =dict(gridcolor="#1f2937", linecolor="#1f2937",
                       tickcolor="#1f2937", tickfont=dict(size=10)),
)


raw_df, is_sample = load_sales_data(include_segments=False, copy_uploaded=False)
raw_df["Date"] = pd.to_datetime(raw_df["Date"])
monthly = prepare_series(raw_df)

# ═════════════════════════════════════════════
# HERO
# ═════════════════════════════════════════════
st.markdown("""
<div class="hero-header">
  <div class="badge">🎮 MODULE 04 — SCENARIO SIMULATION</div>
  <div class="hero-title">What-If Scenario Simulator</div>
  <div class="hero-sub">Adjust business levers and instantly see how they impact your forecasted revenue</div>
</div>
""", unsafe_allow_html=True)

if is_sample:
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(251,191,36,0.08),rgba(245,158,11,0.04));
                border:1px solid rgba(251,191,36,0.3);border-radius:12px;
                padding:14px 20px;margin-bottom:20px;display:flex;align-items:center;gap:12px;">
      <span style="font-size:1.4rem;">⚠️</span>
      <div>
        <div style="font-size:0.88rem;font-weight:600;color:#fbbf24;">Using Sample Data</div>
        <div style="font-size:0.76rem;color:#92400e;margin-top:2px;">
          Go to <b style="color:#fbbf24;">📁 Data Upload</b> in the sidebar to simulate on your real sales data.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── What is scenario simulation? ──────────
with st.expander("📖 What is Scenario Simulation? — Click to learn", expanded=False):
    st.markdown("""
    **Scenario Simulation** lets you answer *"what if"* questions about your business without
    waiting for real results.

    Instead of just seeing one forecast, you can **pull levers** — like increasing marketing spend,
    offering discounts, launching a new sales channel — and instantly see how those decisions
    would change your forecasted revenue.

    **Example questions you can answer:**
    - 💰 *"If I boost marketing by 20%, how much extra revenue do I make over 12 months?"*
    - 📉 *"If I run a 15% discount campaign, does total revenue still go up?"*
    - 🌱 *"If I add an online store channel, what's the revenue impact?"*
    - ⚠️ *"What's the worst case if 10% of my customers churn?"*

    **How the levers work:**
    | Lever | What it simulates |
    |-------|-------------------|
    | Growth Rate | Annual revenue growth assumption |
    | Seasonality Boost | Amplify or reduce seasonal peaks |
    | Discount Impact | Revenue effect of price discounts |
    | Marketing Boost | Extra revenue from campaign spend |
    | Customer Churn | Revenue lost from customer dropoff |
    | New Sales Channel | Uplift from adding a new channel |

    **The 3 scenarios (Best / Base / Worst)** are pre-built combinations of optimistic,
    neutral, and pessimistic lever settings. You can also build your own custom scenario.
    """)


# ═════════════════════════════════════════════
# STEP 1 — FORECAST HORIZON
# ═════════════════════════════════════════════
st.markdown('<div class="section-label">⚙️ Step 1 — Simulation Settings</div>',
            unsafe_allow_html=True)

s1, s2, _ = st.columns([1.5, 1.5, 3])
with s1:
    sim_periods = st.slider("Forecast Horizon (months)", 3, 24, 12, 1,
                            help="How many months to simulate into the future.")
with s2:
    base_model = st.selectbox("Base Forecast Model",
                              [
                                  "Smart Ensemble (recommended)",
                                  "Holt's Double Smoothing",
                                  "Linear Trend",
                                  "Moving Average",
                                  "Exponential Smoothing",
                                  "Seasonal Trend",
                              ],
                              help="The baseline forecast that levers are applied on top of.")

model_options = {
    "Smart Ensemble (recommended)": ("smart_ensemble", {}),
    "Holt's Double Smoothing": ("holts", {"alpha": 0.3, "beta": 0.1}),
    "Linear Trend": ("linear_trend", {}),
    "Moving Average": ("moving_average", {"window": 3}),
    "Exponential Smoothing": ("exp_smoothing", {"alpha": 0.3}),
    "Seasonal Trend": ("seasonal_trend", {"season": 12}),
}
base_model_key, base_model_kwargs = model_options[base_model]
try:
    base_forecast = run_forecast(monthly, base_model_key, sim_periods, **base_model_kwargs)
except Exception as exc:
    try:
        base_forecast, fallback_model_key = run_forecast_with_fallback(
            monthly,
            sim_periods,
            preferred_model=base_model_key,
            model_kwargs=base_model_kwargs,
        )
    except Exception as fallback_exc:
        st.error(f"Scenario simulation could not build a baseline forecast: {fallback_exc}")
        render_navigation_link(
            "pages/1_Data_Upload.py",
            "/1_Data_Upload",
            "Review uploaded data",
            "Upload or clean a dataset with enough monthly history for simulation.",
        )
        st.stop()

    fallback_label = next(
        (label for label, (key, _) in model_options.items() if key == fallback_model_key),
        fallback_model_key.replace("_", " ").title(),
    )
    st.warning(
        f"{fallback_label} is being used because {base_model} could not run on this dataset. "
        f"Original issue: {exc}"
    )
    base_model = fallback_label
    base_model_key = fallback_model_key
    base_model_kwargs = {}


# ═════════════════════════════════════════════
# STEP 2 — THREE PRESET SCENARIOS
# ═════════════════════════════════════════════
st.markdown('<div class="section-label">🎯 Step 2 — Choose a Preset Scenario or Build Custom</div>',
            unsafe_allow_html=True)

with st.expander("ℹ️ What are these scenarios?", expanded=False):
    st.markdown("""
    - 🟢 **Best Case** — Everything goes right: strong growth, successful marketing, new channel launched, low churn
    - 🔵 **Base Case** — Realistic middle ground: moderate growth, steady marketing, no major changes
    - 🔴 **Worst Case** — Challenging conditions: slow growth, heavy discounting, high churn, no new channels
    - 🎛️ **Custom** — You control every lever yourself
    """)

preset_cols = st.columns(4)
scenario_presets = {
    "🟢 Best Case": {
        "growth": 20, "seasonality": 10, "discount": 0,
        "marketing": 25, "churn": 0, "new_channel": 15,
        "color": "#34d399", "card_class": "best",
        "desc": "Strong growth, marketing campaign, new channel added, zero churn."
    },
    "🔵 Base Case": {
        "growth": 8, "seasonality": 0, "discount": -5,
        "marketing": 10, "churn": 3, "new_channel": 0,
        "color": "#38bdf8", "card_class": "base",
        "desc": "Moderate growth, mild discounting, steady marketing, small churn."
    },
    "🔴 Worst Case": {
        "growth": -5, "seasonality": -10, "discount": -15,
        "marketing": 0, "churn": 15, "new_channel": 0,
        "color": "#f87171", "card_class": "worst",
        "desc": "Negative growth, heavy discounts, high churn, no new initiatives."
    },
    "🎛️ Custom": {
        "growth": 8, "seasonality": 0, "discount": 0,
        "marketing": 0, "churn": 0, "new_channel": 0,
        "color": "#fbbf24", "card_class": "base",
        "desc": "Set all levers manually below to build your own scenario."
    },
}

selected_preset = st.session_state.get("selected_preset", "🔵 Base Case")
for i, (pname, pdata) in enumerate(scenario_presets.items()):
    with preset_cols[i]:
        is_active = selected_preset == pname
        border    = pdata["color"] if is_active else "#1f2937"
        bg        = hex_to_rgba(pdata["color"], 0.05) if is_active else "#111827"
        st.markdown(f"""
        <div style="background:{bg};border:2px solid {border};border-radius:14px;
                    padding:16px;margin-bottom:8px;min-height:110px;">
          <div style="font-size:0.9rem;font-weight:600;color:#e2e8f0;margin-bottom:5px;">{pname}</div>
          <div style="font-size:0.74rem;color:#64748b;line-height:1.5;">{pdata['desc']}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select" if not is_active else "✅ Active",
                     key=f"preset_{pname}", width="stretch"):
            st.session_state["selected_preset"] = pname
            st.rerun()

selected_preset = st.session_state.get("selected_preset", "🔵 Base Case")
preset = scenario_presets[selected_preset]


# ═════════════════════════════════════════════
# STEP 3 — LEVERS
# ═════════════════════════════════════════════
st.markdown('<div class="section-label">🎛️ Step 3 — Adjust Business Levers</div>',
            unsafe_allow_html=True)

with st.expander("ℹ️ How do the levers work?", expanded=False):
    st.markdown("""
    Each lever represents a real business decision or market condition.
    Drag the sliders to adjust the impact, then watch all charts update instantly.

    | Lever | Range | Effect |
    |-------|-------|--------|
    | **Annual Growth Rate** | -30% to +50% | Compounds monthly over the forecast period |
    | **Seasonality Boost** | -20% to +20% | Amplifies or flattens seasonal peaks/troughs |
    | **Discount Impact** | -30% to 0% | Direct revenue reduction from price cuts |
    | **Marketing Boost** | 0% to +40% | Revenue uplift that ramps up over 3 months |
    | **Customer Churn** | 0% to 25% | Gradual revenue loss from losing customers |
    | **New Sales Channel** | 0% to +30% | Flat % revenue uplift from a new channel |
    """)

is_custom = selected_preset == "🎛️ Custom"

lc1, lc2 = st.columns(2)

with lc1:
    st.markdown('<div style="font-size:0.75rem;color:#34d399;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin:0 0 10px 0;">Growth and Demand Drivers</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="lever-group">
      <div class="lever-title">📈 Annual Growth Rate</div>
      <div class="lever-desc">Expected yearly revenue growth. Positive = growing business.
      Negative = declining revenue. Applied as compound monthly growth.</div>
    </div>""", unsafe_allow_html=True)
    growth_pct = st.slider("Growth Rate %", -30, 50,
                           preset["growth"] if is_custom else preset["growth"],
                           disabled=not is_custom, key="growth",
                           help="Positive = business growing, Negative = declining")

    st.markdown("""
    <div class="lever-group">
      <div class="lever-title">🌊 Seasonality Boost</div>
      <div class="lever-desc">Amplify or dampen seasonal patterns. +10% makes peaks higher
      and troughs lower. -10% flattens the curve.</div>
    </div>""", unsafe_allow_html=True)
    seasonality_boost = st.slider("Seasonality Boost %", -20, 20,
                                  preset["seasonality"], disabled=not is_custom,
                                  key="seasonality",
                                  help="Amplifies or flattens seasonal highs and lows")

    st.markdown("""
    <div class="lever-group">
      <div class="lever-title">🏷️ Discount / Price Impact</div>
      <div class="lever-desc">Revenue effect of running discounts or price changes.
      -10% means a 10% reduction in revenue from price cuts.</div>
    </div>""", unsafe_allow_html=True)
    discount_impact = st.slider("Discount Impact %", -30, 0,
                                preset["discount"], disabled=not is_custom,
                                key="discount",
                                help="Negative values reduce revenue due to discounting")

with lc2:
    st.markdown('<div style="font-size:0.75rem;color:#fbbf24;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin:0 0 10px 0;">Campaign and Risk Drivers</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="lever-group">
      <div class="lever-title">📣 Marketing Campaign Boost</div>
      <div class="lever-desc">Extra revenue from a marketing campaign. Effect ramps up
      over 3 months as the campaign gains traction.</div>
    </div>""", unsafe_allow_html=True)
    marketing_boost = st.slider("Marketing Boost %", 0, 40,
                                preset["marketing"], disabled=not is_custom,
                                key="marketing",
                                help="Revenue uplift from marketing; ramps up over 3 months")

    st.markdown("""
    <div class="lever-group">
      <div class="lever-title">👥 Customer Churn Impact</div>
      <div class="lever-desc">Revenue lost from customers leaving over time. 10% churn
      means you lose 10% of your revenue base by end of period.</div>
    </div>""", unsafe_allow_html=True)
    churn_impact = st.slider("Churn Impact %", 0, 25,
                             preset["churn"], disabled=not is_custom,
                             key="churn",
                             help="Gradual revenue loss from customer attrition")

    st.markdown("""
    <div class="lever-group">
      <div class="lever-title">🛍️ New Sales Channel</div>
      <div class="lever-desc">Revenue uplift from launching a new channel (e.g. online store,
      marketplace, distributor). Applied as a flat % on top of forecast.</div>
    </div>""", unsafe_allow_html=True)
    new_channel_pct = st.slider("New Channel Uplift %", 0, 30,
                                preset["new_channel"], disabled=not is_custom,
                                key="new_channel",
                                help="Flat % revenue uplift from a new sales channel")

if not is_custom:
    growth_pct        = preset["growth"]
    seasonality_boost = preset["seasonality"]
    discount_impact   = preset["discount"]
    marketing_boost   = preset["marketing"]
    churn_impact      = preset["churn"]
    new_channel_pct   = preset["new_channel"]


# ═════════════════════════════════════════════
# COMPUTE ALL 4 SCENARIOS
# ═════════════════════════════════════════════
def scenario_forecast(growth, seasonality, discount, marketing, churn, channel):
    return apply_levers(base_forecast, growth, seasonality, discount, marketing, churn, channel)

custom_vals = scenario_forecast(growth_pct, seasonality_boost, discount_impact,
                                marketing_boost, churn_impact, new_channel_pct)
best_vals   = scenario_forecast(20, 10, 0,   25, 0,  15)
base_vals   = scenario_forecast(8,  0,  -5,  10, 3,   0)
worst_vals  = scenario_forecast(-5,-10, -15,  0, 15,  0)


# ═════════════════════════════════════════════
# KPI CARDS — IMPACT SUMMARY
# ═════════════════════════════════════════════
base_total   = base_forecast["Forecast"].sum()
custom_total = custom_vals.sum()
delta_vs_base = custom_total - base_total
delta_pct     = (delta_vs_base / base_total * 100) if base_total else 0
peak_month    = base_forecast["Date"].iloc[np.argmax(custom_vals)].strftime("%b %Y")
avg_monthly   = custom_vals.mean()

color_delta = "green" if delta_vs_base >= 0 else "red"
arrow       = "▲" if delta_vs_base >= 0 else "▼"

def lever_impact(g, s, d, m, c, ch):
    return apply_levers(base_forecast, g, s, d, m, c, ch).sum() - base_total

impacts = {
    "Growth Rate": lever_impact(growth_pct, 0, 0, 0, 0, 0),
    "Seasonality": lever_impact(0, seasonality_boost, 0, 0, 0, 0),
    "Discount": lever_impact(0, 0, discount_impact, 0, 0, 0),
    "Marketing": lever_impact(0, 0, 0, marketing_boost, 0, 0),
    "Churn": lever_impact(0, 0, 0, 0, churn_impact, 0),
    "New Channel": lever_impact(0, 0, 0, 0, 0, new_channel_pct),
}

positive_drivers = [(name, value) for name, value in impacts.items() if value > 0]
negative_drivers = [(name, value) for name, value in impacts.items() if value < 0]
top_positive = max(positive_drivers, key=lambda item: item[1]) if positive_drivers else None
top_negative = min(negative_drivers, key=lambda item: item[1]) if negative_drivers else None
upside_total = sum(value for value in impacts.values() if value > 0)
downside_total = abs(sum(value for value in impacts.values() if value < 0))
assumption_load = (
    abs(growth_pct) * 0.7
    + abs(seasonality_boost) * 0.6
    + abs(discount_impact) * 0.8
    + marketing_boost * 0.7
    + churn_impact * 1.0
    + new_channel_pct * 0.7
)
if assumption_load < 20:
    assumption_label = "Conservative"
    assumption_color = "#34d399"
    assumption_text = "The scenario uses mild assumptions, so it is easier to defend as a planning case."
elif assumption_load < 45:
    assumption_label = "Balanced"
    assumption_color = "#38bdf8"
    assumption_text = "The scenario has a reasonable mix of upside and risk assumptions."
elif assumption_load < 70:
    assumption_label = "Aggressive"
    assumption_color = "#fbbf24"
    assumption_text = "The scenario depends on stronger business changes, so validate the assumptions before planning inventory or budgets."
else:
    assumption_label = "High Risk"
    assumption_color = "#f87171"
    assumption_text = "The scenario stacks several strong assumptions, so treat it as a stress case rather than a base plan."

break_even_month = "Already above base"
if delta_vs_base < 0:
    break_even_month = "Not reached"
elif delta_vs_base > 0:
    cumulative_diff = np.cumsum(custom_vals - base_forecast["Forecast"].values)
    positive_months = np.where(cumulative_diff > 0)[0]
    if len(positive_months) > 0:
        break_even_month = base_forecast["Date"].iloc[int(positive_months[0])].strftime("%b %Y")

if delta_vs_base > 0:
    summary_intro = f"This scenario increases projected revenue by {delta_pct:.1f}% compared with the baseline forecast."
elif delta_vs_base < 0:
    summary_intro = f"This scenario lowers projected revenue by {abs(delta_pct):.1f}% compared with the baseline forecast."
else:
    summary_intro = "This scenario keeps projected revenue almost unchanged versus the baseline forecast."

summary_parts = []
if top_positive:
    summary_parts.append(f"The strongest uplift comes from {top_positive[0].lower()} ({format_inr(top_positive[1])}).")
if top_negative:
    summary_parts.append(f"The biggest drag comes from {top_negative[0].lower()} ({format_inr(abs(top_negative[1]))} impact).")
summary_parts.append(f"The highest projected month is {peak_month}.")
scenario_summary_text = " ".join([summary_intro] + summary_parts)

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card blue">
    <div class="kpi-label">Baseline Revenue</div>
    <div class="kpi-value">{format_inr(base_total)}</div>
    <div class="kpi-sub">forecast without scenario levers</div>
  </div>
  <div class="kpi-card {color_delta}">
    <div class="kpi-label">Scenario Revenue</div>
    <div class="kpi-value">{format_inr(custom_total)}</div>
    <div class="kpi-sub">{selected_preset} over {sim_periods} months</div>
  </div>
  <div class="kpi-card {color_delta}">
    <div class="kpi-label">Net Change</div>
    <div class="kpi-value" style="font-size:1.3rem;color:{'#34d399' if delta_vs_base>=0 else '#f87171'};">
      {'+' if delta_vs_base>=0 else '-'}{format_inr(abs(delta_vs_base))}
    </div>
    <div class="kpi-sub">difference from baseline revenue</div>
  </div>
  <div class="kpi-card amber">
    <div class="kpi-label">Percent Impact</div>
    <div class="kpi-value" style="color:{'#34d399' if delta_vs_base>=0 else '#f87171'};">{arrow} {abs(delta_pct):.1f}%</div>
    <div class="kpi-sub">change versus baseline forecast</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="summary-panel">
  <div class="summary-title">What this scenario means</div>
  <div class="summary-text">{scenario_summary_text}</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="driver-grid">
  <div class="driver-card">
    <div class="driver-label">Biggest Positive Driver</div>
    <div class="driver-value">{top_positive[0] if top_positive else 'No positive uplift selected'}</div>
    <div class="driver-sub">
      {f'Adds about {format_inr(top_positive[1])} over the baseline forecast.' if top_positive else 'Increase growth, marketing, seasonality, or new channel values to create upside.'}
    </div>
  </div>
  <div class="driver-card">
    <div class="driver-label">Biggest Risk Driver</div>
    <div class="driver-value">{top_negative[0] if top_negative else 'No major downside driver'}</div>
    <div class="driver-sub">
      {f'Removes about {format_inr(abs(top_negative[1]))} from the baseline forecast.' if top_negative else 'No discount or churn drag is meaningfully reducing revenue in the current setup.'}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="driver-grid">
  <div class="driver-card">
    <div class="driver-label">Assumption Realism</div>
    <div class="driver-value" style="color:{assumption_color};">{assumption_label}</div>
    <div class="driver-sub">{assumption_text}</div>
  </div>
  <div class="driver-card">
    <div class="driver-label">Upside vs Downside</div>
    <div class="driver-value">{format_inr(upside_total)} upside · {format_inr(downside_total)} downside</div>
    <div class="driver-sub">Compares all positive and negative lever effects before they combine into the final scenario.</div>
  </div>
  <div class="driver-card">
    <div class="driver-label">Break-even Timing</div>
    <div class="driver-value">{break_even_month}</div>
    <div class="driver-sub">First month where cumulative scenario revenue is above the baseline forecast.</div>
  </div>
  <div class="driver-card">
    <div class="driver-label">Base Forecast Model</div>
    <div class="driver-value">{base_model}</div>
    <div class="driver-sub">Scenario levers are applied on top of this selected baseline.</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════
# CHART 1 — SCENARIO COMPARISON LINE CHART
# ═════════════════════════════════════════════
st.markdown('<div class="section-label">📈 Baseline vs Your Scenario</div>',
            unsafe_allow_html=True)

with st.expander("ℹ️ How to read this chart?", expanded=False):
    st.markdown("""
    - **Grey line** = recent historical sales
    - **Blue dashed line** = baseline forecast with no extra scenario changes
    - **Highlighted line** = your selected scenario
    - **Bars** = monthly difference between the scenario and the baseline forecast
    """)

history_display = monthly.tail(24)
fig1 = go.Figure()

# historical
fig1.add_trace(go.Scatter(
    x=history_display["Date"], y=history_display["Sales"],
    mode="lines", name="Historical Sales",
    line=dict(color="#475569", width=2),
    hovertemplate="<b>%{x|%b %Y}</b><br>Actual: ₹%{y:,.0f}<extra></extra>",
))

# base forecast
fig1.add_trace(go.Scatter(
    x=base_forecast["Date"], y=base_forecast["Forecast"],
    mode="lines", name="Base Forecast (no levers)",
    line=dict(color="#38bdf8", width=1.8, dash="dot"),
    hovertemplate="<b>%{x|%b %Y}</b><br>Base: ₹%{y:,.0f}<extra></extra>",
))

# current scenario (thicker, prominent)
fig1.add_trace(go.Scatter(
    x=base_forecast["Date"], y=custom_vals,
    mode="lines+markers", name="Selected Scenario",
    line=dict(color=preset["color"], width=3),
    marker=dict(size=5, color=preset["color"]),
    hovertemplate="<b>%{x|%b %Y}</b><br>Simulated: ₹%{y:,.0f}<extra></extra>",
))

monthly_diff = custom_vals - base_forecast["Forecast"].values
fig1.add_trace(go.Bar(
    x=base_forecast["Date"],
    y=monthly_diff,
    name="Monthly Difference",
    marker=dict(color=["rgba(52,211,153,0.55)" if val >= 0 else "rgba(248,113,113,0.55)" for val in monthly_diff]),
    opacity=0.45,
    yaxis="y2",
    hovertemplate="<b>%{x|%b %Y}</b><br>Difference: ₹%{y:,.0f}<extra></extra>",
))

# vertical divider
last_hist = monthly["Date"].iloc[-1]
div_y_max = max(best_vals.max(), history_display["Sales"].max()) * 1.08
fig1.add_trace(go.Scatter(
    x=[last_hist, last_hist], y=[0, div_y_max],
    mode="lines", name="Simulation Start",
    line=dict(color="#334155", width=1.5, dash="dot"),
    hoverinfo="skip", showlegend=True,
))

fig1.update_layout(
    **CHART_BG, height=420,
    title="Baseline Forecast vs Selected Scenario",
    yaxis_title="Monthly Revenue (₹)", xaxis_title="",
    hovermode="x unified",
    yaxis2=dict(
        title=dict(text="Difference (₹)", font=dict(size=11, color="#94a3b8")),
        overlaying="y",
        side="right",
        showgrid=False,
        zeroline=False,
        tickfont=dict(size=10, color="#64748b"),
    ),
)
fig1.update_layout(
    margin=dict(l=16, r=16, t=44, b=72),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.18,
        xanchor="left",
        x=0,
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", size=10),
    ),
)
st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
st.plotly_chart(fig1, width="stretch", config={"displayModeBar": False})
st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════
# CHART 2 — LEVER IMPACT WATERFALL
# ═════════════════════════════════════════════
st.markdown('<div class="section-label">🧭 All Scenario Paths</div>',
            unsafe_allow_html=True)

fig_all = go.Figure()
fig_all.add_trace(go.Scatter(
    x=history_display["Date"], y=history_display["Sales"],
    mode="lines", name="Historical Sales",
    line=dict(color="#475569", width=2),
    hovertemplate="<b>%{x|%b %Y}</b><br>Actual: ₹%{y:,.0f}<extra></extra>",
))
fig_all.add_trace(go.Scatter(
    x=base_forecast["Date"], y=base_forecast["Forecast"],
    mode="lines", name="Base Forecast",
    line=dict(color="#38bdf8", width=1.8, dash="dot"),
    hovertemplate="<b>%{x|%b %Y}</b><br>Base: ₹%{y:,.0f}<extra></extra>",
))
fig_all.add_trace(go.Scatter(
    x=base_forecast["Date"], y=worst_vals,
    fill=None, mode="lines", showlegend=False,
    line=dict(width=0), hoverinfo="skip",
))
fig_all.add_trace(go.Scatter(
    x=base_forecast["Date"], y=best_vals,
    fill="tonexty", mode="lines", showlegend=False,
    line=dict(width=0),
    fillcolor="rgba(52,211,153,0.04)",
    hoverinfo="skip",
))
fig_all.add_trace(go.Scatter(
    x=base_forecast["Date"], y=worst_vals,
    mode="lines", name="Worst Case",
    line=dict(color="#f87171", width=1.8, dash="dash"),
    hovertemplate="<b>%{x|%b %Y}</b><br>Worst: ₹%{y:,.0f}<extra></extra>",
))
fig_all.add_trace(go.Scatter(
    x=base_forecast["Date"], y=best_vals,
    mode="lines", name="Best Case",
    line=dict(color="#34d399", width=1.8, dash="dash"),
    hovertemplate="<b>%{x|%b %Y}</b><br>Best: ₹%{y:,.0f}<extra></extra>",
))
fig_all.add_trace(go.Scatter(
    x=base_forecast["Date"], y=custom_vals,
    mode="lines+markers", name="Selected Scenario",
    line=dict(color=preset["color"], width=3),
    marker=dict(size=5, color=preset["color"]),
    hovertemplate="<b>%{x|%b %Y}</b><br>Selected: ₹%{y:,.0f}<extra></extra>",
))
fig_all.add_trace(go.Scatter(
    x=[last_hist, last_hist], y=[0, div_y_max],
    mode="lines", name="Simulation Start",
    line=dict(color="#334155", width=1.5, dash="dot"),
    hoverinfo="skip", showlegend=True,
))
fig_all.update_layout(
    **CHART_BG, height=390,
    title="All Scenarios in One View",
    yaxis_title="Monthly Revenue (₹)", xaxis_title="",
    hovermode="x unified",
)
st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
st.plotly_chart(fig_all, width="stretch", config={"displayModeBar": False})
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">🌊 Lever Impact Breakdown</div>',
            unsafe_allow_html=True)

with st.expander("ℹ️ What does this chart show?", expanded=False):
    st.markdown("""
    A **waterfall chart** shows how each individual lever contributes to the total revenue change.
    - **Green bars** = levers that increase revenue (positive impact)
    - **Red bars** = levers that decrease revenue (negative impact)
    - The final bar = net total change from all levers combined
    This helps you understand *which lever has the biggest impact* on your bottom line.
    """)

levers_df = pd.DataFrame({
    "Lever":  list(impacts.keys()),
    "Impact": list(impacts.values()),
})
levers_df = levers_df[levers_df["Impact"].abs() > 0.01]
levers_df = levers_df.sort_values("Impact", ascending=False)

colors_wf = ["#34d399" if v >= 0 else "#f87171" for v in levers_df["Impact"]]

fig2 = go.Figure(go.Bar(
    x=levers_df["Lever"],
    y=levers_df["Impact"],
    marker=dict(color=colors_wf, line=dict(width=0)),
    text=[f"{'+'if v>=0 else ''}₹{v/1000:.1f}K" for v in levers_df["Impact"]],
    textposition="outside",
    textfont=dict(color="#94a3b8", size=10),
    hovertemplate="<b>%{x}</b><br>Impact: ₹%{y:,.0f}<extra></extra>",
))

fig2.update_layout(
    **CHART_BG, height=300,
    title="Individual Lever Revenue Impact vs Base Forecast",
    yaxis_title="Revenue Impact (₹)", xaxis_title="",
    showlegend=False,
)
fig2.add_hline(y=0, line_color="#334155", line_width=1)

st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})
st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════
# CHART 3 — CUMULATIVE REVENUE COMPARISON
# ═════════════════════════════════════════════
col_l, col_r = st.columns(2)

with col_l:
    st.markdown('<div class="section-label">📊 Cumulative Revenue Growth</div>',
                unsafe_allow_html=True)
    with st.expander("ℹ️ What does this chart show?", expanded=False):
        st.markdown("""
        Shows the **running total** of revenue over the simulation period for each scenario.
        The steeper the slope, the faster revenue accumulates.
        The gap between lines at the right edge = total revenue difference between scenarios.
        """)

    fig3 = go.Figure()
    scenarios_cum = {
        "🟢 Best Case":  (np.cumsum(best_vals),  "#34d399"),
        "🔵 Base Case":  (np.cumsum(base_vals),  "#38bdf8"),
        "🔴 Worst Case": (np.cumsum(worst_vals), "#f87171"),
        f"✨ {selected_preset}": (np.cumsum(custom_vals), preset["color"]),
    }
    for sname, (cum_vals, col) in scenarios_cum.items():
        lw = 3 if "✨" in sname else 1.5
        fig3.add_trace(go.Scatter(
            x=base_forecast["Date"], y=cum_vals,
            mode="lines", name=sname,
            line=dict(color=col, width=lw),
            hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{sname}: ₹%{{y:,.0f}}<extra></extra>",
        ))
    fig3.update_layout(
        **CHART_BG, height=310,
        title="Cumulative Revenue — All Scenarios",
        yaxis_title="Cumulative Revenue (₹)", xaxis_title="",
    )
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig3, width="stretch", config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with col_r:
    st.markdown('<div class="section-label">📋 Scenario Comparison Table</div>',
                unsafe_allow_html=True)
    with st.expander("ℹ️ What does this table show?", expanded=False):
        st.markdown("""
        Side-by-side numbers for all 4 scenarios so you can directly compare:
        - **Total Revenue** = sum of all forecasted months
        - **Avg Monthly** = average per month
        - **vs Base** = how much more or less than the plain base forecast
        - **Peak Month** = the highest revenue month in that scenario
        """)

    scenario_summary = []
    for sname, s_vals in [
        ("🟢 Best Case",  best_vals),
        ("🔵 Base Case",  base_vals),
        ("🔴 Worst Case", worst_vals),
        (f"✨ {selected_preset}", custom_vals),
    ]:
        total = s_vals.sum()
        diff  = total - base_forecast["Forecast"].sum()
        scenario_summary.append({
            "Scenario":      sname,
            "Total Revenue": f"₹{total/1e6:.2f}M",
            "Avg Monthly":   f"₹{s_vals.mean():,.0f}",
            "vs Base":       f"{'+'if diff>=0 else ''}₹{diff/1000:.0f}K",
            "Peak Month":    base_forecast["Date"].iloc[np.argmax(s_vals)].strftime("%b %Y"),
        })

    comp_html = """
    <div style="background:#111827;border:1px solid #1f2937;border-radius:12px;overflow:hidden;">
    <table class="comp-table">
      <tr>
        <th>Scenario</th><th>Total Revenue</th>
        <th>Avg Monthly</th><th>vs Base</th><th>Peak Month</th>
      </tr>"""
    for row in scenario_summary:
        diff_str = row["vs Base"]
        diff_col = "#34d399" if "+" in diff_str else "#f87171"
        comp_html += f"""
      <tr>
        <td style="font-weight:500;">{row['Scenario']}</td>
        <td style="color:#e2e8f0;font-weight:600;">{row['Total Revenue']}</td>
        <td>{row['Avg Monthly']}</td>
        <td style="color:{diff_col};font-weight:600;">{diff_str}</td>
        <td style="color:#64748b;">{row['Peak Month']}</td>
      </tr>"""
    comp_html += "</table></div>"
    st.markdown(comp_html, unsafe_allow_html=True)

    # insight callout
    best_total  = best_vals.sum()
    worst_total = worst_vals.sum()
    spread      = best_total - worst_total
    st.markdown(f"""
    <div style="background:rgba(251,191,36,0.05);border:1px solid rgba(251,191,36,0.2);
                border-radius:10px;padding:14px 16px;margin-top:14px;font-size:0.79rem;color:#94a3b8;">
      💡 <b style="color:#fbbf24;">Revenue at Stake:</b>
      The gap between best and worst case is
      <b style="color:#e2e8f0;">₹{spread/1e6:.2f}M</b> over {sim_periods} months —
      that's how much your business decisions can shift the outcome.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════
# EXPORT
# ═════════════════════════════════════════════
st.markdown('<div class="section-label">⬇️ Export Simulation Results</div>',
            unsafe_allow_html=True)

export_df = pd.DataFrame({
    "Month":       base_forecast["Date"].dt.strftime("%b %Y"),
    "Base_Forecast": base_forecast["Forecast"].round(2),
    "Best_Case":   best_vals.round(2),
    "Base_Case":   base_vals.round(2),
    "Worst_Case":  worst_vals.round(2),
    "Custom_Scenario": custom_vals.round(2),
})
assumption_export_df = pd.DataFrame([
    {"Setting": "Base Model", "Value": base_model},
    {"Setting": "Selected Preset", "Value": selected_preset},
    {"Setting": "Forecast Horizon", "Value": f"{sim_periods} months"},
    {"Setting": "Growth Rate %", "Value": growth_pct},
    {"Setting": "Seasonality Boost %", "Value": seasonality_boost},
    {"Setting": "Discount Impact %", "Value": discount_impact},
    {"Setting": "Marketing Boost %", "Value": marketing_boost},
    {"Setting": "Churn Impact %", "Value": churn_impact},
    {"Setting": "New Channel Uplift %", "Value": new_channel_pct},
    {"Setting": "Assumption Realism", "Value": assumption_label},
    {"Setting": "Break-even Timing", "Value": break_even_month},
])

e1, e2, e3, e4 = st.columns([1.4, 1.4, 1.4, 1.4])
with e1:
    st.download_button(
        "⬇️ Download All Scenarios CSV",
        export_df.to_csv(index=False).encode("utf-8"),
        file_name="scenario_simulation.csv",
        mime="text/csv",
        width="stretch",
    )
with e2:
    summary_export_df = pd.DataFrame(scenario_summary) if "scenario_summary" in locals() else pd.DataFrame()
    st.download_button(
        "Download Scenario Summary CSV",
        summary_export_df.to_csv(index=False).encode("utf-8"),
        file_name="scenario_summary.csv",
        mime="text/csv",
        width="stretch",
    )
with e3:
    st.download_button(
        "Download Assumptions CSV",
        assumption_export_df.to_csv(index=False).encode("utf-8"),
        file_name="scenario_assumptions.csv",
        mime="text/csv",
        width="stretch",
    )
with e4:
    render_navigation_link(
        "pages/3_Forecasting.py",
        "/3_Forecasting",
        "🔮 Go to Forecasting →",
        "Open the forecasting page to inspect the base forecast behind this simulation.",
    )


# ═════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;margin-top:40px;padding:16px;border-top:1px solid #1f2937;
            color:#334155;font-size:0.72rem;font-family:'Inter',sans-serif;">
  AI Sales Forecasting System · Scenario Simulation · 6 levers · 4 scenarios · Live preview
</div>
""", unsafe_allow_html=True)
