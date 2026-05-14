import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
from utils.app_helpers import format_inr, load_sales_data
from utils.ui_theme import apply_theme, render_sidebar_status

from services.insight_service import (
    generate_insights,
    generate_alerts,
    detect_anomalies,
    generate_root_cause_insights,
    generate_recommendations,
)

st.set_page_config(page_title="Sales Analysis", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0a0c10; color: #e2e8f0; }
.stApp { background: #0a0c10; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0b1120 0%, #0f172a 100%); border-right: 1px solid #1f2937; }
[data-testid="stSidebarNav"] { padding-top: 1rem; }
[data-testid="stSidebarNav"] > div:first-child { display: none; }
[data-testid="stSidebarNav"]::before { content: "Sales AI"; display: block; padding: 0.2rem 0.9rem 0.1rem 0.9rem; font-size: 1.12rem; font-weight: 700; color: #f8fafc; }
[data-testid="stSidebarNav"]::after { content: "Forecasting Workspace"; display: block; padding: 0 0.9rem 0.75rem 0.9rem; margin-bottom: 0.6rem; border-bottom: 1px solid #1f2937; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #64748b; }
[data-testid="stSidebarNav"] ul { padding: 0 0.55rem 0.45rem 0.55rem; gap: 0.28rem; }
[data-testid="stSidebarNavLink"] { border-radius: 12px; padding: 0.42rem 0.55rem; border: 1px solid transparent; transition: background 0.2s ease, border-color 0.2s ease; }
[data-testid="stSidebarNavLink"]:hover { background: rgba(148, 163, 184, 0.08); border-color: #1f2937; }
[data-testid="stSidebarNavLink"] span { color: #cbd5e1 !important; font-size: 0.92rem; font-weight: 600; }
[data-testid="stSidebarNavLink"][aria-current="page"] { background: linear-gradient(135deg, rgba(99,102,241,0.16), rgba(59,130,246,0.14)); border-color: rgba(99,102,241,0.35); }
[data-testid="stSidebarNavLink"][aria-current="page"] span { color: #f8fafc !important; }
.hero-header { background: linear-gradient(135deg, #0f1623 0%, #211a12 58%, #451a03 100%); border: 1px solid rgba(251,146,60,0.34); border-radius: 16px; padding: 28px 48px; margin-bottom: 28px; min-height: 184px; box-sizing: border-box; display: flex; flex-direction: column; justify-content: center; position: relative; overflow: hidden; }
.hero-header::before { content: ''; position: absolute; top: -60px; right: -60px; width: 200px; height: 200px; border-radius: 50%; background: radial-gradient(circle, rgba(251,146,60,0.15) 0%, transparent 72%); }
.hero-title { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.01em; color: #f0f9ff; margin: 0 0 6px 0; }
.hero-sub { font-size: 0.82rem; color: #fed7aa; font-weight: 400; opacity: 0.74; }
.badge { display: inline-block; background: rgba(251,146,60,0.13); border: 1px solid rgba(251,146,60,0.34); color: #fdba74; border-radius: 20px; padding: 3px 12px; font-size: 0.72rem; font-weight: 500; margin-bottom: 12px; }
.section-label { font-size: 0.95rem; font-weight: 600; color: #e2e8f0; margin: 28px 0 8px 0; display: flex; align-items: center; gap: 8px; }
.section-desc { font-size: 0.82rem; color: #64748b; margin-bottom: 14px; line-height: 1.5; border-left: 2px solid rgba(251,146,60,0.45); padding-left: 10px; }
.kpi-card { background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 18px 20px; position: relative; overflow: hidden; height: 100%; }
.kpi-card::after { content:''; position:absolute; left:0; right:0; bottom:0; height:2px; }
.kpi-card.blue::after { background: linear-gradient(90deg,#fb923c,#f97316); }
.kpi-card.green::after { background: linear-gradient(90deg,#34d399,#10b981); }
.kpi-card.amber::after { background: linear-gradient(90deg,#fbbf24,#f59e0b); }
.kpi-card.rose::after { background: linear-gradient(90deg,#f87171,#ef4444); }
.kpi-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 500; margin-bottom: 8px; }
.kpi-value { font-size: 1.5rem; font-weight: 700; color: #f1f5f9; line-height: 1; }
.kpi-sub { font-size: 0.76rem; color: #64748b; margin-top: 6px; }
.chart-wrap { background: #111827; border: 1px solid #1f2937; border-radius: 14px; padding: 8px; margin-bottom: 20px; }
.alert { border-radius:10px; padding:12px 14px; margin-bottom:10px; font-size:0.82rem; }
.alert-danger { background:rgba(248,113,113,0.08); border:1px solid rgba(248,113,113,0.25); color:#fca5a5; }
.alert-warning { background:rgba(251,191,36,0.08); border:1px solid rgba(251,191,36,0.25); color:#fcd34d; }
.alert-info { background:rgba(251,146,60,0.08); border:1px solid rgba(251,146,60,0.25); color:#fdba74; }
.ai-box { background: linear-gradient(135deg, rgba(251,146,60,0.08), rgba(245,158,11,0.05)); border:1px solid rgba(251,146,60,0.2); border-radius:14px; padding:18px 20px; margin-bottom:14px; }
.ai-title { font-size:0.88rem; font-weight:700; color:#fdba74; margin-bottom:8px; }
.ai-item { font-size:0.8rem; color:#cbd5e1; line-height:1.6; margin-bottom:7px; }
.mix-grid { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:14px; margin:8px 0 18px 0; }
.mix-card { background:#111827; border:1px solid #1f2937; border-radius:14px; padding:16px 18px; position:relative; overflow:hidden; }
.mix-card::after { content:''; position:absolute; left:0; right:0; bottom:0; height:2px; background:linear-gradient(90deg,#fb923c,#fbbf24); }
.mix-label { font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:0.08em; font-weight:700; margin-bottom:8px; }
.mix-value { font-size:1.08rem; color:#f8fafc; font-weight:700; line-height:1.3; margin-bottom:4px; }
.mix-sub { font-size:0.78rem; color:#94a3b8; line-height:1.55; }
</style>
""", unsafe_allow_html=True)

apply_theme("#fb923c")
render_sidebar_status()

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94a3b8", size=11),
    title_font=dict(family="Inter", color="#e2e8f0", size=13),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1f2937", borderwidth=1, font=dict(color="#94a3b8", size=10)),
    margin=dict(l=16, r=16, t=40, b=16),
    xaxis=dict(gridcolor="#1f2937", linecolor="#1f2937", tickcolor="#1f2937"),
    yaxis=dict(gridcolor="#1f2937", linecolor="#1f2937", tickcolor="#1f2937"),
    colorway=["#38bdf8","#34d399","#fbbf24","#f87171","#a78bfa","#fb923c"],
)

df, is_sample = load_sales_data()
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)
df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
df["Week"] = df["Date"].dt.to_period("W").dt.start_time

monthly = df.groupby("Month")["Sales"].sum().reset_index()
weekly = df.groupby("Week")["Sales"].sum().reset_index()

insights = generate_insights(df)
alerts = generate_alerts(df)
anomalies = detect_anomalies(monthly.rename(columns={"Month": "Date"}), method="rolling_deviation", threshold=2.5, rolling_window=6)
root_causes = generate_root_cause_insights(df)
recommendations = generate_recommendations(df)

anomaly_count = len(anomalies)
top_anomaly_date = anomalies.iloc[0]["Date"] if anomaly_count > 0 else None
top_anomaly_value = anomalies.iloc[0]["Sales"] if anomaly_count > 0 else None
spike_drop_summary = "No major anomalies detected."
if anomaly_count > 0:
    top = anomalies.sort_values("anomaly_score", ascending=False).iloc[0]
    spike_drop_summary = f"Top anomaly on {pd.to_datetime(top['Date']).strftime('%d %b %Y')} with {top['severity']} severity and score {top['anomaly_score']}."

total_revenue = df["Sales"].sum()
avg_daily = df["Sales"].mean()
best_month_label = monthly.loc[monthly["Sales"].idxmax(), "Month"].strftime("%b %Y")
best_month_sales = monthly["Sales"].max()
latest_month_sales = float(monthly["Sales"].iloc[-1]) if len(monthly) else 0.0
previous_month_sales = float(monthly["Sales"].iloc[-2]) if len(monthly) >= 2 else latest_month_sales
latest_mom_pct = ((latest_month_sales - previous_month_sales) / previous_month_sales * 100) if previous_month_sales else 0.0
first_period_avg = float(monthly["Sales"].head(min(3, len(monthly))).mean()) if len(monthly) else 0.0
recent_period_avg = float(monthly["Sales"].tail(min(3, len(monthly))).mean()) if len(monthly) else 0.0
overall_growth_pct = ((recent_period_avg - first_period_avg) / first_period_avg * 100) if first_period_avg else 0.0
category_concentration_text = "No category column available."
if "Category" in df.columns:
    category_totals = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    if not category_totals.empty:
        category_concentration_text = f"{category_totals.index[0]} contributes {category_totals.iloc[0] / category_totals.sum() * 100:.1f}% of category revenue."

st.markdown("""
<div class="hero-header">
  <div class="badge">📊 MODULE 02 — AI ANALYSIS</div>
  <div class="hero-title">Sales Intelligence Dashboard</div>
  <div class="hero-sub">Interactive analytics, anomaly detection, and AI-generated business insights</div>
</div>
""", unsafe_allow_html=True)

if is_sample:
    st.markdown('<div class="alert alert-warning">⚠️ Using sample data. Upload your own dataset to generate real anomaly detection and business insights.</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f'<div class="kpi-card blue"><div class="kpi-label">Total Revenue</div><div class="kpi-value">₹{total_revenue/1e5:.2f}L</div><div class="kpi-sub">Across {len(df):,} rows</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi-card green"><div class="kpi-label">Average Daily Sales</div><div class="kpi-value">₹{avg_daily:,.0f}</div><div class="kpi-sub">Daily revenue average</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi-card amber"><div class="kpi-label">Best Month</div><div class="kpi-value">{best_month_label}</div><div class="kpi-sub">₹{best_month_sales:,.0f}</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="kpi-card rose"><div class="kpi-label">Detected Anomalies</div><div class="kpi-value">{anomaly_count}</div><div class="kpi-sub">Rolling deviation method</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Executive Interpretation</div>', unsafe_allow_html=True)
exec_signal = "growing" if overall_growth_pct > 5 else ("declining" if overall_growth_pct < -5 else "stable")
mom_signal = "up" if latest_mom_pct > 0 else ("down" if latest_mom_pct < 0 else "flat")
st.markdown(f"""
<div class="mix-grid">
  <div class="mix-card">
    <div class="mix-label">Overall Direction</div>
    <div class="mix-value">{exec_signal.title()} ({overall_growth_pct:+.1f}%)</div>
    <div class="mix-sub">Recent 3-month average compared with the first 3-month average.</div>
  </div>
  <div class="mix-card">
    <div class="mix-label">Latest Momentum</div>
    <div class="mix-value">{mom_signal.title()} ({latest_mom_pct:+.1f}%)</div>
    <div class="mix-sub">Latest month versus previous month.</div>
  </div>
  <div class="mix-card">
    <div class="mix-label">Concentration Signal</div>
    <div class="mix-value">{category_concentration_text}</div>
    <div class="mix-sub">Use this to judge dependency on one category.</div>
  </div>
</div>
""", unsafe_allow_html=True)

if alerts:
    st.markdown('<div class="section-label">🚨 Alerts</div>', unsafe_allow_html=True)
    for text, level in alerts:
        cls = {"danger": "alert-danger", "warning": "alert-warning", "info": "alert-info"}.get(level, "alert-info")
        st.markdown(f'<div class="alert {cls}">{text}</div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">🧠 AI Anomaly Detection</div>', unsafe_allow_html=True)
st.markdown("""<div class="section-desc">This section highlights unusual spikes or drops in revenue. These anomalies can point to promotions, seasonality shocks, stock-outs, one-time bulk orders, or data quality issues.</div>""", unsafe_allow_html=True)

a1, a2, a3 = st.columns(3)
with a1:
    st.markdown(f'<div class="kpi-card blue"><div class="kpi-label">Anomaly Count</div><div class="kpi-value">{anomaly_count}</div><div class="kpi-sub">Unusual points detected</div></div>', unsafe_allow_html=True)
with a2:
    anomaly_date_text = pd.to_datetime(top_anomaly_date).strftime("%d %b %Y") if top_anomaly_date is not None else "—"
    st.markdown(f'<div class="kpi-card amber"><div class="kpi-label">Top Anomaly Date</div><div class="kpi-value">{anomaly_date_text}</div><div class="kpi-sub">Most abnormal point</div></div>', unsafe_allow_html=True)
with a3:
    anomaly_value_text = f"₹{top_anomaly_value:,.0f}" if top_anomaly_value is not None else "—"
    st.markdown(f'<div class="kpi-card rose"><div class="kpi-label">Anomaly Value</div><div class="kpi-value">{anomaly_value_text}</div><div class="kpi-sub">Value at top anomaly</div></div>', unsafe_allow_html=True)

st.markdown(f'<div style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:14px 16px;margin-bottom:18px;color:#cbd5e1;"><b>Spike / Drop Summary:</b><br>{spike_drop_summary}</div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">📈 Trend Analysis with Anomaly Overlay</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly["Sales"], mode="lines+markers", name="Monthly Revenue", line=dict(color="#38bdf8", width=2.5), marker=dict(size=5)))
    if anomaly_count > 0:
        fig.add_trace(go.Scatter(x=anomalies["Date"], y=anomalies["Sales"], mode="markers", name="Anomalies", marker=dict(color="#f87171", size=11, symbol="diamond"), hovertemplate="<b>%{x|%b %Y}</b><br>Anomaly: ₹%{y:,.0f}<extra></extra>"))
    fig.update_layout(**CHART_LAYOUT, height=340, title="Monthly Revenue with Anomalies", yaxis_title="Revenue (₹)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
with c2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=weekly["Week"], y=weekly["Sales"], mode="lines", name="Weekly Revenue", line=dict(color="#34d399", width=2.2)))
    fig2.update_layout(**CHART_LAYOUT, height=340, title="Weekly Revenue Trend", yaxis_title="Revenue (₹)")
    st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})

if len(monthly) >= 2:
    st.markdown('<div class="section-label">📉 Momentum Analysis</div>', unsafe_allow_html=True)
    st.markdown("""<div class="section-desc">These charts explain whether revenue is accelerating, slowing down, or stabilizing after seasonal movement.</div>""", unsafe_allow_html=True)

    momentum = monthly.copy()
    momentum["MoM %"] = momentum["Sales"].pct_change() * 100
    momentum["Rolling 3M Avg"] = momentum["Sales"].rolling(3, min_periods=1).mean()
    momentum["MoM Color"] = np.where(momentum["MoM %"] >= 0, "#34d399", "#f87171")

    m1, m2 = st.columns(2)
    with m1:
        fig_mom = go.Figure(go.Bar(
            x=momentum["Month"],
            y=momentum["MoM %"],
            marker=dict(color=momentum["MoM Color"]),
            hovertemplate="<b>%{x|%b %Y}</b><br>MoM Change: %{y:.1f}%<extra></extra>",
        ))
        fig_mom.add_hline(y=0, line=dict(color="#475569", width=1))
        fig_mom.update_layout(**CHART_LAYOUT, height=300, title="Month-over-Month Revenue Change", yaxis_title="MoM Change (%)")
        st.plotly_chart(fig_mom, width="stretch", config={"displayModeBar": False})

    with m2:
        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(
            x=momentum["Month"],
            y=momentum["Sales"],
            mode="lines",
            name="Monthly Revenue",
            line=dict(color="rgba(56,189,248,0.45)", width=1.6),
        ))
        fig_roll.add_trace(go.Scatter(
            x=momentum["Month"],
            y=momentum["Rolling 3M Avg"],
            mode="lines+markers",
            name="3M Rolling Average",
            line=dict(color="#fbbf24", width=2.4),
            marker=dict(size=4),
        ))
        fig_roll.update_layout(**CHART_LAYOUT, height=300, title="Revenue vs 3-Month Rolling Average", yaxis_title="Revenue (₹)")
        st.plotly_chart(fig_roll, width="stretch", config={"displayModeBar": False})

st.markdown('<div class="section-label">📊 Category, Region, and Distribution</div>', unsafe_allow_html=True)
left, right = st.columns(2)
with left:
    if "Category" in df.columns:
        cat = df.groupby("Category")["Sales"].sum().sort_values(ascending=True).reset_index()
        bar = go.Figure(go.Bar(x=cat["Sales"], y=cat["Category"], orientation="h", marker=dict(color=cat["Sales"], colorscale=[[0, "#1e3a5f"], [0.5, "#0ea5e9"], [1, "#38bdf8"]])))
        bar.update_layout(**CHART_LAYOUT, height=320, title="Revenue by Category", xaxis_title="Revenue (₹)")
        st.plotly_chart(bar, width="stretch", config={"displayModeBar": False})
with right:
    if "Region" in df.columns:
        reg = df.groupby("Region")["Sales"].sum().reset_index()
        donut = go.Figure(go.Pie(labels=reg["Region"], values=reg["Sales"], hole=0.55, marker=dict(colors=["#38bdf8", "#34d399", "#fbbf24", "#f87171"])))
        donut.update_layout(**CHART_LAYOUT, height=320, title="Revenue by Region")
        st.plotly_chart(donut, width="stretch", config={"displayModeBar": False})

hist = px.histogram(df, x="Sales", nbins=30, title="Sales Distribution")
hist.update_layout(**CHART_LAYOUT, height=320)
st.plotly_chart(hist, width="stretch", config={"displayModeBar": False})

if "Category" in df.columns and "Region" in df.columns:
    st.markdown('<div class="section-label">🧩 Segment Mix Heatmap</div>', unsafe_allow_html=True)
    st.markdown("""<div class="section-desc">This heatmap shows which category-region combinations drive revenue, making concentration and weak spots easier to spot.</div>""", unsafe_allow_html=True)
    mix = df.pivot_table(
        index="Category",
        columns="Region",
        values="Sales",
        aggfunc="sum",
        fill_value=0,
    )
    fig_mix = px.imshow(
        mix,
        text_auto=".2s",
        aspect="auto",
        color_continuous_scale=[[0, "#0f172a"], [0.45, "#0ea5e9"], [1, "#34d399"]],
        title="Revenue Heatmap by Category and Region",
    )
    fig_mix.update_layout(
        **CHART_LAYOUT,
        height=max(320, 52 * len(mix.index)),
        xaxis_title="Region",
        yaxis_title="Category",
        coloraxis_colorbar=dict(title="Revenue"),
    )
    st.plotly_chart(fig_mix, width="stretch", config={"displayModeBar": False})

drilldown_options = [col for col in ["Category", "Region"] if col in df.columns]
if drilldown_options:
    st.markdown('<div class="section-label">Segment Trend Drill-down</div>', unsafe_allow_html=True)
    st.markdown("""<div class="section-desc">Use this to inspect whether one category or region is improving, declining, or behaving differently from the total sales trend.</div>""", unsafe_allow_html=True)

    d1, d2 = st.columns([1, 2])
    with d1:
        segment_col = st.selectbox("Segment type", drilldown_options)
        segment_values = (
            df.groupby(segment_col)["Sales"]
            .sum()
            .sort_values(ascending=False)
            .index
            .astype(str)
            .tolist()
        )
        selected_segment = st.selectbox("Segment value", segment_values)

    segment_df = df[df[segment_col].astype(str) == selected_segment].copy()
    segment_monthly = segment_df.groupby("Month")["Sales"].sum().reset_index()
    overall_monthly = monthly.rename(columns={"Sales": "Total Sales"})
    segment_share = segment_monthly.merge(overall_monthly, on="Month", how="left")
    segment_share["Share %"] = np.where(
        segment_share["Total Sales"] > 0,
        segment_share["Sales"] / segment_share["Total Sales"] * 100,
        0,
    )

    with d2:
        segment_total = float(segment_df["Sales"].sum())
        segment_share_total = segment_total / total_revenue * 100 if total_revenue else 0
        segment_latest = float(segment_monthly["Sales"].iloc[-1]) if not segment_monthly.empty else 0.0
        segment_prev = float(segment_monthly["Sales"].iloc[-2]) if len(segment_monthly) >= 2 else segment_latest
        segment_mom = ((segment_latest - segment_prev) / segment_prev * 100) if segment_prev else 0.0
        st.markdown(f"""
        <div class="mix-grid" style="grid-template-columns:repeat(3,minmax(0,1fr));margin-top:0;">
          <div class="mix-card"><div class="mix-label">Segment Revenue</div><div class="mix-value">₹{segment_total:,.0f}</div><div class="mix-sub">{segment_share_total:.1f}% of total revenue</div></div>
          <div class="mix-card"><div class="mix-label">Latest Segment Month</div><div class="mix-value">₹{segment_latest:,.0f}</div><div class="mix-sub">Latest available month</div></div>
          <div class="mix-card"><div class="mix-label">Segment MoM</div><div class="mix-value">{segment_mom:+.1f}%</div><div class="mix-sub">Latest vs previous month</div></div>
        </div>
        """, unsafe_allow_html=True)

    fig_segment = go.Figure()
    fig_segment.add_trace(go.Scatter(
        x=monthly["Month"],
        y=monthly["Sales"],
        mode="lines",
        name="Total Revenue",
        line=dict(color="rgba(148,163,184,0.45)", width=1.6),
        yaxis="y",
    ))
    fig_segment.add_trace(go.Scatter(
        x=segment_monthly["Month"],
        y=segment_monthly["Sales"],
        mode="lines+markers",
        name=f"{selected_segment} Revenue",
        line=dict(color="#38bdf8", width=2.6),
        marker=dict(size=5),
        yaxis="y",
    ))
    fig_segment.add_trace(go.Bar(
        x=segment_share["Month"],
        y=segment_share["Share %"],
        name="Share of Total",
        marker=dict(color="rgba(52,211,153,0.35)"),
        yaxis="y2",
        hovertemplate="<b>%{x|%b %Y}</b><br>Share: %{y:.1f}%<extra></extra>",
    ))
    segment_layout = CHART_LAYOUT.copy()
    segment_layout.update({
        "height": 340,
        "title": f"{segment_col} Trend: {selected_segment}",
        "yaxis": dict(title="Revenue (₹)", gridcolor="#1f2937", linecolor="#1f2937", tickcolor="#1f2937"),
        "yaxis2": dict(title="Share (%)", overlaying="y", side="right", showgrid=False, linecolor="#1f2937", tickcolor="#1f2937"),
        "barmode": "overlay",
    })
    fig_segment.update_layout(**segment_layout)
    st.plotly_chart(fig_segment, width="stretch", config={"displayModeBar": False})

numeric_df = df.select_dtypes(include=np.number).copy()
numeric_cols = numeric_df.columns.tolist()
if "Sales" in numeric_cols and len(numeric_cols) > 1:
    corr_df = numeric_df.corr(numeric_only=True)
    sales_corr = (
        corr_df["Sales"]
        .drop(labels=["Sales"], errors="ignore")
        .dropna()
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )

    if not sales_corr.empty:
        st.markdown('<div class="section-label">Correlation Analysis</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="section-desc">
        This section checks how strongly extra numeric columns move with Sales. Stronger relationships can help forecasting models
        learn better demand signals when those fields are available during training.
        </div>""", unsafe_allow_html=True)

        corr_left, corr_right = st.columns([1.35, 1])
        with corr_left:
            fig_corr = px.imshow(
                corr_df,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Numeric Feature Correlation Heatmap",
            )
            fig_corr.update_layout(
                **CHART_LAYOUT,
                height=380,
                xaxis_title="",
                yaxis_title="",
                coloraxis_colorbar=dict(title="Corr."),
            )
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.plotly_chart(fig_corr, width="stretch", config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        with corr_right:
            top_corr = sales_corr.head(3)
            strongest_name = top_corr.index[0]
            strongest_value = float(top_corr.iloc[0])
            direction = "positive" if strongest_value >= 0 else "negative"
            strength = (
                "strong" if abs(strongest_value) >= 0.6 else
                "moderate" if abs(strongest_value) >= 0.3 else
                "weak"
            )
            corr_items = "".join(
                f"<div class='ai-item'><b>{col}</b>: {val:+.2f} correlation with Sales</div>"
                for col, val in top_corr.items()
            )
            st.markdown(f"""
            <div class="ai-box">
              <div class="ai-title">Columns Most Related to Sales</div>
              {corr_items}
              <div class="ai-item" style="border-top:1px solid rgba(99,102,241,0.18);padding-top:10px;margin-top:10px;">
                <b>{strongest_name}</b> has the strongest {direction} relationship with Sales
                ({strongest_value:+.2f}), which is a {strength} signal. Features with stronger
                relationships may improve model performance because they add useful predictive context.
              </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown('<div class="section-label">🤖 AI Insights</div>', unsafe_allow_html=True)
st.markdown("""<div class="section-desc">This section combines anomaly detection, trend signals, concentration risk, segment weakness, and recommendation logic to make the dashboard feel more like an AI analysis system rather than a static report.</div>""", unsafe_allow_html=True)

col_ai1, col_ai2 = st.columns(2)
with col_ai1:
    st.markdown('<div class="ai-box"><div class="ai-title">Detected Insights</div>', unsafe_allow_html=True)
    for icon, title, detail, level in insights[:6]:
        st.markdown(f'<div class="ai-item">{icon} <b>{title}</b> — {detail}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="ai-box"><div class="ai-title">Root-Cause Signals</div>', unsafe_allow_html=True)
    for item in root_causes:
        st.markdown(f'<div class="ai-item">• {item}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col_ai2:
    st.markdown('<div class="ai-box"><div class="ai-title">AI Recommendations</div>', unsafe_allow_html=True)
    for item in recommendations:
        st.markdown(f'<div class="ai-item">• {item}</div>', unsafe_allow_html=True)
    if anomaly_count > 0:
        st.markdown(f'<div class="ai-item">• Detected anomalies: <b>{anomaly_count}</b></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    strongest_growth_driver = "Not enough signal"
    weakest_segment = "Not enough signal"
    forecast_warning = "No major warning signal"
    if "Category" in df.columns:
        cat_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
        strongest_growth_driver = f"Top category: {cat_sales.index[0]}"
        weakest_segment = f"Weakest category share: {cat_sales.index[-1]}"
    if anomaly_count > 0 or any("volatility" in rec.lower() for rec in recommendations):
        forecast_warning = "High volatility / anomaly presence may reduce forecast confidence."
    st.markdown('<div class="ai-box"><div class="ai-title">AI Summary Cards</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ai-item">• <b>Strongest growth driver:</b> {strongest_growth_driver}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ai-item">• <b>Weakest segment:</b> {weakest_segment}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ai-item">• <b>Forecast warning signal:</b> {forecast_warning}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

business_recommendations = []

if "Region" in df.columns:
    region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    if not region_sales.empty:
        top_region = region_sales.index[0]
        business_recommendations.append(
            f"Increase focus on the top-performing region, {top_region}, by protecting inventory availability and repeating successful campaigns there."
        )

if "Category" in df.columns:
    category_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    if not category_sales.empty:
        top_category = category_sales.index[0]
        top_category_share = category_sales.iloc[0] / category_sales.sum() * 100 if category_sales.sum() else 0
        if top_category_share >= 45:
            business_recommendations.append(
                f"Reduce dependency on {top_category}, which contributes {top_category_share:.1f}% of category revenue; grow secondary categories to lower business risk."
            )
        else:
            weak_category = category_sales.index[-1]
            business_recommendations.append(
                f"Review the weakest category, {weak_category}, and test pricing, promotion, or assortment changes to improve contribution."
            )

if len(monthly) >= 2:
    weakest_month_row = monthly.loc[monthly["Sales"].idxmin()]
    weakest_month_label = pd.to_datetime(weakest_month_row["Month"]).strftime("%b %Y")
    business_recommendations.append(
        f"Investigate weak-month performance in {weakest_month_label}; check stockouts, seasonality, discounts, or campaign gaps before setting targets."
    )

if anomaly_count > 0:
    business_recommendations.append(
        f"Monitor and explain {anomaly_count} anomaly point(s) before trusting forecasts, because unusual spikes or drops can distort model accuracy."
    )

if len(monthly) < 24:
    business_recommendations.append(
        "Improve data volume by collecting at least 24 months of consistent sales history for more reliable ML accuracy and seasonality detection."
    )
else:
    business_recommendations.append(
        "Use the available historical depth to compare forecast models regularly and keep the best-performing model updated as patterns change."
    )

business_recommendations = business_recommendations[:5]
st.markdown('<div class="section-label">Business Recommendations</div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-desc">
Decision-support actions generated from sales trends, category/region concentration, anomalies, and data readiness.
</div>""", unsafe_allow_html=True)
recommendation_cards = ""
for idx, rec in enumerate(business_recommendations, start=1):
    recommendation_cards += f"""
    <div style="background:#111827;border:1px solid #1f2937;border-radius:12px;
                padding:14px 16px;margin-bottom:10px;color:#cbd5e1;line-height:1.6;">
      <span style="display:inline-block;background:rgba(52,211,153,0.10);border:1px solid rgba(52,211,153,0.25);
                   color:#34d399;border-radius:999px;padding:2px 9px;font-size:0.68rem;font-weight:700;margin-right:8px;">
        ACTION {idx}
      </span>
      {rec}
    </div>
    """
st.markdown(recommendation_cards, unsafe_allow_html=True)

severity_counts = (
    anomalies["severity"].value_counts().reindex(["low", "medium", "high"], fill_value=0)
    if not anomalies.empty and "severity" in anomalies.columns
    else pd.Series([0, 0, 0], index=["low", "medium", "high"])
)

st.markdown('<div class="section-label">Anomaly Severity Breakdown</div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-desc">
Low, medium, and high severity anomalies help explain whether unusual sales movements are minor noise or stronger forecasting risks.
</div>""", unsafe_allow_html=True)

s1, s2, s3 = st.columns(3)
severity_cards = [
    (s1, "Low Severity", int(severity_counts["low"]), "#38bdf8", "minor unusual movement"),
    (s2, "Medium Severity", int(severity_counts["medium"]), "#fbbf24", "needs review"),
    (s3, "High Severity", int(severity_counts["high"]), "#f87171", "strong forecasting risk"),
]
for col, label, count, color, desc in severity_cards:
    with col:
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1f2937;border-radius:12px;
                    padding:18px 20px;margin-bottom:14px;">
          <div style="font-size:0.74rem;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;
                      font-weight:600;margin-bottom:8px;">{label}</div>
          <div style="font-size:1.9rem;font-weight:700;color:{color};line-height:1;">{count}</div>
          <div style="font-size:0.76rem;color:#94a3b8;margin-top:8px;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="section-label">Detected Anomaly Table</div>', unsafe_allow_html=True)
anomaly_report = anomalies.copy()
if not anomaly_report.empty and "Date" in anomaly_report.columns:
    anomaly_report["Date"] = pd.to_datetime(anomaly_report["Date"]).dt.strftime("%d %b %Y")

if anomaly_count == 0:
    st.info("No anomalies detected with the current method and threshold.")
else:
    anomaly_table = anomaly_report.copy()
    preferred_cols = ["Date", "Sales", "anomaly_score", "severity"]
    available_cols = [col for col in preferred_cols if col in anomaly_table.columns]
    if "Sales" in anomaly_table.columns:
        anomaly_table["Sales"] = anomaly_table["Sales"].apply(lambda x: f"₹{x:,.0f}")
    if "anomaly_score" in anomaly_table.columns:
        anomaly_table["anomaly_score"] = anomaly_table["anomaly_score"].round(2)
    rename_map = {
        "anomaly_score": "Score",
        "severity": "Severity",
    }
    anomaly_table = anomaly_table[available_cols].rename(columns=rename_map)
    st.dataframe(anomaly_table, width="stretch", hide_index=True)

st.download_button(
    label="Download Anomaly Report CSV",
    data=anomaly_report.to_csv(index=False).encode("utf-8"),
    file_name="anomaly_report.csv",
    mime="text/csv",
    width="stretch",
)
