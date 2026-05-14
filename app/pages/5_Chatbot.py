import streamlit as st
import pandas as pd
import html
import re
import plotly.graph_objects as go
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from services.chatbot_service import build_hybrid_chat_response
from utils.app_helpers import load_sales_data
from utils.ui_theme import apply_theme, render_sidebar_status

CHAT_RESPONSE_VERSION = 5

st.set_page_config(page_title="Sales AI Chatbot", page_icon="🤖", layout="wide")

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
    background: linear-gradient(135deg, #0f1623 0%, #241426 58%, #4a1238 100%);
    border: 1px solid rgba(244,114,182,0.34);
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
    top: -50px; right: -50px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(244,114,182,0.16) 0%, transparent 72%);
}
.hero-title { font-size:1.6rem; font-weight:700; letter-spacing:-0.01em; color:#f0f9ff; margin:0 0 6px 0; }
.hero-sub { font-size:0.82rem; color:#fbcfe8; opacity:0.74; }
.badge {
    display:inline-block;
    background:rgba(244,114,182,0.13);
    border:1px solid rgba(244,114,182,0.34);
    color:#f9a8d4;
    border-radius:20px;
    padding:3px 12px;
    font-size:0.72rem;
    font-weight:500;
    margin-bottom:10px;
}

.mode-pill {
    display:inline-block;
    border-radius:999px;
    padding:3px 10px;
    font-size:0.68rem;
    font-weight:600;
    margin-bottom:8px;
}
.mode-direct { color:#34d399; background:rgba(52,211,153,0.12); border:1px solid rgba(52,211,153,0.25); }
.mode-hybrid { color:#f9a8d4; background:rgba(244,114,182,0.12); border:1px solid rgba(244,114,182,0.25); }
.mode-llm { color:#fbbf24; background:rgba(251,191,36,0.12); border:1px solid rgba(251,191,36,0.25); }

.chat-user { display:flex; justify-content:flex-end; margin-bottom:16px; }
.chat-user .bubble {
    background:linear-gradient(135deg,#9d174d,#be185d);
    border:1px solid rgba(244,114,182,0.34);
    color:#e2e8f0;
    border-radius:16px 16px 4px 16px;
    padding:12px 16px;
    max-width:75%;
    font-size:0.85rem;
    line-height:1.6;
}

.chat-bot { display:flex; align-items:flex-start; gap:10px; margin-bottom:16px; }
.bot-avatar {
    width:34px; height:34px; border-radius:50%;
    background:linear-gradient(135deg,#be185d,#7c2d12);
    display:flex; align-items:center; justify-content:center;
    font-size:1rem; flex-shrink:0; margin-top:2px;
}
.chat-bot .bubble {
    background:#111827;
    border:1px solid #1f2937;
    color:#e2e8f0;
    border-radius:4px 16px 16px 16px;
    padding:12px 16px;
    max-width:85%;
    font-size:0.85rem;
    line-height:1.7;
}
.chat-bot .bubble p { margin:0 0 8px 0; }
.chat-bot .bubble p:last-child { margin:0; }
.chat-bot .bubble h4 {
    margin:0 0 10px 0;
    color:#f8fafc;
    font-size:0.95rem;
    font-weight:700;
}
.chat-bot .bubble ul {
    margin:10px 0 0 0;
    padding-left:18px;
}
.chat-bot .bubble li {
    margin-bottom:6px;
}
.chat-bot .bubble strong { color:#bfdbfe; }

.quick-note {
    background:#111827;
    border:1px solid #1f2937;
    border-radius:12px;
    padding:14px 16px;
    margin-bottom:16px;
}
.quick-note-title {
    font-size:0.84rem;
    font-weight:600;
    color:#e2e8f0;
    margin-bottom:4px;
}
.quick-note-desc {
    font-size:0.76rem;
    color:#64748b;
    line-height:1.6;
}

.chat-container {
    max-height: 520px;
    overflow-y: auto;
    padding: 16px;
    background: #0d1117;
    border: 1px solid #1f2937;
    border-radius: 14px;
    margin-bottom: 16px;
}

.transparency-box {
    background:rgba(244,114,182,0.06);
    border:1px solid rgba(244,114,182,0.2);
    border-radius:12px;
    padding:14px 16px;
    margin-top:8px;
}
.transparency-box b { color:#fbcfe8; }
.chart-panel {
    background:#111827;
    border:1px solid #1f2937;
    border-radius:12px;
    padding:8px;
    margin:0 0 16px 44px;
    max-width:680px;
}

.prompt-group {
    background:#111827;
    border:1px solid #1f2937;
    border-radius:14px;
    padding:14px;
    height:100%;
    margin-bottom:12px;
}
.prompt-group-title {
    font-size:0.72rem;
    color:#64748b;
    text-transform:uppercase;
    letter-spacing:0.08em;
    font-weight:700;
    margin-bottom:10px;
}
.prompt-group-sub {
    font-size:0.76rem;
    color:#94a3b8;
    line-height:1.5;
    margin-bottom:12px;
}
.chat-toolbar {
    background:#111827;
    border:1px solid rgba(244,114,182,0.18);
    border-radius:14px;
    padding:14px 16px;
    margin:0 0 18px 0;
}
.chat-toolbar-label {
    font-size:0.7rem;
    color:#64748b;
    text-transform:uppercase;
    letter-spacing:0.08em;
    font-weight:700;
    margin-bottom:6px;
}
.chat-toolbar-value {
    font-size:0.9rem;
    color:#e2e8f0;
    font-weight:600;
}
.chat-toolbar-sub {
    font-size:0.76rem;
    color:#94a3b8;
    line-height:1.5;
    margin-top:4px;
}
.capability-strip {
    display:grid;
    grid-template-columns:repeat(4,minmax(0,1fr));
    gap:10px;
    margin:0 0 18px 0;
}
.capability-pill {
    background:#111827;
    border:1px solid rgba(244,114,182,0.16);
    border-radius:12px;
    padding:10px 12px;
    font-size:0.76rem;
    color:#cbd5e1;
    line-height:1.45;
}
.capability-pill b { color:#fbcfe8; }
@media (max-width: 900px) {
    .capability-strip { grid-template-columns:1fr 1fr; }
}
</style>
""", unsafe_allow_html=True)

apply_theme("#f472b6")
render_sidebar_status()


def render_user_message(text: str):
    st.markdown(f"""
    <div class="chat-user">
      <div class="bubble">{text}</div>
    </div>
    """, unsafe_allow_html=True)


def mode_label_html(mode: str) -> str:
    if mode == "direct_data":
        return '<div class="mode-pill mode-direct">Direct data analysis</div>'
    if mode == "hybrid":
        return '<div class="mode-pill mode-hybrid">Hybrid: data + Gemini explanation</div>'
    return '<div class="mode-pill mode-llm">AI explanation</div>'


def format_chat_html(text: str) -> str:
    """Render lightweight Markdown-style chatbot text inside the custom HTML bubble."""
    safe_text = html.escape(text)
    safe_text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", safe_text)

    blocks = []
    list_items = []
    for raw_line in safe_text.splitlines():
        line = raw_line.strip()
        if not line:
            if list_items:
                blocks.append("<ul>" + "".join(list_items) + "</ul>")
                list_items = []
            continue

        if line.startswith("### "):
            if list_items:
                blocks.append("<ul>" + "".join(list_items) + "</ul>")
                list_items = []
            blocks.append(f"<h4>{line[4:]}</h4>")
        elif line.startswith("- "):
            list_items.append(f"<li>{line[2:]}</li>")
        else:
            if list_items:
                blocks.append("<ul>" + "".join(list_items) + "</ul>")
                list_items = []
            blocks.append(f"<p>{line}</p>")

    if list_items:
        blocks.append("<ul>" + "".join(list_items) + "</ul>")

    return "".join(blocks)


def render_assistant_message(text: str, mode: str):
    formatted = format_chat_html(text)
    st.markdown(f"""
    <div class="chat-bot">
      <div class="bot-avatar">🤖</div>
      <div>
        {mode_label_html(mode)}
        <div class="bubble">{formatted}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def build_chat_chart(chart: dict):
    if not chart:
        return None

    chart_df = pd.DataFrame(chart.get("data", []))
    if chart_df.empty or not {"label", "sales"}.issubset(chart_df.columns):
        return None

    palette = ["#38bdf8", "#34d399", "#fbbf24", "#a78bfa", "#f87171", "#fb923c", "#818cf8"]
    colors = [palette[i % len(palette)] for i in range(len(chart_df))]
    if chart.get("type") == "line":
        fig = go.Figure(go.Scatter(
            x=chart_df["label"],
            y=chart_df["sales"],
            mode="lines+markers",
            line=dict(color="#38bdf8", width=2.5),
            marker=dict(size=5, color="#38bdf8"),
            hovertemplate="<b>%{x}</b><br>Sales: ₹%{y:,.0f}<extra></extra>",
        ))
    else:
        fig = go.Figure(go.Bar(
            x=chart_df["label"],
            y=chart_df["sales"],
            marker=dict(
                color=colors,
                line=dict(width=0),
            ),
            text=[f"₹{value:,.0f}" for value in chart_df["sales"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Sales: ₹%{y:,.0f}<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#94a3b8", size=11),
        title=dict(text=chart.get("title", "Sales Difference"), font=dict(color="#e2e8f0", size=13)),
        margin=dict(l=16, r=16, t=44, b=16),
        height=260,
        showlegend=False,
        xaxis=dict(gridcolor="#1f2937", linecolor="#1f2937", title=""),
        yaxis=dict(gridcolor="#1f2937", linecolor="#1f2937", title="Revenue (₹)"),
    )
    return fig


def render_assistant_chart(chart: dict, chart_key: str):
    fig = build_chat_chart(chart)
    if fig is None:
        return

    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=chart_key)
    st.markdown('</div>', unsafe_allow_html=True)


def render_assistant_download(table_rows: list, download_key: str):
    if not table_rows:
        return

    export_df = pd.DataFrame(table_rows)
    if export_df.empty:
        return

    st.download_button(
        "Download forecast CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="chatbot_forecast.csv",
        mime="text/csv",
        key=download_key,
    )


def process_prompt(prompt: str, df: pd.DataFrame):
    st.session_state.chat_messages.append({
        "role": "user",
        "content": prompt
    })

    history_for_service = []
    for msg in st.session_state.chat_messages[:-1]:
        if msg["role"] == "user":
            history_for_service.append({"role": "user", "content": msg["content"]})
        else:
            history_for_service.append({"role": "assistant", "content": msg["content"]})

    result = build_hybrid_chat_response(
        query=prompt,
        df=df,
        api_key=st.session_state.gemini_api_key or None,
        conversation_messages=history_for_service,
        model="gemini-2.5-flash",
    )

    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": result["final_answer"],
        "mode": result["mode"],
        "intent": result["intent"],
        "chart": result.get("chart"),
        "table": result.get("table"),
    })


df, is_sample = load_sales_data()

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""

if st.session_state.get("chat_response_version") != CHAT_RESPONSE_VERSION:
    st.session_state.chat_messages = []
    st.session_state.chat_response_version = CHAT_RESPONSE_VERSION


st.markdown("""
<div class="hero-header">
  <div class="badge">🤖 MODULE 05 — GEMINI CHATBOT</div>
  <div class="hero-title">Grounded Sales Intelligence Assistant</div>
  <div class="hero-sub">Ask quick sales questions and get concise, data-backed business answers</div>
</div>
""", unsafe_allow_html=True)

if is_sample:
    st.info("You are using sample data. Upload your own dataset to get project-specific answers.")

toolbar_col1, toolbar_col2, toolbar_col3 = st.columns([1.25, 1.25, 0.9])
with toolbar_col1:
    st.markdown(
        f"""
        <div class="chat-toolbar">
          <div class="chat-toolbar-label">Dataset</div>
          <div class="chat-toolbar-value">{"Sample sales data" if is_sample else "Uploaded sales data"}</div>
          <div class="chat-toolbar-sub">{len(df):,} rows available for grounded answers.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with toolbar_col2:
    st.markdown(
        f"""
        <div class="chat-toolbar">
          <div class="chat-toolbar-label">Forecast behavior</div>
          <div class="chat-toolbar-value">Respects scope and horizon</div>
          <div class="chat-toolbar-sub">Category-specific prompts use that category, and 2-year prompts use 24 months.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with toolbar_col3:
    st.markdown(
        f"""
        <div class="chat-toolbar">
          <div class="chat-toolbar-label">Conversation</div>
          <div class="chat-toolbar-value">{len(st.session_state.chat_messages)} message(s)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Clear chat", width="stretch"):
        st.session_state.chat_messages = []
        st.rerun()

st.markdown("""
<div class="capability-strip">
  <div class="capability-pill"><b>Data</b><br>summary, columns, cleaning, quality</div>
  <div class="capability-pill"><b>Models</b><br>best model, MAPE, RMSE, confidence</div>
  <div class="capability-pill"><b>Forecasts</b><br>overall, category, region, horizon</div>
  <div class="capability-pill"><b>Project</b><br>workflow, modules, tech stack</div>
</div>
""", unsafe_allow_html=True)

with st.expander("🔑 Gemini API setup", expanded=False):
    st.session_state.gemini_api_key = st.text_input(
        "Gemini API Key",
        value=st.session_state.gemini_api_key,
        type="password",
        placeholder="Paste your Gemini API key here",
    )

    st.markdown("""
    <div class="quick-note">
      <div class="quick-note-title">How answers are generated</div>
      <div class="quick-note-desc">
        Structured questions are answered from direct dataset calculations first.
        Gemini is used mainly for broader reasoning, explanation, and business-style follow-up questions.
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("**Smart prompts**")
st.markdown('<div class="ui-section-note">Use these as quick starters when you want a focused answer instead of typing a full question.</div>', unsafe_allow_html=True)

prompt_groups = [
    (
        "Data",
        "Understand the loaded dataset and data quality.",
        [
            ("Show overall summary", "Show overall summary of my sales data"),
            ("Show columns", "What columns are in my data?"),
            ("Data quality", "Show data quality, missing values, and duplicates"),
            ("Compare this month vs last month", "Compare this month vs last month"),
        ],
    ),
    (
        "Performance",
        "Find what stands out and where performance is strongest or weakest.",
        [
            ("Detect anomalies", "Detect anomalies in my data"),
            ("Best category", "What is my best category?"),
            ("Best region", "What is my best region?"),
            ("Weak region", "What is my weakest region?"),
        ],
    ),
    (
        "Forecast",
        "Explore future outlook and simple what-if questions.",
        [
            ("Forecast next 6 months", "Forecast next 6 months"),
            ("Forecast by category, 2 years", "Forecast by category for next 2 years"),
            ("What if sales increase by 20%", "What if sales increase by 20%"),
        ],
    ),
    (
        "Model & Project",
        "Ask how the forecasting system works.",
        [
            ("Best model", "Which model is best and why?"),
            ("Explain MAPE", "What is MAPE and how should I read it?"),
            ("Project workflow", "What does this project do and what modules are included?"),
        ],
    ),
]

group_cols = st.columns(4)
prompt_idx = 0
for group_col, (group_title, group_sub, items) in zip(group_cols, prompt_groups):
    with group_col:
        st.markdown(
            f"""
            <div class="prompt-group">
              <div class="prompt-group-title">{group_title}</div>
              <div class="prompt-group-sub">{group_sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for label, smart_prompt in items:
            if st.button(label, key=f"smart_prompt_{prompt_idx}", width="stretch"):
                process_prompt(smart_prompt, df)
                st.rerun()
            prompt_idx += 1

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

if not st.session_state.chat_messages:
    st.markdown("""
    <div style="text-align:center;padding:56px 20px;">
      <div style="font-size:2.4rem;margin-bottom:12px;">🤖</div>
      <div style="font-size:0.9rem;font-weight:600;color:#e2e8f0;margin-bottom:8px;">
        Hello! I'm your structured Sales AI Assistant
      </div>
      <div style="font-size:0.78rem;color:#475569;max-width:480px;margin:0 auto;line-height:1.6;">
        Ask about trends, categories, regions, anomalies, comparisons, or sales behavior.
        Press Enter to send your question.
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for idx, msg in enumerate(st.session_state.chat_messages):
        if msg["role"] == "user":
            render_user_message(msg["content"])
        else:
            render_assistant_message(msg["content"], msg.get("mode", "llm_only"))
            render_assistant_chart(msg.get("chart"), f"chat-chart-{idx}")
            render_assistant_download(msg.get("table"), f"chat-download-{idx}")

st.markdown('</div>', unsafe_allow_html=True)

prompt = st.chat_input("Ask something about your sales data. Press Enter to send.")
if prompt and prompt.strip():
    process_prompt(prompt.strip(), df)
    st.rerun()

st.markdown("""
<div class="transparency-box">
  <b>Transparency note:</b><br>
  Answers are grounded in your uploaded sales data whenever possible.
  Direct questions use calculated values first, while Gemini can add broader explanation when an API key is available.
</div>
""", unsafe_allow_html=True)
