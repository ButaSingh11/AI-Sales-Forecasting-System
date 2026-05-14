import pandas as pd
import streamlit as st


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return 56, 189, 248
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def apply_theme(accent: str = "#38bdf8") -> None:
    r, g, b = _hex_to_rgb(accent)
    st.markdown(
        f"""
        <style>
        :root {{
            --app-accent: {accent};
            --app-accent-soft: rgba({r}, {g}, {b}, 0.14);
            --app-accent-border: rgba({r}, {g}, {b}, 0.28);
            --app-bg: #0a0c10;
            --app-surface: #111827;
            --app-surface-2: #0f1623;
            --app-border: #1f2937;
            --app-text: #e2e8f0;
            --app-muted: #94a3b8;
            --app-subtle: #64748b;
        }}

        html, body, [class*="css"] {{
            background-color: var(--app-bg);
            color: var(--app-text);
        }}

        .stApp {{
            background: var(--app-bg);
        }}

        .main .block-container {{
            max-width: 1360px;
            padding-top: 1.4rem;
            padding-bottom: 2.75rem;
            padding-left: 1.2rem;
            padding-right: 1.2rem;
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0b1120 0%, #0f172a 100%);
            border-right: 1px solid var(--app-border);
        }}

        [data-testid="stSidebarNav"] {{
            padding-top: 1rem;
        }}

        [data-testid="stSidebarNav"] > div:first-child {{
            display: none;
        }}

        [data-testid="stSidebarNav"]::before {{
            content: "Sales AI";
            display: block;
            padding: 0.2rem 0.9rem 0.1rem 0.9rem;
            font-size: 1.12rem;
            font-weight: 700;
            color: #f8fafc;
        }}

        [data-testid="stSidebarNav"]::after {{
            content: "Forecasting Workspace";
            display: block;
            padding: 0 0.9rem 0.75rem 0.9rem;
            margin-bottom: 0.6rem;
            border-bottom: 1px solid var(--app-border);
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--app-subtle);
        }}

        [data-testid="stSidebarNav"] ul {{
            padding: 0 0.55rem 0.45rem 0.55rem;
            gap: 0.28rem;
        }}

        [data-testid="stSidebarNavLink"] {{
            border-radius: 12px;
            padding: 0.42rem 0.55rem;
            border: 1px solid transparent;
            transition: background 0.2s ease, border-color 0.2s ease, transform 0.2s ease;
        }}

        [data-testid="stSidebarNavLink"]:hover {{
            background: rgba(148, 163, 184, 0.08);
            border-color: var(--app-border);
            transform: translateX(1px);
        }}

        [data-testid="stSidebarNavLink"] span {{
            color: #cbd5e1 !important;
            font-size: 0.92rem;
            font-weight: 600;
        }}

        [data-testid="stSidebarNav"] ul li:first-child [data-testid="stSidebarNavLink"] span {{
            display: none !important;
        }}

        [data-testid="stSidebarNav"] ul li:first-child [data-testid="stSidebarNavLink"]::after {{
            content: "Home";
            color: #cbd5e1;
            font-size: 0.92rem;
            font-weight: 600;
        }}

        [data-testid="stSidebarNav"] ul li:first-child [data-testid="stSidebarNavLink"][aria-current="page"]::after {{
            color: #f8fafc;
        }}

        [data-testid="stSidebarNavLink"][aria-current="page"] {{
            background: linear-gradient(135deg, var(--app-accent-soft), rgba(255,255,255,0.02));
            border-color: var(--app-accent-border);
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
        }}

        [data-testid="stSidebarNavLink"][aria-current="page"] span {{
            color: #f8fafc !important;
        }}

        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button {{
            min-height: 2.9rem;
            border-radius: 11px !important;
            border: 1px solid var(--app-border) !important;
            background: linear-gradient(135deg, rgba({r}, {g}, {b}, 0.18), rgba({r}, {g}, {b}, 0.10)) !important;
            color: #f8fafc !important;
            font-weight: 600 !important;
            box-shadow: none !important;
            transition: transform 0.18s ease, border-color 0.18s ease, background 0.18s ease !important;
        }}

        div[data-testid="stButton"] > button:hover,
        div[data-testid="stDownloadButton"] > button:hover {{
            border-color: var(--app-accent-border) !important;
            background: linear-gradient(135deg, rgba({r}, {g}, {b}, 0.25), rgba({r}, {g}, {b}, 0.14)) !important;
            transform: translateY(-1px);
        }}

        div[data-testid="stButton"] > button:focus,
        div[data-testid="stDownloadButton"] > button:focus,
        div[data-testid="stButton"] > button:focus-visible,
        div[data-testid="stDownloadButton"] > button:focus-visible {{
            border-color: var(--app-accent) !important;
            box-shadow: 0 0 0 0.2rem rgba({r}, {g}, {b}, 0.16) !important;
        }}

        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stSelectbox"] > div[data-baseweb="select"] > div,
        div[data-testid="stMultiSelect"] > div[data-baseweb="select"] > div,
        div[data-testid="stDateInput"] input {{
            background: var(--app-surface) !important;
            border: 1px solid var(--app-border) !important;
            border-radius: 12px !important;
            color: var(--app-text) !important;
        }}

        div[data-testid="stTextInput"] input:focus,
        div[data-testid="stTextArea"] textarea:focus,
        div[data-testid="stNumberInput"] input:focus,
        div[data-testid="stDateInput"] input:focus {{
            border-color: var(--app-accent) !important;
            box-shadow: 0 0 0 0.2rem rgba({r}, {g}, {b}, 0.14) !important;
        }}

        div[data-testid="stFileUploader"] {{
            background: rgba(15, 22, 35, 0.55);
            border: 1px solid var(--app-border);
            border-radius: 14px;
            padding: 0.35rem;
        }}

        div[data-testid="stFileUploaderDropzone"] {{
            background: transparent !important;
            border: 1px dashed var(--app-accent-border) !important;
            border-radius: 12px !important;
        }}

        div[data-testid="stExpander"] {{
            border: 1px solid var(--app-border);
            border-radius: 14px;
            background: rgba(17, 24, 39, 0.48);
            overflow: hidden;
            margin-bottom: 0.85rem;
        }}

        div[data-testid="stExpander"] summary {{
            background: rgba(15, 22, 35, 0.8);
        }}

        div[data-testid="stExpander"] summary:hover {{
            background: rgba(15, 22, 35, 0.96);
        }}

        div[data-testid="stAlert"] {{
            border-radius: 12px;
        }}

        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {{
            border: 1px solid var(--app-border);
            border-radius: 12px;
            overflow: hidden;
        }}

        div[data-testid="stMetric"] {{
            background: rgba(17, 24, 39, 0.68);
            border: 1px solid var(--app-border);
            border-radius: 14px;
            padding: 0.7rem 0.85rem;
        }}

        div[data-testid="stMetric"] label {{
            color: var(--app-subtle) !important;
        }}

        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
            color: #f8fafc;
        }}

        .stMarkdown p, .stMarkdown li {{
            color: #cbd5e1;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.5rem;
        }}

        .stTabs [data-baseweb="tab"] {{
            background: rgba(17, 24, 39, 0.72);
            border: 1px solid var(--app-border);
            border-radius: 999px;
            color: var(--app-muted);
            padding: 0.38rem 0.9rem;
        }}

        .stTabs [aria-selected="true"] {{
            color: #f8fafc !important;
            border-color: var(--app-accent-border) !important;
            background: var(--app-accent-soft) !important;
        }}

        .ui-section-note {{
            font-size: 0.8rem;
            color: var(--app-muted);
            margin: -0.1rem 0 0.9rem 0;
            line-height: 1.6;
        }}

        .ui-mini-title {{
            font-size: 0.72rem;
            font-weight: 700;
            color: var(--app-subtle);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin: 0 0 0.55rem 0;
        }}

        .sidebar-status-card {{
            background: rgba(15, 22, 35, 0.9);
            border: 1px solid var(--app-border);
            border-radius: 14px;
            padding: 0.9rem 0.95rem;
            margin: 0.9rem 0 0.85rem 0;
        }}

        .sidebar-status-head {{
            font-size: 0.72rem;
            font-weight: 700;
            color: var(--app-subtle);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.65rem;
        }}

        .sidebar-status-title {{
            font-size: 0.85rem;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 0.22rem;
        }}

        .sidebar-status-sub {{
            font-size: 0.74rem;
            color: var(--app-muted);
            line-height: 1.55;
        }}

        .sidebar-status-chip {{
            display: inline-block;
            margin-top: 0.55rem;
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            font-size: 0.7rem;
            font-weight: 600;
            border: 1px solid var(--app-accent-border);
            background: var(--app-accent-soft);
            color: #f8fafc;
        }}

        .sidebar-status-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.55rem;
            margin-top: 0.75rem;
        }}

        .sidebar-mini-stat {{
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.04);
            border-radius: 10px;
            padding: 0.48rem 0.55rem;
        }}

        .sidebar-mini-label {{
            font-size: 0.65rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--app-subtle);
            margin-bottom: 0.18rem;
        }}

        .sidebar-mini-value {{
            font-size: 0.78rem;
            color: #f8fafc;
            font-weight: 600;
            line-height: 1.35;
        }}

        @media (max-width: 1100px) {{
            .main .block-container {{
                padding-left: 1rem;
                padding-right: 1rem;
            }}
        }}

        @media (max-width: 900px) {{
            .summary-grid,
            .status-grid,
            .kpi-grid,
            .workflow-grid,
            .reco-grid,
            .driver-grid {{
                grid-template-columns: 1fr !important;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_status() -> None:
    with st.sidebar:
        has_data = "uploaded_df" in st.session_state and st.session_state.uploaded_df is not None

        if not has_data:
            st.markdown(
                """
                <div class="sidebar-status-card">
                  <div class="sidebar-status-head">Workspace Status</div>
                  <div class="sidebar-status-title">No dataset loaded</div>
                  <div class="sidebar-status-sub">Upload a CSV or Excel file to unlock page-specific analysis, forecasting, simulation, and chatbot answers.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        df = st.session_state.uploaded_df
        file_name = st.session_state.get("file_name", "Uploaded dataset")

        date_range_text = "Not available"
        if isinstance(df, pd.DataFrame) and "Date" in df.columns:
            date_values = pd.to_datetime(df["Date"], errors="coerce").dropna()
            if not date_values.empty:
                date_range_text = f"{date_values.min():%d %b %Y} to {date_values.max():%d %b %Y}"

        selected_model = st.session_state.get("selected_model")
        selected_model_text = selected_model.replace("_", " ").title() if selected_model else "Not selected"

        st.markdown(
            f"""
            <div class="sidebar-status-card">
              <div class="sidebar-status-head">Workspace Status</div>
              <div class="sidebar-status-title">{file_name}</div>
              <div class="sidebar-status-sub">Your working dataset is ready for cross-page analysis.</div>
              <div class="sidebar-status-chip">Live session data</div>
              <div class="sidebar-status-grid">
                <div class="sidebar-mini-stat">
                  <div class="sidebar-mini-label">Rows</div>
                  <div class="sidebar-mini-value">{len(df):,}</div>
                </div>
                <div class="sidebar-mini-stat">
                  <div class="sidebar-mini-label">Columns</div>
                  <div class="sidebar-mini-value">{df.shape[1]}</div>
                </div>
                <div class="sidebar-mini-stat">
                  <div class="sidebar-mini-label">Date Range</div>
                  <div class="sidebar-mini-value">{date_range_text}</div>
                </div>
                <div class="sidebar-mini-stat">
                  <div class="sidebar-mini-label">Forecast Model</div>
                  <div class="sidebar-mini-value">{selected_model_text}</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
