import streamlit as st
import pandas as pd
import numpy as np
import io
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
from utils.app_helpers import normalize_text_columns, render_navigation_link
from utils.ui_theme import apply_theme, render_sidebar_status

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Data Upload",
    page_icon="📁",
    layout="wide",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
def build_cleaning_report(raw_df: pd.DataFrame, clean_df: pd.DataFrame, date_col, sales_col) -> dict:
    rows_before = len(raw_df)
    rows_after = len(clean_df)
    duplicates_removed = int(raw_df.duplicated().sum())
    missing_before = int(raw_df.isnull().sum().sum())
    missing_after = int(clean_df.isnull().sum().sum())
    missing_fixed = max(missing_before - missing_after, 0)

    converted_columns = set()
    if date_col and date_col in raw_df.columns:
        if date_col != "Date" or pd.api.types.is_datetime64_any_dtype(raw_df[date_col]):
            converted_columns.add(date_col)
    if sales_col and sales_col in raw_df.columns:
        if sales_col != "Sales" or pd.api.types.is_numeric_dtype(raw_df[sales_col]):
            converted_columns.add(sales_col)
    converted_cols = len(converted_columns)

    outliers_detected = 0
    if sales_col and sales_col in raw_df.columns and pd.api.types.is_numeric_dtype(raw_df[sales_col]):
        sales_values = raw_df[sales_col].dropna()
        if len(sales_values) >= 4:
            q1 = sales_values.quantile(0.25)
            q3 = sales_values.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers_detected = int(((sales_values < lower) | (sales_values > upper)).sum())

    return {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "duplicates_removed": duplicates_removed,
        "missing_fixed": missing_fixed,
        "columns_converted": converted_cols,
        "outliers_detected": outliers_detected,
    }


def render_cleaning_report(report: dict):
    if not report:
        return

    st.markdown('<div class="section-label">Before vs After Cleaning Report</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="status-grid">
      <div class="status-card info">
        <div class="status-label">Rows Before</div>
        <div class="status-value">{report.get("rows_before", 0):,}</div>
        <div class="status-sub">raw uploaded rows</div>
      </div>
      <div class="status-card ok">
        <div class="status-label">Rows After</div>
        <div class="status-value">{report.get("rows_after", 0):,}</div>
        <div class="status-sub">model-ready rows</div>
      </div>
      <div class="status-card {'ok' if report.get("duplicates_removed", 0) == 0 else 'warn'}">
        <div class="status-label">Duplicates Removed</div>
        <div class="status-value">{report.get("duplicates_removed", 0):,}</div>
        <div class="status-sub">repeated rows cleaned</div>
      </div>
      <div class="status-card {'ok' if report.get("missing_fixed", 0) == 0 else 'warn'}">
        <div class="status-label">Missing Values Fixed</div>
        <div class="status-value">{report.get("missing_fixed", 0):,}</div>
        <div class="status-sub">filled or removed safely</div>
      </div>
      <div class="status-card info">
        <div class="status-label">Columns Converted</div>
        <div class="status-value">{report.get("columns_converted", 0):,}</div>
        <div class="status-sub">standardized for analysis</div>
      </div>
      <div class="status-card {'ok' if report.get("outliers_detected", 0) == 0 else 'warn'}">
        <div class="status-label">Outliers Detected</div>
        <div class="status-value">{report.get("outliers_detected", 0):,}</div>
        <div class="status-sub">IQR check on sales</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_date_range_card(container, clean_df: pd.DataFrame):
    if "Date" in clean_df.columns:
        date_values = pd.to_datetime(clean_df["Date"], errors="coerce").dropna()
        if not date_values.empty:
            start_date = date_values.min().strftime("%d %b %Y")
            end_date = date_values.max().strftime("%d %b %Y")
            date_html = f"{start_date}<br><span style='color:#64748b;font-size:0.95rem;'>to</span><br>{end_date}"
        else:
            date_html = "—"
    else:
        date_html = "—"

    container.markdown(f"""
    <div style="padding:0.35rem 0;">
      <div style="font-size:0.875rem;font-weight:600;color:#f8fafc;margin-bottom:0.35rem;">
        Date Range
      </div>
      <div style="font-size:1.35rem;font-weight:600;color:#f8fafc;line-height:1.25;white-space:normal;">
        {date_html}
      </div>
    </div>
    """, unsafe_allow_html=True)


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

/* hero */
.hero-header {
    background: linear-gradient(135deg, #0f1623 0%, #0f1f33 58%, #082f49 100%);
    border: 1px solid rgba(14,165,233,0.34);
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
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(56,189,248,0.14) 0%, transparent 72%);
}
.hero-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: #f0f9ff;
    margin: 0 0 6px 0;
}
.hero-sub {
    font-size: 0.82rem;
    color: #7dd3fc;
    font-weight: 400;
    opacity: 0.72;
    letter-spacing: 0;
    text-transform: none;
}
.badge {
    display: inline-block;
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.34);
    color: #38bdf8;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.72rem;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.05em;
    margin-bottom: 12px;
}

/* upload zone */
.upload-zone {
    background: #0f1623;
    border: 2px dashed #1e3a5f;
    border-radius: 16px;
    padding: 36px;
    text-align: center;
    margin-bottom: 24px;
    transition: border-color .2s;
}
.upload-zone:hover { border-color: #38bdf8; }
.upload-icon { font-size: 2.5rem; margin-bottom: 10px; }
.upload-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 4px;
}
.upload-hint { font-size: 0.8rem; color: #475569; }

/* status cards */
.status-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin: 20px 0 24px 0; }
.status-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 18px 20px;
    min-height: 118px;
    box-sizing: border-box;
}
.status-card.ok    { border-left: 3px solid #34d399; }
.status-card.warn  { border-left: 3px solid #fbbf24; }
.status-card.error { border-left: 3px solid #f87171; }
.status-card.info  { border-left: 3px solid #38bdf8; }
.status-label {
    font-size: 0.68rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'Inter', sans-serif;
    margin-bottom: 6px;
}
.status-value {
    font-family: 'Inter', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1;
}
.status-sub { font-size: 0.75rem; color: #475569; margin-top: 3px; }

/* section labels */
.section-label {
    font-family: 'Inter', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 24px 0 6px 0;
}
.section-desc {
    font-size: 0.82rem;
    color: #475569;
    margin-bottom: 14px;
    line-height: 1.5;
    border-left: 2px solid #1e3a5f;
    padding-left: 10px;
}

/* validation checks */
.check-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 8px;
    margin-bottom: 8px;
    font-size: 0.83rem;
}
.check-icon { font-size: 1rem; min-width: 20px; }
.check-text { color: #cbd5e1; flex: 1; }
.check-badge-ok    { background:rgba(52,211,153,0.12); color:#34d399; border:1px solid rgba(52,211,153,0.3); border-radius:12px; padding:2px 10px; font-size:.7rem; font-family:'Inter',sans-serif; }
.check-badge-warn  { background:rgba(251,191,36,0.12);  color:#fbbf24; border:1px solid rgba(251,191,36,0.3);  border-radius:12px; padding:2px 10px; font-size:.7rem; font-family:'Inter',sans-serif; }
.check-badge-error { background:rgba(248,113,113,0.12); color:#f87171; border:1px solid rgba(248,113,113,0.3); border-radius:12px; padding:2px 10px; font-size:.7rem; font-family:'Inter',sans-serif; }

/* column map table */
.col-map {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 20px;
}
.col-map-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #1f2937;
    font-size: 0.82rem;
}
.col-map-row:last-child { border-bottom: none; }
.col-name { font-family:'Inter',sans-serif; color:#38bdf8; }
.col-type { color:#64748b; font-size:.75rem; }
.col-sample { color:#94a3b8; font-size:.75rem; max-width:200px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }

/* success banner */
.success-banner {
    background: linear-gradient(135deg, rgba(52,211,153,0.1), rgba(16,185,129,0.05));
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 12px;
    padding: 20px 24px;
    margin: 20px 0;
    display: flex;
    align-items: center;
    gap: 16px;
}
.success-icon { font-size: 2rem; }
.success-text { font-family:'Inter',sans-serif; font-size:1rem; font-weight:700; color:#34d399; }
.success-sub  { font-size:0.8rem; color:#475569; margin-top:3px; }

/* sample download */
.sample-box {
    background: #0f1623;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 24px;
}
.sample-title { font-family:'Inter',sans-serif; font-size:.95rem; font-weight:700; color:#93c5fd; margin-bottom:6px; }
.sample-desc  { font-size:.8rem; color:#475569; margin-bottom:12px; }

/* dataframe styling override */
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* streamlit button */
.stButton > button {
    background: linear-gradient(135deg, rgba(56,189,248,0.22), rgba(56,189,248,0.12)) !important;
    color: #f8fafc !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    border: 1px solid rgba(56,189,248,0.28) !important;
    border-radius: 11px !important;
    padding: 10px 18px !important;
    font-size: 0.88rem !important;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(56,189,248,0.28), rgba(56,189,248,0.16)) !important;
    border-color: rgba(56,189,248,0.42) !important;
}
</style>
""", unsafe_allow_html=True)

apply_theme("#38bdf8")
render_sidebar_status()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def generate_sample_csv() -> bytes:
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    n = len(dates)
    trend      = np.linspace(8000, 22000, n)
    seasonality = 3500 * np.sin(2 * np.pi * dates.dayofyear / 365)
    weekly     = 800  * np.sin(2 * np.pi * dates.dayofweek / 7)
    noise      = np.random.normal(0, 600, n)
    sales      = np.maximum(trend + seasonality + weekly + noise, 0).round(2)
    categories = np.random.choice(["Electronics","Apparel","Home & Garden","Sports","Beauty"], n, p=[0.3,0.25,0.2,0.15,0.1])
    regions    = np.random.choice(["North","South","East","West"], n, p=[0.3,0.25,0.25,0.2])
    orders     = (sales / np.random.uniform(45, 85, n)).astype(int)
    returns    = np.random.randint(0, 30, n)
    df = pd.DataFrame({"Date":dates,"Sales":sales,"Category":categories,"Region":regions,"Orders":orders,"Returns":returns})
    return df.to_csv(index=False).encode("utf-8")


def try_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
    return df


def detect_date_col(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        if "date" in col.lower():
            return col
    return None


def detect_sales_col(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if any(k in col.lower() for k in ["sale","revenue","amount","total","price","value"]):
            if pd.api.types.is_numeric_dtype(df[col]):
                return col
    # fallback: first numeric col
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None


def run_validations(df: pd.DataFrame, date_col, sales_col):
    checks = []

    # 1. rows
    n = len(df)
    checks.append(("Minimum Rows (≥ 30)", n >= 30,
                   f"{n:,} rows found", "warn" if 30 <= n < 100 else ("ok" if n >= 100 else "error")))

    # 2. date column
    checks.append(("Date Column Detected", date_col is not None,
                   f"`{date_col}`" if date_col else "No date column found",
                   "ok" if date_col else "error"))

    # 3. sales column
    checks.append(("Sales / Revenue Column", sales_col is not None,
                   f"`{sales_col}`" if sales_col else "No numeric sales column found",
                   "ok" if sales_col else "error"))

    # 4. missing values
    total_missing = df.isnull().sum().sum()
    miss_pct = total_missing / (df.shape[0] * df.shape[1]) * 100
    checks.append(("Missing Values", miss_pct < 20,
                   f"{total_missing} nulls ({miss_pct:.1f}%)",
                   "ok" if miss_pct == 0 else ("warn" if miss_pct < 20 else "error")))

    # 5. duplicates
    dups = df.duplicated().sum()
    checks.append(("Duplicate Rows", dups == 0,
                   f"{dups} duplicates found",
                   "ok" if dups == 0 else "warn"))

    # 6. negative sales
    if sales_col:
        sales_numeric = pd.to_numeric(df[sales_col], errors="coerce")
        neg = (sales_numeric < 0).sum()
        checks.append(("No Negative Sales", neg == 0,
                       f"{neg} negative values",
                       "ok" if neg == 0 else "warn"))

    if date_col and sales_col and date_col in df.columns and sales_col in df.columns:
        temp = df[[date_col, sales_col]].copy()
        temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
        temp[sales_col] = pd.to_numeric(temp[sales_col], errors="coerce")
        temp = temp.dropna(subset=[date_col, sales_col])

        if not temp.empty:
            monthly = temp.groupby(temp[date_col].dt.to_period("M"))[sales_col].sum()
            month_count = len(monthly)
            history_level = "ok" if month_count >= 12 else ("warn" if month_count >= 6 else "error")
            checks.append((
                "Monthly Forecast History",
                month_count >= 6,
                f"{month_count} monthly observation(s) found",
                history_level,
            ))

            full_months = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
            missing_months = full_months.difference(monthly.index)
            checks.append((
                "Monthly Gaps",
                len(missing_months) == 0,
                f"{len(missing_months)} missing month(s) in the date range",
                "ok" if len(missing_months) == 0 else "warn",
            ))

    if date_col and "Category" in df.columns:
        duplicate_date_category = df.duplicated(subset=[date_col, "Category"]).sum()
        checks.append((
            "Duplicate Date / Category Rows",
            duplicate_date_category == 0,
            f"{duplicate_date_category} repeated date-category row(s)",
            "ok" if duplicate_date_category == 0 else "warn",
        ))

    return checks


def clean_dataframe_live(df: pd.DataFrame, date_col, sales_col):
    """Yields (step_label, detail, level, df_snapshot) at each cleaning step."""
    df = df.copy()

    # ── STEP 1: Remove duplicates ──────────────
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    yield ("🗑️ Remove Duplicate Rows",
           f"Scanned {before:,} rows — removed {removed} duplicate{'s' if removed != 1 else ''}. {len(df):,} rows remain.",
           "ok" if removed == 0 else "warn", df.copy())

    # ── STEP 2: Drop rows missing Date/Sales ───
    critical_cols = [c for c in [date_col, sales_col] if c]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    dropped = before - len(df)
    yield ("🚫 Drop Rows with Missing Date / Sales",
           f"Checked critical columns {critical_cols} — dropped {dropped} row{'s' if dropped != 1 else ''} with nulls. {len(df):,} rows remain.",
           "ok" if dropped == 0 else "warn", df.copy())

    # ── STEP 3: Fill numeric missing values ────
    num_cols = df.select_dtypes(include="number").columns.tolist()
    filled_count = 0
    for col in num_cols:
        n = df[col].isnull().sum()
        if n > 0:
            df[col] = df[col].fillna(df[col].median())
            filled_count += n
    yield ("🔢 Fill Missing Numeric Values with Median",
           f"Checked {len(num_cols)} numeric column(s) — filled {filled_count} missing cell{'s' if filled_count != 1 else ''} using column median.",
           "ok" if filled_count == 0 else "warn", df.copy())

    # ── STEP 4: Fill text/category missing values
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    filled_text = 0
    for col in obj_cols:
        n = df[col].isnull().sum()
        if n > 0:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(mode_val)
            filled_text += n
    yield ("🔤 Fill Missing Text Values with Mode",
           f"Checked {len(obj_cols)} text column(s) — filled {filled_text} missing cell{'s' if filled_text != 1 else ''} using most common value.",
           "ok" if filled_text == 0 else "warn", df.copy())

    # ── STEP 5: Normalize text encoding ────────
    before_text_df = df.copy()
    df = normalize_text_columns(df)
    changed_cells = 0
    for col in before_text_df.select_dtypes(include=["object", "string", "category"]).columns:
        changed_cells += int((before_text_df[col].astype(str) != df[col].astype(str)).sum())
    yield ("🧼 Normalize Text Labels",
           f"Checked text labels for encoding issues and standardized spacing. Updated {changed_cells} cell{'s' if changed_cells != 1 else ''}.",
           "ok" if changed_cells == 0 else "warn", df.copy())

    # ── STEP 6: Clip negative sales ────────────
    if sales_col and sales_col in df.columns:
        neg_count = (df[sales_col] < 0).sum()
        df[sales_col] = df[sales_col].clip(lower=0)
    else:
        neg_count = 0
    yield ("📉 Clip Negative Sales Values to Zero",
           f"Found {neg_count} negative value{'s' if neg_count != 1 else ''} in '{sales_col}' — all clipped to 0.",
           "ok" if neg_count == 0 else "warn", df.copy())

    # ── STEP 7: Standardise column names ───────
    rename_map = {}
    if date_col  and date_col  != "Date":  rename_map[date_col]  = "Date"
    if sales_col and sales_col != "Sales": rename_map[sales_col] = "Sales"
    if rename_map:
        df = df.rename(columns=rename_map)
    renamed_str = ", ".join([f"'{k}' → '{v}'" for k, v in rename_map.items()]) if rename_map else "No renaming needed"
    yield ("🏷️ Standardise Column Names",
           f"{renamed_str}. Analysis page will now recognise Date & Sales columns automatically.",
           "ok", df.copy())

    # ── STEP 8: Reset index ────────────────────
    df = df.reset_index(drop=True)
    yield ("✅ Reset Index & Finalise",
           f"Index reset. Final dataset: {len(df):,} clean rows × {df.shape[1]} columns — ready for Analysis.",
           "ok", df.copy())


# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="badge">📁 MODULE 01 — DATA UPLOAD</div>
  <div class="hero-title">Upload Your Sales Data</div>
  <div class="hero-sub">CSV or Excel · Auto-cleaned · Validated before analysis</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SAMPLE FILE DOWNLOAD
# ─────────────────────────────────────────────
st.markdown("""
<div class="sample-box">
  <div class="sample-title">🧪 Don't have a file yet?</div>
  <div class="sample-desc">Download our sample sales dataset to explore all features.
  It contains 3 years of daily sales with Category, Region, Orders & Returns columns.</div>
</div>
""", unsafe_allow_html=True)

st.download_button(
    label="⬇️ Download Sample CSV",
    data=generate_sample_csv(),
    file_name="sample_sales.csv",
    mime="text/csv",
)

st.markdown("---")

# ─────────────────────────────────────────────
# FILE UPLOADER
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">📂 Upload Your File</div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-desc">
Supported formats: <b>CSV</b> and <b>Excel (.xlsx)</b>. Your file should have at least
a <b>Date</b> column and a <b>Sales / Revenue</b> column. Extra columns like Category,
Region, Orders are optional but unlock more charts in Analysis.
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop your CSV or Excel file here",
    type=["csv", "xlsx"],
    label_visibility="collapsed",
)

# ─────────────────────────────────────────────
# PROCESS FILE
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# PROCESS FILE
# ─────────────────────────────────────────────

# ── If data already cleaned, show success state immediately ──
if st.session_state.get("upload_success") and uploaded_file is None:
    clean_df = st.session_state["uploaded_df"]
    removed  = st.session_state.get("rows_cleaned", 0)
    fname    = st.session_state.get("file_name", "your file")

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(52,211,153,0.08),rgba(16,185,129,0.04));
                border:1px solid rgba(52,211,153,0.25);border-radius:12px;
                padding:18px 22px;margin:16px 0;display:flex;align-items:center;gap:16px;">
      <div style="font-size:1.8rem;">🎉</div>
      <div>
        <div style="font-size:0.95rem;font-weight:700;color:#34d399;margin-bottom:3px;">
          Data Already Loaded!
        </div>
        <div style="font-size:0.78rem;color:#475569;">
          <b style="color:#94a3b8;">{fname}</b> · {len(clean_df):,} clean rows · ready for analysis
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    render_cleaning_report(st.session_state.get("cleaning_report"))

    # cleaning log if available
    if st.session_state.get("cleaning_steps"):
        with st.expander("🧹 View Cleaning Log", expanded=False):
            for i, (l, d, lv) in enumerate(st.session_state["cleaning_steps"]):
                icon   = "✅" if lv == "ok" else "⚠️"
                color  = "#34d399" if lv == "ok" else "#fbbf24"
                bg     = "rgba(52,211,153,0.06)" if lv == "ok" else "rgba(251,191,36,0.06)"
                border = "rgba(52,211,153,0.25)" if lv == "ok" else "rgba(251,191,36,0.25)"
                st.markdown(f"""
                <div style="display:flex;align-items:flex-start;gap:12px;
                            background:{bg};border:1px solid {border};
                            border-radius:10px;padding:12px 16px;margin-bottom:8px;">
                  <div style="font-size:1.1rem;margin-top:1px;">{icon}</div>
                  <div>
                    <div style="font-size:0.85rem;font-weight:600;color:{color};margin-bottom:3px;">
                      Step {i+1} — {l}
                    </div>
                    <div style="font-size:0.78rem;color:#94a3b8;line-height:1.5;">{d}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">👀 Cleaned Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(clean_df.head(8), width="stretch", height=240)

    # stats
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Clean Rows",    f"{len(clean_df):,}")
    s2.metric("Columns",       f"{clean_df.shape[1]}")
    render_date_range_card(s3, clean_df)
    s4.metric("Total Revenue", f"₹{clean_df['Sales'].sum():,.0f}"
              if "Sales" in clean_df.columns else "—")

    st.download_button(
        label="Download Cleaned Dataset CSV",
        data=clean_df.to_csv(index=False).encode("utf-8"),
        file_name=f"cleaned_{fname.replace(' ', '_')}.csv",
        mime="text/csv",
        width="stretch",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(56,189,248,0.06),rgba(14,165,233,0.03));
                border:1px solid rgba(56,189,248,0.2);border-radius:14px;
                padding:20px 24px;display:flex;align-items:center;
                justify-content:space-between;gap:20px;margin-bottom:16px;">
      <div>
        <div style="font-size:0.92rem;font-weight:600;color:#e2e8f0;margin-bottom:4px;">
          ✅ Your data is clean and ready!
        </div>
        <div style="font-size:0.78rem;color:#475569;">
          Head over to the Analysis page to explore charts, trends and insights.
        </div>
      </div>
      <div style="font-size:1.6rem;">📊</div>
    </div>
    """, unsafe_allow_html=True)

    b1, b2, _ = st.columns([1.2, 1.4, 2])
    with b1:
        render_navigation_link(
            "pages/2_Analysis.py",
            "/2_Analysis",
            "📊 Go to Analysis →",
            "Open the analysis dashboard for the cleaned dataset.",
        )
    with b2:
        if st.button("🔄 Upload Another File", width="stretch"):
            for key in ["uploaded_df", "upload_success", "file_name", "rows_cleaned", "cleaning_steps", "cleaning_report"]:
                st.session_state.pop(key, None)
            st.rerun()

elif uploaded_file is not None:

    # ── read ──────────────────────────────────
    try:
        if uploaded_file.name.endswith(".xlsx"):
            raw_df = pd.read_excel(uploaded_file)
        else:
            raw_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")
        st.stop()

    raw_df = normalize_text_columns(raw_df)
    raw_df = try_parse_dates(raw_df)
    date_col  = detect_date_col(raw_df)
    sales_col = detect_sales_col(raw_df)

    # ── STATUS CARDS ──────────────────────────
    rows, cols = raw_df.shape
    missing_pct = raw_df.isnull().sum().sum() / (rows * cols) * 100
    dups = raw_df.duplicated().sum()

    st.markdown(f"""
    <div class="status-grid">
      <div class="status-card info">
        <div class="status-label">Total Rows</div>
        <div class="status-value">{rows:,}</div>
        <div class="status-sub">data points loaded</div>
      </div>
      <div class="status-card info">
        <div class="status-label">Columns</div>
        <div class="status-value">{cols}</div>
        <div class="status-sub">features detected</div>
      </div>
      <div class="status-card {'ok' if missing_pct == 0 else 'warn' if missing_pct < 20 else 'error'}">
        <div class="status-label">Missing Values</div>
        <div class="status-value">{missing_pct:.1f}%</div>
        <div class="status-sub">{'clean ✓' if missing_pct == 0 else 'will be filled'}</div>
      </div>
      <div class="status-card {'ok' if dups == 0 else 'warn'}">
        <div class="status-label">Duplicates</div>
        <div class="status-value">{dups}</div>
        <div class="status-sub">{'none found ✓' if dups == 0 else 'will be removed'}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── COLUMN MAP ────────────────────────────
    st.markdown('<div class="section-label">🗂️ Detected Columns</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-desc">
    These are the columns found in your file. The system auto-detects your Date and
    Sales columns. Make sure they are correctly identified before proceeding.
    </div>""", unsafe_allow_html=True)

    # ── header row
    h1, h2, h3, h4 = st.columns([2, 2, 3, 1])
    h1.markdown("<span style='color:#475569;font-size:0.72rem;font-weight:600;'>COLUMN NAME</span>", unsafe_allow_html=True)
    h2.markdown("<span style='color:#475569;font-size:0.72rem;font-weight:600;'>DATA TYPE</span>",   unsafe_allow_html=True)
    h3.markdown("<span style='color:#475569;font-size:0.72rem;font-weight:600;'>SAMPLE VALUE</span>",unsafe_allow_html=True)
    h4.markdown("<span style='color:#475569;font-size:0.72rem;font-weight:600;'>ROLE</span>",        unsafe_allow_html=True)
    st.markdown("<hr style='margin:4px 0 8px 0;border-color:#1f2937;'>", unsafe_allow_html=True)

    for col in raw_df.columns:
        dtype  = str(raw_df[col].dtype)
        sample = str(raw_df[col].dropna().iloc[0]) if not raw_df[col].dropna().empty else "—"
        sample = (sample[:35] + "...") if len(sample) > 35 else sample

        if col == date_col:
            badge = "🔵 DATE"
            badge_color = "#38bdf8"
        elif col == sales_col:
            badge = "🟢 SALES"
            badge_color = "#34d399"
        else:
            badge = "· COL"
            badge_color = "#475569"

        c1, c2, c3, c4 = st.columns([2, 2, 3, 1])
        c1.markdown(f"<span style='font-family:monospace;color:#38bdf8;font-size:0.82rem;'>{col}</span>", unsafe_allow_html=True)
        c2.markdown(f"<span style='color:#64748b;font-size:0.78rem;'>{dtype}</span>",                     unsafe_allow_html=True)
        c3.markdown(f"<span style='color:#94a3b8;font-size:0.78rem;'>e.g. {sample}</span>",               unsafe_allow_html=True)
        c4.markdown(f"<span style='color:{badge_color};font-size:0.72rem;font-weight:600;'>{badge}</span>", unsafe_allow_html=True)

    st.markdown("<hr style='margin:8px 0 20px 0;border-color:#1f2937;'>", unsafe_allow_html=True)

    # ── COLUMN OVERRIDE ───────────────────────
    with st.expander("⚙️ Override column detection (optional)", expanded=False):
        oc1, oc2 = st.columns(2)
        date_col  = oc1.selectbox("Date column",  raw_df.columns.tolist(),
                                   index=raw_df.columns.tolist().index(date_col) if date_col in raw_df.columns else 0)
        sales_col = oc2.selectbox("Sales column", raw_df.columns.tolist(),
                                   index=raw_df.columns.tolist().index(sales_col) if sales_col in raw_df.columns else 0)

    # ── VALIDATION CHECKS ─────────────────────
    st.markdown('<div class="section-label">✅ Data Validation Checks</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-desc">
    Automated checks run on your dataset before it's passed to the Analysis and Forecasting
    modules. Green = all good. Yellow = minor issue (auto-fixed). Red = critical problem.
    </div>""", unsafe_allow_html=True)

    checks = run_validations(raw_df, date_col, sales_col)
    has_error = False
    for name, passed, detail, level in checks:
        icon  = "✅" if level == "ok" else ("⚠️" if level == "warn" else "❌")
        badge = f'<span class="check-badge-{level}">{"PASS" if level=="ok" else ("WARN" if level=="warn" else "FAIL")}</span>'
        st.markdown(f"""
        <div class="check-row">
          <span class="check-icon">{icon}</span>
          <span class="check-text"><b>{name}</b> — {detail}</span>
          {badge}
        </div>""", unsafe_allow_html=True)
        if level == "error":
            has_error = True

    # ── DATA PREVIEW ──────────────────────────
    st.markdown('<div class="section-label">👀 Raw Data Preview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-desc">
    First 10 rows of your uploaded file — exactly as loaded, before any cleaning.
    Scroll right to see all columns.
    </div>""", unsafe_allow_html=True)
    st.dataframe(raw_df.head(10), width="stretch", height=280)

    # ── CLEAN & SAVE ──────────────────────────
    st.markdown("---")
    if has_error:
        st.warning("⚠️ Critical validation errors found. Please fix your file and re-upload.")
    else:
        btn_col, _ = st.columns([1, 2])
        with btn_col:
            start_clean = st.button("🚀 Clean & Save Data for Analysis")

        if start_clean:
            st.markdown('<div class="section-label">🧹 Cleaning Your Data</div>', unsafe_allow_html=True)

            # ── progress bar + status text ──
            progress_bar  = st.progress(0)
            status_text   = st.empty()
            steps_done    = []
            total_steps   = 8
            clean_df      = raw_df.copy()

            # placeholder for the final step cards (rendered ONCE at end)
            log_placeholder = st.empty()

            for idx, (label, detail, level, clean_df) in enumerate(
                clean_dataframe_live(raw_df.copy(), date_col, sales_col)
            ):
                steps_done.append((label, detail, level))
                pct = int(((idx + 1) / total_steps) * 100)

                # update progress + status smoothly
                progress_bar.progress(pct)
                icon = "✅" if level == "ok" else "⚠️"
                status_text.markdown(
                    f"<span style='font-size:0.82rem;color:#94a3b8;'>"
                    f"{icon} Step {idx+1}/{total_steps} — <b style='color:#e2e8f0;'>{label}</b></span>",
                    unsafe_allow_html=True
                )

            # ── all steps done — render log ONCE cleanly ──
            progress_bar.progress(100)
            status_text.markdown(
                "<span style='font-size:0.82rem;color:#34d399;font-weight:600;'>"
                "✅ All steps complete — data is clean!</span>",
                unsafe_allow_html=True
            )

            # build full log html in one shot
            log_html = ""
            for i, (l, d, lv) in enumerate(steps_done):
                icon   = "✅" if lv == "ok" else "⚠️"
                color  = "#34d399" if lv == "ok" else "#fbbf24"
                bg     = "rgba(52,211,153,0.05)" if lv == "ok" else "rgba(251,191,36,0.05)"
                border = "rgba(52,211,153,0.2)" if lv == "ok" else "rgba(251,191,36,0.2)"
                log_html += f"""
                <div style="display:flex;align-items:flex-start;gap:12px;
                            background:{bg};border:1px solid {border};
                            border-radius:10px;padding:11px 16px;margin-bottom:7px;">
                  <span style="font-size:1rem;margin-top:2px;min-width:20px;">{icon}</span>
                  <div style="flex:1;">
                    <div style="font-size:0.82rem;font-weight:600;color:{color};margin-bottom:2px;">
                      Step {i+1} &nbsp;·&nbsp; {l}
                    </div>
                    <div style="font-size:0.75rem;color:#64748b;line-height:1.5;">{d}</div>
                  </div>
                </div>"""

            log_placeholder.markdown(
                f"<div style='margin-top:12px;'>{log_html}</div>",
                unsafe_allow_html=True
            )

            # save to session state
            st.session_state["uploaded_df"]    = clean_df
            st.session_state["upload_success"] = True
            st.session_state["file_name"]      = uploaded_file.name
            st.session_state["rows_cleaned"]   = len(raw_df) - len(clean_df)
            st.session_state["cleaning_steps"] = steps_done
            st.session_state["cleaning_report"] = build_cleaning_report(raw_df, clean_df, date_col, sales_col)

        # ── show success + buttons whenever session has clean data ──
        if st.session_state.get("upload_success"):
            clean_df = st.session_state["uploaded_df"]
            removed  = st.session_state.get("rows_cleaned", 0)
            fname    = st.session_state.get("file_name", "your file")

            # success banner
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(52,211,153,0.08),rgba(16,185,129,0.04));
                        border:1px solid rgba(52,211,153,0.25);border-radius:12px;
                        padding:18px 22px;margin:16px 0;display:flex;align-items:center;gap:16px;">
              <div style="font-size:1.8rem;">🎉</div>
              <div>
                <div style="font-size:0.95rem;font-weight:700;color:#34d399;margin-bottom:3px;">
                  Data Ready for Analysis!
                </div>
                <div style="font-size:0.78rem;color:#475569;">
                  <b style="color:#94a3b8;">{fname}</b> · {len(clean_df):,} clean rows ·
                  {removed} rows removed during cleaning
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            render_cleaning_report(st.session_state.get("cleaning_report"))

            # cleaned preview
            st.markdown('<div class="section-label">🧹 Cleaned Data Preview</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="section-desc">
            This is what the Analysis and Forecasting modules will use. Duplicates removed,
            missing values filled, negative sales clipped to zero, columns standardised.
            </div>""", unsafe_allow_html=True)
            st.dataframe(clean_df.head(10), width="stretch", height=260)

            # stats
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Clean Rows",   f"{len(clean_df):,}")
            s2.metric("Columns",      f"{clean_df.shape[1]}")
            render_date_range_card(s3, clean_df)
            s4.metric("Total Revenue", f"₹{clean_df['Sales'].sum():,.0f}"
                      if "Sales" in clean_df.columns else "—")

            st.download_button(
                label="Download Cleaned Dataset CSV",
                data=clean_df.to_csv(index=False).encode("utf-8"),
                file_name=f"cleaned_{fname.replace(' ', '_')}.csv",
                mime="text/csv",
                width="stretch",
            )

            # Go to Analysis CTA
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style="background:linear-gradient(135deg,rgba(56,189,248,0.06),rgba(14,165,233,0.03));
                        border:1px solid rgba(56,189,248,0.2);border-radius:14px;
                        padding:20px 24px;display:flex;align-items:center;
                        justify-content:space-between;gap:20px;margin-bottom:16px;">
              <div>
                <div style="font-size:0.92rem;font-weight:600;color:#e2e8f0;margin-bottom:4px;">
                  ✅ Your data is clean and ready!
                </div>
                <div style="font-size:0.78rem;color:#475569;">
                  Head over to the Analysis page to explore charts, trends and insights.
                </div>
              </div>
              <div style="font-size:1.6rem;">📊</div>
            </div>
            """, unsafe_allow_html=True)

            b1, b2, _ = st.columns([1.2, 1.4, 2])
            with b1:
                render_navigation_link(
                    "pages/2_Analysis.py",
                    "/2_Analysis",
                    "📊 Go to Analysis →",
                    "Open the analysis dashboard for the cleaned dataset.",
                )
            with b2:
                if st.button("🔄 Upload Another File", width="stretch"):
                    for key in ["uploaded_df", "upload_success", "file_name", "rows_cleaned", "cleaning_steps", "cleaning_report"]:
                        st.session_state.pop(key, None)
                    st.rerun()

else:
    # ── truly no file, no session ──────────
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;">
      <div style="font-size:3rem;margin-bottom:12px;">📂</div>
      <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-bottom:8px;">
        No file uploaded yet
      </div>
      <div style="font-size:0.82rem;color:#475569;">
        Upload a CSV or Excel file above, or download the sample file to get started.
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── footer ────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:40px;padding:16px;border-top:1px solid #1f2937;
            color:#334155;font-size:0.72rem;font-family:'Inter',sans-serif;">
  AI Sales Forecasting System · Data Upload Module · CSV & Excel supported
</div>
""", unsafe_allow_html=True)
