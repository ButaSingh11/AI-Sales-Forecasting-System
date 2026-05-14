from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data
def generate_sample_sales_data(include_segments: bool = True) -> pd.DataFrame:
    """Return a consistent demo sales dataset for pages without uploaded data."""
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    n = len(dates)

    trend = np.linspace(8000, 22000, n)
    seasonality = 3500 * np.sin(2 * np.pi * dates.dayofyear / 365)
    weekly = 800 * np.sin(2 * np.pi * dates.dayofweek / 7)
    noise = np.random.normal(0, 600, n)
    sales = np.maximum(trend + seasonality + weekly + noise, 0).round(2)

    data = {"Date": dates, "Sales": sales}
    if include_segments:
        data["Category"] = np.random.choice(
            ["Electronics", "Apparel", "Home & Garden", "Sports", "Beauty"],
            n,
            p=[0.3, 0.25, 0.2, 0.15, 0.1],
        )
        data["Region"] = np.random.choice(
            ["North", "South", "East", "West"],
            n,
            p=[0.3, 0.25, 0.25, 0.2],
        )
        data["Orders"] = (sales / np.random.uniform(45, 85, n)).astype(int)
        data["Returns"] = np.random.randint(0, 30, n)

    return pd.DataFrame(data)


def repair_text_encoding(value):
    """Repair common mojibake without harming already-clean values."""
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return text

    repaired = text
    mojibake_markers = ("\u00c3", "\u00e2", "\u00f0", "\u00c2")
    if any(marker in repaired for marker in mojibake_markers):
        for source_encoding in ("cp1252", "latin1"):
            try:
                repaired = repaired.encode(source_encoding).decode("utf-8")
                break
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue

    replacements = {
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\xa0": " ",
    }
    for old, new in replacements.items():
        repaired = repaired.replace(old, new)

    repaired = unicodedata.normalize("NFKC", repaired)
    return " ".join(repaired.split())


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize text columns so labels render cleanly across the app."""
    normalized_df = df.copy()
    text_columns = normalized_df.select_dtypes(include=["object", "string", "category"]).columns
    for col in text_columns:
        normalized_df[col] = normalized_df[col].apply(repair_text_encoding)
    return normalized_df


def load_sales_data(include_segments: bool = True, copy_uploaded: bool = True) -> tuple[pd.DataFrame, bool]:
    """Load uploaded sales data, or fall back to consistent sample data."""
    uploaded_df = st.session_state.get("uploaded_df")
    if uploaded_df is not None:
        return (uploaded_df.copy() if copy_uploaded else uploaded_df), False
    return normalize_text_columns(generate_sample_sales_data(include_segments=include_segments)), True


def format_inr(value: float) -> str:
    """Format Indian currency compactly with ASCII-safe text."""
    if abs(value) >= 1e7:
        return f"Rs {value/1e7:.2f}Cr"
    if abs(value) >= 1e5:
        return f"Rs {value/1e5:.2f}L"
    if abs(value) >= 1e3:
        return f"Rs {value/1e3:.1f}K"
    return f"Rs {value:,.0f}"


def render_navigation_link(page: str,
                           fallback_url: str,
                           label: str,
                           help_text: str | None = None) -> None:
    """Render a multipage navigation control with a resilient URL fallback."""
    try:
        st.page_link(
            page,
            label=label,
            use_container_width=True,
            help=help_text,
        )
    except Exception:
        st.link_button(
            label,
            fallback_url,
            use_container_width=True,
            help=help_text,
        )


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_value = hex_color.lstrip("#")
    red, green, blue = (
        int(hex_value[0:2], 16),
        int(hex_value[2:4], 16),
        int(hex_value[4:6], 16),
    )
    return f"rgba({red},{green},{blue},{alpha})"
