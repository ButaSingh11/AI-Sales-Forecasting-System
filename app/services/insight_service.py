
import pandas as pd
import numpy as np
from typing import List, Tuple


def generate_insights(df: pd.DataFrame) -> List[Tuple[str, str, str, str]]:
    """
    Generate a list of auto insights from the sales dataframe.
    Returns list of (icon, title, detail, level) tuples.
    level: 'positive' | 'warning' | 'neutral'
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    insights: List[Tuple[str, str, str, str]] = []

    # Monthly aggregation
    df["Month"] = df["Date"].dt.to_period("M")
    monthly = df.groupby("Month")["Sales"].sum()

    if len(monthly) < 2:
        return [(
            "ℹ️",
            "Not enough data",
            "Upload at least 2 months of data to generate insights.",
            "neutral",
        )]

    # 1. Overall trend
    if len(monthly) >= 6:
        first_half = monthly.iloc[:len(monthly) // 2].mean()
        second_half = monthly.iloc[len(monthly) // 2:].mean()
        growth = (second_half - first_half) / first_half * 100 if first_half else 0
        if growth > 10:
            insights.append((
                "📈",
                "Strong Upward Trend",
                f"Revenue grew {growth:.1f}% comparing first half to second half of data.",
                "positive",
            ))
        elif growth < -10:
            insights.append((
                "📉",
                "Declining Revenue",
                f"Revenue fell {abs(growth):.1f}% comparing first half to second half. Investigate root causes.",
                "warning",
            ))
        else:
            insights.append((
                "➡️",
                "Stable Revenue",
                f"Revenue is relatively flat ({growth:+.1f}% trend).",
                "neutral",
            ))

    # 2. Best and worst month
    best_month = monthly.idxmax()
    worst_month = monthly.idxmin()
    insights.append((
        "🏆",
        f"Best Month: {best_month}",
        f"Peak revenue of ₹{monthly.max():,.0f} — consider replicating what worked.",
        "positive",
    ))
    insights.append((
        "⚠️",
        f"Weakest Month: {worst_month}",
        f"Lowest revenue at ₹{monthly.min():,.0f} — a candidate for targeted promotions.",
        "warning",
    ))

    # 3. Recent momentum
    if len(monthly) >= 4:
        last_3 = monthly.iloc[-3:].mean()
        prev_3 = monthly.iloc[-6:-3].mean() if len(monthly) >= 6 else monthly.iloc[:-3].mean()
        mom = (last_3 - prev_3) / prev_3 * 100 if prev_3 else 0
        level = "positive" if mom > 5 else ("warning" if mom < -5 else "neutral")
        emoji = "🚀" if mom > 5 else ("🔻" if mom < -5 else "💤")
        insights.append((
            emoji,
            f"Recent Momentum: {mom:+.1f}%",
            f"Last 3 months avg ₹{last_3:,.0f} vs prior comparison period ₹{prev_3:,.0f}.",
            level,
        ))

    # 4. Category insight
    if "Category" in df.columns:
        cat_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
        top_cat = cat_sales.index[0]
        top_pct = cat_sales.iloc[0] / cat_sales.sum() * 100
        if top_pct > 50:
            insights.append((
                "⚠️",
                "Category Concentration Risk",
                f"{top_cat} drives {top_pct:.0f}% of revenue. High dependence increases business risk.",
                "warning",
            ))
        else:
            insights.append((
                "🏷️",
                f"Top Category: {top_cat}",
                f"Contributes {top_pct:.0f}% of total revenue.",
                "positive",
            ))

    # 5. Region insight
    if "Region" in df.columns:
        reg_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=True)
        weak_reg = reg_sales.index[0]
        weak_pct = reg_sales.iloc[0] / reg_sales.sum() * 100
        insights.append((
            "🌍",
            f"Underperforming Region: {weak_reg}",
            f"Only {weak_pct:.0f}% of revenue — potential growth opportunity.",
            "warning" if weak_pct < 15 else "neutral",
        ))

    # 6. Volatility
    cv = monthly.std() / monthly.mean() * 100 if monthly.mean() > 0 else 0
    if cv > 30:
        insights.append((
            "📊",
            "High Sales Volatility",
            f"Month-to-month variation is {cv:.0f}% (coefficient of variation). High volatility makes forecasting harder.",
            "warning",
        ))
    else:
        insights.append((
            "✅",
            "Consistent Sales Pattern",
            f"Revenue volatility is {cv:.0f}% — relatively stable and predictable.",
            "positive",
        ))

    return insights


def generate_alerts(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Generate urgent business alerts.
    Returns list of (alert_text, level) tuples.
    level: 'danger' | 'warning' | 'info'
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    alerts: List[Tuple[str, str]] = []

    df["Month"] = df["Date"].dt.to_period("M")
    monthly = df.groupby("Month")["Sales"].sum()

    if len(monthly) >= 3:
        last_month = monthly.iloc[-1]
        prev_month = monthly.iloc[-2]
        drop_pct = (prev_month - last_month) / prev_month * 100 if prev_month else 0
        if drop_pct > 20:
            alerts.append((
                f"🚨 Revenue dropped {drop_pct:.0f}% last month vs previous month.",
                "danger",
            ))
        elif drop_pct > 10:
            alerts.append((
                f"⚠️ Revenue declined {drop_pct:.0f}% month-over-month.",
                "warning",
            ))

    if len(monthly) >= 3:
        last3 = monthly.iloc[-3:].values
        if last3[0] > last3[1] > last3[2]:
            alerts.append((
                "📉 3 consecutive months of declining revenue detected.",
                "warning",
            ))

    return alerts


def detect_anomalies(df: pd.DataFrame,
                     method: str = "zscore",
                     threshold: float = 2.5,
                     rolling_window: int = 6) -> pd.DataFrame:
    """
    Detect anomalies in sales data.

    Supported methods:
    - zscore
    - iqr
    - rolling_deviation

    Returns a DataFrame with:
    Date, Sales, anomaly_score, severity, method
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if "Sales" not in df.columns or len(df) < 3:
        return pd.DataFrame(columns=["Date", "Sales", "anomaly_score", "severity", "method"])

    results = pd.DataFrame(columns=["Date", "Sales", "anomaly_score", "severity", "method"])

    if method == "zscore":
        mean_val = df["Sales"].mean()
        std_val = df["Sales"].std()
        if std_val == 0 or pd.isna(std_val):
            return results
        z_scores = (df["Sales"] - mean_val) / std_val
        flagged = df.loc[z_scores.abs() > threshold, ["Date", "Sales"]].copy()
        flagged["anomaly_score"] = z_scores.loc[flagged.index].abs().round(2)
        flagged["severity"] = np.where(
            flagged["anomaly_score"] >= 4, "high",
            np.where(flagged["anomaly_score"] >= 3, "medium", "low")
        )
        flagged["method"] = "zscore"
        results = flagged

    elif method == "iqr":
        q1 = df["Sales"].quantile(0.25)
        q3 = df["Sales"].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or pd.isna(iqr):
            return results
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flagged = df.loc[(df["Sales"] < lower) | (df["Sales"] > upper), ["Date", "Sales"]].copy()
        flagged["anomaly_score"] = (
            np.maximum((flagged["Sales"] - upper).abs(), (lower - flagged["Sales"]).abs()) / max(iqr, 1)
        ).round(2)
        flagged["severity"] = np.where(
            flagged["anomaly_score"] >= 3, "high",
            np.where(flagged["anomaly_score"] >= 2, "medium", "low")
        )
        flagged["method"] = "iqr"
        results = flagged

    elif method == "rolling_deviation":
        rolling_mean = df["Sales"].rolling(rolling_window, min_periods=3).mean()
        rolling_std = df["Sales"].rolling(rolling_window, min_periods=3).std()
        deviation = (df["Sales"] - rolling_mean).abs()
        ratio = deviation / rolling_std.replace(0, np.nan)
        flagged = df.loc[ratio > threshold, ["Date", "Sales"]].copy()
        flagged["anomaly_score"] = ratio.loc[flagged.index].fillna(0).round(2)
        flagged["severity"] = np.where(
            flagged["anomaly_score"] >= 4, "high",
            np.where(flagged["anomaly_score"] >= 3, "medium", "low")
        )
        flagged["method"] = "rolling_deviation"
        results = flagged

    else:
        raise ValueError("Unknown anomaly detection method. Use: zscore, iqr, or rolling_deviation.")

    return results.reset_index(drop=True)


def generate_root_cause_insights(df: pd.DataFrame) -> List[str]:
    """
    Generate simple root-cause style business insights.

    Examples:
    - Sales declined mainly due to South region
    - Apparel contributed most to the recent drop
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    insights: List[str] = []

    if len(df) < 30:
        return ["Not enough data for root-cause style analysis."]

    df["Month"] = df["Date"].dt.to_period("M")
    monthly = df.groupby("Month")["Sales"].sum()

    if len(monthly) >= 2:
        last_month = monthly.iloc[-1]
        prev_month = monthly.iloc[-2]
        delta = last_month - prev_month

        if delta < 0:
            if "Region" in df.columns:
                recent = df[df["Month"].isin([monthly.index[-2], monthly.index[-1]])].copy()
                region_pivot = recent.pivot_table(
                    index="Month", columns="Region", values="Sales", aggfunc="sum"
                ).fillna(0)
                if len(region_pivot) >= 2:
                    region_change = region_pivot.iloc[-1] - region_pivot.iloc[-2]
                    weakest_region = region_change.idxmin()
                    insights.append(
                        f"Sales declined mainly due to weaker performance in {weakest_region} region."
                    )

            if "Category" in df.columns:
                recent = df[df["Month"].isin([monthly.index[-2], monthly.index[-1]])].copy()
                cat_pivot = recent.pivot_table(
                    index="Month", columns="Category", values="Sales", aggfunc="sum"
                ).fillna(0)
                if len(cat_pivot) >= 2:
                    cat_change = cat_pivot.iloc[-1] - cat_pivot.iloc[-2]
                    weakest_cat = cat_change.idxmin()
                    insights.append(
                        f"{weakest_cat} contributed most to the recent revenue drop."
                    )

        elif delta > 0:
            if "Region" in df.columns:
                recent = df[df["Month"].isin([monthly.index[-2], monthly.index[-1]])].copy()
                region_pivot = recent.pivot_table(
                    index="Month", columns="Region", values="Sales", aggfunc="sum"
                ).fillna(0)
                if len(region_pivot) >= 2:
                    region_change = region_pivot.iloc[-1] - region_pivot.iloc[-2]
                    strongest_region = region_change.idxmax()
                    insights.append(
                        f"Recent growth was led mainly by stronger performance in {strongest_region} region."
                    )

            if "Category" in df.columns:
                recent = df[df["Month"].isin([monthly.index[-2], monthly.index[-1]])].copy()
                cat_pivot = recent.pivot_table(
                    index="Month", columns="Category", values="Sales", aggfunc="sum"
                ).fillna(0)
                if len(cat_pivot) >= 2:
                    cat_change = cat_pivot.iloc[-1] - cat_pivot.iloc[-2]
                    strongest_cat = cat_change.idxmax()
                    insights.append(
                        f"{strongest_cat} contributed most to the recent growth."
                    )

    anomaly_df = detect_anomalies(df, method="zscore", threshold=2.5)
    if not anomaly_df.empty:
        top_anomaly = anomaly_df.sort_values("anomaly_score", ascending=False).iloc[0]
        insights.append(
            f"An unusual sales spike/drop was detected on {top_anomaly['Date'].strftime('%d %b %Y')} "
            f"with {top_anomaly['severity']} severity."
        )

    if not insights:
        insights.append("No strong root-cause signal detected; sales movement appears broadly distributed.")

    return insights


def generate_recommendations(df: pd.DataFrame) -> List[str]:
    """
    Generate stronger recommendation logic based on trend, volatility, anomalies,
    and concentration patterns.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    recommendations: List[str] = []

    df["Month"] = df["Date"].dt.to_period("M")
    monthly = df.groupby("Month")["Sales"].sum()

    if len(monthly) >= 3:
        last_3 = monthly.iloc[-3:].mean()
        prev_3 = monthly.iloc[-6:-3].mean() if len(monthly) >= 6 else monthly.iloc[:-3].mean()
        if prev_3 > 0:
            change_pct = (last_3 - prev_3) / prev_3 * 100
            if change_pct < -5:
                recommendations.append("Consider targeted promotions or retention campaigns to reverse the recent slowdown.")
            elif change_pct > 5:
                recommendations.append("Demand is strengthening — consider increasing inventory and marketing support.")

    anomaly_df = detect_anomalies(df, method="rolling_deviation", threshold=2.5)
    if len(anomaly_df) >= 3:
        recommendations.append("Multiple anomalies were detected. Review events, promotions, stock-outs, or data quality issues.")

    if "Category" in df.columns:
        cat_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
        top_pct = cat_sales.iloc[0] / cat_sales.sum() * 100 if cat_sales.sum() > 0 else 0
        if top_pct > 50:
            recommendations.append("Revenue is highly concentrated in one category. Diversify category performance to reduce business risk.")

    if "Region" in df.columns:
        reg_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=True)
        weak_pct = reg_sales.iloc[0] / reg_sales.sum() * 100 if reg_sales.sum() > 0 else 0
        if weak_pct < 15:
            recommendations.append(f"Focus growth efforts on the underperforming region: {reg_sales.index[0]}.")

    volatility = monthly.std() / monthly.mean() * 100 if monthly.mean() > 0 else 0
    if volatility > 30:
        recommendations.append("High volatility reduces forecast reliability. Consider separating seasonal and campaign-driven effects.")

    if not recommendations:
        recommendations.append("Sales patterns look relatively stable. Continue monitoring trends and test incremental optimizations.")

    return recommendations
