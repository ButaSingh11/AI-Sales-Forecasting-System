import numpy as np
import pandas as pd


def apply_levers(forecast_df: pd.DataFrame,
                 growth_pct: float        = 0,
                 seasonality_boost: float = 0,
                 discount_impact: float   = 0,
                 marketing_boost: float   = 0,
                 churn_impact: float      = 0,
                 new_channel_pct: float   = 0) -> np.ndarray:
    """
    Apply business levers to a base forecast array.
    Returns adjusted forecast values as a numpy array.

    Parameters
    ----------
    forecast_df      : DataFrame with 'Forecast' column (base forecast)
    growth_pct       : Annual growth rate % (e.g. 10 = 10% annual growth)
    seasonality_boost: Amplify/dampen seasonal pattern % (-20 to +20)
    discount_impact  : Revenue reduction from discounts % (≤ 0)
    marketing_boost  : Revenue uplift from marketing % (ramps over 3 months)
    churn_impact     : Revenue loss from customer churn % (gradual)
    new_channel_pct  : Flat revenue uplift from new channel %
    """
    f = forecast_df["Forecast"].values.copy().astype(float)
    n = len(f)

    # 1. Compound monthly growth
    monthly_growth = (1 + growth_pct / 100) ** (1 / 12) - 1
    growth_mult    = np.array([(1 + monthly_growth) ** (i + 1) for i in range(n)])
    f *= growth_mult

    # 2. Seasonality overlay (sine wave — peaks mid-period)
    if seasonality_boost != 0:
        wave = np.sin(np.linspace(0, 2 * np.pi, n))
        f   += f * (seasonality_boost / 100) * wave

    # 3. Discount — flat % reduction
    f *= (1 + discount_impact / 100)

    # 4. Marketing — ramps up over first 3 months
    if marketing_boost > 0:
        ramp = np.minimum(np.arange(1, n + 1) / 3.0, 1.0)
        f   *= (1 + (marketing_boost / 100) * ramp)

    # 5. Churn — gradual revenue erosion
    if churn_impact > 0:
        churn_curve = np.array([1 - (churn_impact / 100) * (i / n) for i in range(n)])
        f          *= np.maximum(churn_curve, 0.5)   # floor at 50% loss

    # 6. New channel — flat uplift
    f *= (1 + new_channel_pct / 100)

    return np.maximum(f, 0)


def build_preset_scenarios(forecast_df: pd.DataFrame) -> dict:
    """Return Best / Base / Worst scenario arrays."""
    return {
        "best" : apply_levers(forecast_df, 20,  10,   0, 25,  0, 15),
        "base" : apply_levers(forecast_df,  8,   0,  -5, 10,  3,  0),
        "worst": apply_levers(forecast_df, -5, -10, -15,  0, 15,  0),
    }


def lever_impact_breakdown(forecast_df: pd.DataFrame,
                           growth_pct: float,
                           seasonality_boost: float,
                           discount_impact: float,
                           marketing_boost: float,
                           churn_impact: float,
                           new_channel_pct: float) -> dict:
    """
    Compute the individual revenue impact of each lever.
    Returns dict of {lever_name: revenue_delta}.
    """
    base_total = forecast_df["Forecast"].sum()

    def delta(g=0, s=0, d=0, m=0, c=0, ch=0):
        return apply_levers(forecast_df, g, s, d, m, c, ch).sum() - base_total

    return {
        "Growth Rate"  : delta(g=growth_pct),
        "Seasonality"  : delta(s=seasonality_boost),
        "Discount"     : delta(d=discount_impact),
        "Marketing"    : delta(m=marketing_boost),
        "Churn"        : delta(c=churn_impact),
        "New Channel"  : delta(ch=new_channel_pct),
    }
