"""
trend_engine.py
Quantitative trend detection using hurst, slope, and structural bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine_utils import get_series, normalize


def _rolling_slope(close: pd.Series, window: int = 20, min_periods: int = 5) -> pd.Series:
    """
    Fast rolling slope approximation: mean of diff over window.
    Keeps it vectorised and stable for high-frequency data.
    """
    return close.diff().rolling(window=window, min_periods=min_periods).mean()


def compute_trend_engine(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    hurst = get_series(df, "hurst_exponent", default=0.36).clip(lower=0, upper=1)

    slope_raw = None
    for col in ("trend_strength", "slope"):
        if col in df.columns:
            slope_raw = df[col].astype(float)
            break

    if slope_raw is None:
        price_col = "close" if "close" in df.columns else ("price" if "price" in df.columns else None)
        if price_col:
            slope_raw = _rolling_slope(df[price_col].astype(float))
        else:
            slope_raw = pd.Series(0.0, index=df.index)

    slope_norm = normalize(slope_raw).fillna(0.0)

    htf_pos = get_series(df, "htf_range_position", default=0.0)

    structure = pd.Series(0.0, index=df.index)
    price_col = "close" if "close" in df.columns else ("price" if "price" in df.columns else None)
    if price_col:
        rolling_mean = df[price_col].astype(float).rolling(window=50, min_periods=10).mean()
        structure = (df[price_col].astype(float) > rolling_mean).astype(float).fillna(0.0)

    raw_strength = 0.7 * slope_norm + 0.2 * hurst + 0.1 * htf_pos + 0.1 * structure
    trend_strength = normalize(raw_strength).clip(-1, 1).fillna(0.0)

    trend_flag = (
        (trend_strength > 0.3)
        | ((hurst > 0.55) & (slope_norm > 0.1))
        | (slope_norm > 0.4)
    ).astype(int)
    trend_direction = np.sign(slope_norm).clip(-1, 1).astype(int)

    out["trend_strength_score"] = trend_strength
    out["trend_flag"] = trend_flag
    out["trend_direction"] = trend_direction
    return out
