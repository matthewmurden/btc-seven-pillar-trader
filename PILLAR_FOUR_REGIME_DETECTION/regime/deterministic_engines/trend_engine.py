"""
Trend Engine
Detects structural trend states using slope, efficiency, HH/HL structure,
and HTF alignment from raw regime features.
"""

import pandas as pd
import numpy as np

def apply_trend_engine(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Trend strength: smoothed slope & ATR-adjusted slope
    slope = out["trend_slope"]
    efficiency = out["trend_strength"]
    atr_adj = out["atr_adjusted_trend"]

    # High Timeframe (HTF) alignment
    htf_score = (
        out["htf_trend_5m"]
        + out["htf_trend_15m"]
        + out["htf_trend_1h"]
    ) / 3.0

    # Structural trend components
    hh_score = out["higher_high_score"]
    hl_score = out["higher_low_score"]
    structure_score = (hh_score + hl_score) / 2

    # Trend strength score (0–1)
    trend_score = (
        0.4 * slope +
        0.3 * efficiency +
        0.2 * atr_adj +
        0.1 * structure_score
    )

    # Final trend flag
    trend_flag = (trend_score > 0.6).astype(int)

    out["trend_strength_score"] = trend_score.clip(0, 1)
    out["trend_flag"] = trend_flag

    return out