"""
Utility functions for computing raw regime features:
trend, volatility, liquidity, microstructure, and range context.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .rolling_metrics import (
    rolling_slope,
    rolling_mean,
    rolling_std,
    rolling_zscore,
    rolling_min,
    rolling_max,
    hurst_exponent,
)


# -------------------------
# Trend Measures
# -------------------------

def compute_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]

    df["slope_2"] = rolling_slope(close, 2)
    df["slope_5"] = rolling_slope(close, 5)
    df["slope_15"] = rolling_slope(close, 15)

    df["lr_grad_fast"] = df["slope_5"] / (rolling_std(close, 20).replace(0, np.nan))

    df["trend_strength"] = rolling_mean(df["slope_5"].abs(), 20)

    # HTF alignment approximations
    df["htf_trend_5m"] = rolling_mean(df["slope_5"], 5 * 5)
    df["htf_trend_15m"] = rolling_mean(df["slope_5"], 15 * 5)
    df["htf_trend_1h"] = rolling_mean(df["slope_5"], 60 * 5)

    return df


# -------------------------
# Volatility Features
# -------------------------

def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]

    df["realized_vol_1"] = rolling_std(close, 1)
    df["realized_vol_5"] = rolling_std(close, 5)
    df["realized_vol_15"] = rolling_std(close, 15)
    df["realized_vol_30"] = rolling_std(close, 30)

    df["var_ratio_5_20"] = df["realized_vol_5"] / (rolling_std(close, 20).replace(0, np.nan))
    df["var_ratio_20_50"] = (
        rolling_std(close, 20) / rolling_std(close, 50).replace(0, np.nan)
    )

    df["vol_compression_score"] = rolling_zscore(df["realized_vol_5"], 50)
    df["vol_expansion_score"] = rolling_zscore(df["realized_vol_1"], 20)

    df["atr_percentile"] = rolling_zscore(df["realized_vol_15"], 200)
    df["vol_clustering"] = rolling_zscore(df["realized_vol_5"], 30)

    df["vol_breakout_flag"] = (df["realized_vol_1"] > df["realized_vol_15"] * 1.5).astype(int)

    return df


# -------------------------
# Liquidity/Microstructure
# -------------------------

def compute_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    # Proxy features (pure OHLCV only)
    vol = df["volume"]

    df["delta_intensity"] = rolling_zscore(vol.diff().abs(), 20)
    df["imbalance_persistence"] = rolling_mean(vol.diff(), 10)
    df["cvd_slope"] = rolling_slope(vol.cumsum(), 20)

    df["volume_expansion_5"] = rolling_mean(vol, 5)
    df["volume_expansion_20"] = rolling_mean(vol, 20)

    df["liquidity_drought_score"] = (vol < rolling_min(vol, 100)).astype(int)

    return df


# -------------------------
# Microstructure Features
# -------------------------

def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    df["micro_trend_entropy"] = rolling_std((c - o).abs(), 20)

    df["bar_clustering_score"] = rolling_mean((c > o).astype(int).diff().abs(), 20)

    df["chop_index"] = rolling_mean((c.diff().abs() < (h - l)).astype(int), 20)

    df["upper_wick_body"] = (h - np.maximum(c, o)) / (h - l + 1e-9)
    df["lower_wick_body"] = (np.minimum(c, o) - l) / (h - l + 1e-9)

    df["noise_to_signal"] = rolling_std(c.diff(), 20) / (np.abs(rolling_mean(c.diff(), 20)) + 1e-9)

    df["hurst_exponent"] = hurst_exponent(c)

    return df


# -------------------------
# Structure / Range Features
# -------------------------

def compute_range_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]

    df["distance_to_last_swing_high"] = close - rolling_max(close, 20)
    df["distance_to_last_swing_low"] = close - rolling_min(close, 20)

    df["range_compression"] = rolling_zscore(hurst_exponent(close), 50)

    df["htf_range_position"] = (close - rolling_min(close, 200)) / (
        rolling_max(close, 200) - rolling_min(close, 200) + 1e-9
    )

    df["sweep_flags"] = (close < rolling_min(close, 50)).astype(int)

    df["mean_reversion_distance"] = rolling_zscore(close.diff().abs(), 20)

    df["range_tightness_score"] = rolling_std(close, 30)

    return df