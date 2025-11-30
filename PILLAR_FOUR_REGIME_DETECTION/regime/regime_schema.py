"""
Schema definitions and validation for raw regime features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    # Trend
    "slope_5",
    "slope_10",
    "slope_20",
    "slope_50",
    "ols_grad_20",
    "ols_grad_50",
    "trend_strength_20",
    "trend_strength_50",
    "atr_adjusted_trend_14",
    "htf_trend_5m",
    "htf_trend_15m",
    "htf_trend_1h",
    "impulse_correction_ratio",
    "higher_high_score",
    "higher_low_score",
    "bos_count_20",
    "bos_count_50",
    # Volatility
    "realized_vol_1",
    "realized_vol_5",
    "realized_vol_15",
    "realized_vol_30",
    "var_ratio_5_20",
    "var_ratio_20_50",
    "vol_compression_score",
    "vol_expansion_z",
    "atr_percentile_200",
    "vol_clustering",
    "vol_breakout_flag",
    # Liquidity
    "delta_intensity",
    "imbalance_persistence_20",
    "cvd_slope_20",
    "volume_expansion_5",
    "volume_expansion_20",
    "liquidity_drought_score",
    # Microstructure
    "micro_trend_entropy_20",
    "bar_clustering_20",
    "chop_index_20",
    "upper_wick_body",
    "lower_wick_body",
    "noise_to_signal",
    "hurst_50",
    # Structure / Range
    "distance_to_last_swing_high",
    "distance_to_last_swing_low",
    "range_compression_20",
    "htf_range_position_1h",
    "sweep_flag",
    "mean_reversion_distance_20",
    "range_tightness",
]

DTYPES = {col: np.float32 for col in REQUIRED_COLUMNS if col != "timestamp"}
DTYPES["timestamp"] = "datetime64[ns]"


def validate_regime_features(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required regime feature columns: {missing}")
    if df.isna().any().any():
        raise ValueError("NaNs detected in regime features.")
    if not np.isfinite(df.drop(columns=["timestamp"]).to_numpy()).all():
        raise ValueError("Non-finite values detected in regime features.")
