"""
Utility functions for computing raw regime features derived from OHLCV +
Pillar 1 enriched features.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from .rolling_metrics import (
    rolling_vol,
    rolling_variance_ratio,
    rolling_slope,
    entropy,
    hurst,
)


def compute_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df["trend_slope_20"] = rolling_slope(df["close"], 20)
    df["trend_slope_50"] = rolling_slope(df["close"], 50)
    df["trend_slope_volatility"] = df["trend_slope_20"].rolling(50).std().fillna(0.0)
    df["trend_convexity"] = df["trend_slope_20"] - df["trend_slope_50"]
    return df


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df["realized_vol_1"] = rolling_vol(df["close"], 1)
    df["realized_vol_5"] = rolling_vol(df["close"], 5)
    df["realized_vol_15"] = rolling_vol(df["close"], 15)
    df["realized_vol_30"] = rolling_vol(df["close"], 30)

    df["var_ratio_5_20"] = rolling_variance_ratio(df["close"], 5, 20)
    df["var_ratio_20_50"] = rolling_variance_ratio(df["close"], 20, 50)

    df["vol_clustering"] = df["realized_vol_5"].rolling(30).std().fillna(0.0)
    df["vol_compression_score"] = 1.0 / (df["realized_vol_5"] + 1e-9)
    df["vol_expansion_score"] = df["realized_vol_5"]

    return df


def compute_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    df["volume_expansion_5"] = df["volume"] / (df["volume"].rolling(5).mean() + 1e-9)
    df["volume_expansion_20"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-9)

    df["imbalance_persistence"] = df["delta"].rolling(10).mean().fillna(0.0)
    df["liquidity_drought_score"] = (
        (df["volume"] < df["volume"].rolling(50).mean()).astype(int)
    )

    return df


def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    df["chop_index"] = (
        (df["close"].diff().abs().rolling(20).sum()) /
        (df["close"].rolling(20).max() - df["close"].rolling(20).min() + 1e-9)
    )

    df["micro_trend_entropy"] = entropy(df["close"].diff(), 30)
    df["bar_clustering_score"] = (df["close"].diff().rolling(10).apply(
        lambda x: np.sum(np.sign(x) == np.sign(x.mean())) if x.mean() != 0 else 0,
        raw=True
    )).fillna(0.0)

    df["noise_to_signal"] = df["close"].diff().rolling(20).std() / (
        df["close"].rolling(20).mean() + 1e-9
    )

    df["hurst_exponent"] = hurst(df["close"], 80)

    return df


def compute_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    df["range_compression"] = (
        (df["high"] - df["low"]) /
        (df["close"].rolling(50).mean() + 1e-9)
    )

    df["mean_reversion_distance"] = (
        df["close"] - df["close"].rolling(50).mean()
    ).abs()

    df["htf_range_position"] = (
        (df["close"] - df["close"].rolling(150).min()) /
        (df["close"].rolling(150).max() - df["close"].rolling(150).min() + 1e-9)
    )

    return df