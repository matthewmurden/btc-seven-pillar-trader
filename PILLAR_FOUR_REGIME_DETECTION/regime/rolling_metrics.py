"""
Rolling statistical utilities for regime feature construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling linear regression slope using closed-form solution for x=[0..w-1].
    No future leakage; uses trailing window only.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    x = np.arange(window, dtype=np.float64)
    sum_x = x.sum()
    sum_x2 = (x * x).sum()
    denom = window * sum_x2 - sum_x ** 2

    def _calc(arr: np.ndarray) -> float:
        y = arr.astype(np.float64)
        sum_y = y.sum()
        sum_xy = (y * x).sum()
        num = window * sum_xy - sum_x * sum_y
        return num / denom if denom != 0 else 0.0

    return series.rolling(window, min_periods=window).apply(_calc, raw=True)


def rolling_lr_gradient(series: pd.Series, window: int) -> pd.Series:
    """Alias for rolling_slope."""
    return rolling_slope(series, window)


def rolling_volatility(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).std(ddof=0)


def rolling_variance(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).var(ddof=0)


def rolling_entropy(direction_series: pd.Series, window: int) -> pd.Series:
    """
    Shannon entropy over sign of direction_series.
    """
    def _ent(arr: np.ndarray) -> float:
        signs = np.sign(arr)
        _, counts = np.unique(signs, return_counts=True)
        probs = counts / counts.sum()
        return float(-np.sum(probs * np.log2(probs + 1e-12)))

    return direction_series.rolling(window, min_periods=1).apply(_ent, raw=True)


def rolling_chop_index(direction_series: pd.Series, window: int) -> pd.Series:
    """
    Measures alternation in direction: mean absolute diff of sign changes.
    """
    sign = np.sign(direction_series).fillna(0)
    return sign.diff().abs().rolling(window, min_periods=1).mean().fillna(0)


def rolling_range_compression(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    rng = high - low
    return rng / (rng.rolling(window, min_periods=1).mean() + 1e-12)


def rolling_hurst_exponent(series: pd.Series, window: int) -> pd.Series:
    """
    RS-method approximation for Hurst exponent on rolling windows.
    """
    def _hurst(arr: np.ndarray) -> float:
        if len(arr) < 8:
            return 0.5
        mean = arr.mean()
        dev = arr - mean
        cum = np.cumsum(dev)
        R = cum.max() - cum.min()
        S = dev.std(ddof=0) + 1e-12
        return float(np.log(R / S + 1e-12) / np.log(len(arr)))

    return series.rolling(window, min_periods=window).apply(_hurst, raw=True)
