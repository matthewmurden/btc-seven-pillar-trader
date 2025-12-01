"""
Rolling statistical utilities for regime feature construction.
Supports volatility, variance ratios, ATR windows, entropy windows,
microstructure metrics, and multi-horizon rolling statistics.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def rolling_vol(series: pd.Series, window: int) -> pd.Series:
    return series.pct_change().rolling(window).std().fillna(0.0)


def rolling_variance_ratio(series: pd.Series, w_small: int, w_large: int) -> pd.Series:
    var_small = series.pct_change().rolling(w_small).var()
    var_large = series.pct_change().rolling(w_large).var()
    return (var_small / var_large).replace([np.inf, -np.inf], np.nan).fillna(1.0)


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window)
    slopes = (
        series.rolling(window)
        .apply(lambda y: np.polyfit(x, y, 1)[0] if y.size == window else 0.0, raw=True)
        .fillna(0.0)
    )
    return slopes


def entropy(series: pd.Series, window: int = 20) -> pd.Series:
    def _ent(vals):
        hist, _ = np.histogram(vals, bins=8)
        p = hist / (hist.sum() + 1e-9)
        return -(p * np.log(p + 1e-9)).sum()

    return series.rolling(window).apply(_ent, raw=False).fillna(0.0)


def hurst(series: pd.Series, window: int = 80) -> pd.Series:
    """
    Fast, stable rolling Hurst exponent estimator.
    Uses log–log regression of variances.
    """
    series = series.astype(float)

    def _hurst_window(x: np.ndarray) -> float:
        try:
            lags = range(2, 20)
            tau = [np.sqrt(np.std(np.subtract(x[lag:], x[:-lag]))) for lag in lags]
            tau = np.array(tau)

            if np.any(tau <= 0):
                return 0.5

            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst_exp = poly[0] * 2.0

            if np.isnan(hurst_exp) or np.isinf(hurst_exp):
                return 0.5

            return float(np.clip(hurst_exp, 0.0, 1.0))
        except Exception:
            return 0.5

    return (
        series.rolling(window)
        .apply(lambda x: _hurst_window(x.values), raw=False)
        .fillna(0.5)
    )