"""
High-performance rolling metrics used by raw regime feature construction.
Fully vectorised, leakage-proof, and safe for ML pipelines.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# -------------------------
# Safe Rolling Helpers
# -------------------------

def rolling_mean(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window, min_periods=window).mean()


def rolling_std(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window, min_periods=window).std(ddof=0)


def rolling_min(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window, min_periods=window).min()


def rolling_max(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window, min_periods=window).max()


def rolling_sum(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window, min_periods=window).sum()


def rolling_zscore(x: pd.Series, window: int) -> pd.Series:
    m = rolling_mean(x, window)
    s = rolling_std(x, window)
    return (x - m) / (s.replace(0, np.nan))


# -------------------------
# Price Structure Metrics
# -------------------------

def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """
    Linear regression slope over a rolling window.
    Fast implementation using closed-form OLS.
    """
    y = series.values
    n = window

    if len(y) < n:
        return pd.Series(np.zeros_like(y), index=series.index)

    x = np.arange(n)
    denom = n * np.sum(x * x) - np.sum(x) ** 2

    slopes = np.zeros(len(y))
    for i in range(n - 1, len(y)):
        y_window = y[i - n + 1 : i + 1]
        num = n * np.sum(x * y_window) - np.sum(x) * np.sum(y_window)
        slopes[i] = num / denom

    return pd.Series(slopes, index=series.index)


def hurst_exponent(price: pd.Series, window: int = 100) -> pd.Series:
    """
    Hurst exponent estimator (rough, but extremely useful).
    Measures trending (H > 0.5) vs mean-reversion (H < 0.5).
    """
    log_p = np.log(price.replace(0, np.nan).ffill())

    def calc_hurst(sub: np.ndarray) -> float:
        if len(sub) < 20:
            return np.nan
        tau = []
        lagvec = range(2, 20)
        for lag in lagvec:
            pp = np.subtract(sub[lag:], sub[:-lag])
            tau.append(np.sqrt(np.std(pp)))
        lagvec = np.array(list(lagvec))
        tau = np.array(tau)
        return np.polyfit(np.log(lagvec), np.log(tau), 1)[0]

    out = []
    for i in range(len(log_p)):
        if i < window:
            out.append(np.nan)
        else:
            out.append(calc_hurst(log_p.values[i - window : i]))

    return pd.Series(out, index=price.index).fillna(method="bfill").fillna(0.5)