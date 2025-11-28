"""Trend feature group computations."""

from __future__ import annotations

import numpy as np
import pandas as pd


class TrendFeatures:
    """Compute trend-related primitives."""

    def __init__(
        self,
        slope_window: int = 20,
        strength_window: int = 20,
        htf_short: int = 20,
        htf_long: int = 60,
    ) -> None:
        self.slope_window = slope_window
        self.strength_window = strength_window
        self.htf_short = htf_short
        self.htf_long = htf_long

    @staticmethod
    def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
        if window < 2:
            return pd.Series(np.nan, index=series.index)
        idx = np.arange(window, dtype=float)
        x_mean = idx.mean()
        denom = ((idx - x_mean) ** 2).sum()

        def slope(values: np.ndarray) -> float:
            if np.isnan(values).any():
                return np.nan
            y_mean = values.mean()
            num = ((idx - x_mean) * (values - y_mean)).sum()
            return num / denom if denom != 0 else np.nan

        return series.rolling(window=window, min_periods=window).apply(
            slope, raw=True
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        close = out["close"]

        slope = self._rolling_slope(close, self.slope_window)
        out["feature_trend_slope"] = slope

        rolling_std = close.rolling(self.strength_window, min_periods=self.strength_window).std(ddof=0)
        out["feature_trend_strength"] = slope / (rolling_std + 1e-9)

        short_ma = close.rolling(self.htf_short, min_periods=self.htf_short).mean()
        long_ma = close.rolling(self.htf_long, min_periods=self.htf_long).mean()
        out["feature_trend_htf_alignment"] = np.sign(short_ma - long_ma)

        return out
