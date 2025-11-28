"""Volatility mini engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


class VolatilityEngine:
    """Compute volatility_score from volatility-related primitives."""

    def __init__(self, slope_window: int = 20) -> None:
        self.slope_window = slope_window

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

        realized = out.get("feature_volatility_realized", pd.Series(np.nan, index=out.index))
        realized_slope = self._rolling_slope(realized.ffill(), self.slope_window)

        components = [
            out.get("feature_volatility_atr", pd.Series(0, index=out.index)),
            out.get("feature_volatility_atr_expansion", pd.Series(0, index=out.index)),
            realized,
            out.get("feature_volatility_percentile", pd.Series(0, index=out.index)) / 100.0,
            realized_slope,
        ]

        combined = sum(comp.fillna(0) for comp in components) / len(components)
        out["volatility_score"] = 1 / (1 + np.exp(-combined))
        return out
