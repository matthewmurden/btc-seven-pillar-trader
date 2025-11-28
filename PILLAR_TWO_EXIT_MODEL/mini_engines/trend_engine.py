"""Trend mini engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


class TrendEngine:
    """Compute trend_score from trend-related primitives."""

    def __init__(
        self,
        efficiency_window: int = 20,
        ma_fast: int = 20,
        ma_slow: int = 50,
    ) -> None:
        self.efficiency_window = efficiency_window
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        close = out["close"]

        change = close.diff(self.efficiency_window).abs()
        volatility = close.diff().abs().rolling(self.efficiency_window, min_periods=2).sum()
        efficiency_ratio = change / (volatility + 1e-9)
        noise_ratio = 1 - efficiency_ratio

        ma_fast = close.rolling(self.ma_fast, min_periods=self.ma_fast).mean()
        ma_slow = close.rolling(self.ma_slow, min_periods=self.ma_slow).mean()
        ma_distance = (close - ma_slow) / (ma_slow + 1e-9)

        components = [
            out.get("feature_trend_slope", pd.Series(0, index=out.index)),
            out.get("feature_trend_strength", pd.Series(0, index=out.index)),
            out.get("feature_trend_htf_alignment", pd.Series(0, index=out.index)),
            efficiency_ratio.fillna(0),
            (-noise_ratio).fillna(0),
            ma_distance.fillna(0),
        ]

        combined = sum(comp.fillna(0) for comp in components) / len(components)
        out["trend_score"] = np.tanh(combined)
        return out
