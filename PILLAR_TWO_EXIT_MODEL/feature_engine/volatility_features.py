"""Volatility feature group computations."""

from __future__ import annotations

import numpy as np
import pandas as pd


class VolatilityFeatures:
    """Compute volatility-related primitives."""

    def __init__(
        self,
        atr_window: int = 14,
        realized_window: int = 20,
        percentile_window: int = 100,
    ) -> None:
        self.atr_window = atr_window
        self.realized_window = realized_window
        self.percentile_window = percentile_window

    @staticmethod
    def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        prev_close = close.shift(1)
        ranges = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        )
        return ranges.max(axis=1)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        high = out["high"]
        low = out["low"]
        close = out["close"]

        tr = self._true_range(high, low, close)
        atr = tr.rolling(self.atr_window, min_periods=self.atr_window).mean()
        out["feature_volatility_atr"] = atr
        out["feature_volatility_atr_expansion"] = atr.pct_change()

        returns = close.pct_change()
        realized = (
            returns.pow(2)
            .rolling(self.realized_window, min_periods=self.realized_window)
            .mean()
            .pow(0.5)
        )
        out["feature_volatility_realized"] = realized

        def pct_rank(x: np.ndarray) -> float:
            valid = x[~np.isnan(x)]
            if len(valid) == 0:
                return np.nan
            last = valid[-1]
            return (np.sum(valid <= last) / len(valid)) * 100.0

        out["feature_volatility_percentile"] = realized.rolling(
            self.percentile_window, min_periods=self.percentile_window
        ).apply(pct_rank, raw=True)

        return out
