"""Reversion feature group computations."""

from __future__ import annotations

import numpy as np
import pandas as pd


class ReversionFeatures:
    """Compute reversion-related primitives."""

    def __init__(
        self,
        rsi_window: int = 14,
        zscore_window: int = 20,
    ) -> None:
        self.rsi_window = rsi_window
        self.zscore_window = zscore_window

    @staticmethod
    def _compute_rsi(close: pd.Series, window: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

        rs = avg_gain / (avg_loss + 1e-9)
        return 100 - (100 / (1 + rs))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        close = out["close"]

        out["feature_reversion_rsi"] = self._compute_rsi(close, self.rsi_window)

        rolling = close.rolling(self.zscore_window, min_periods=self.zscore_window)
        zscore_dev = (close - rolling.mean()) / (rolling.std(ddof=0) + 1e-9)
        out["feature_reversion_zscore_dev"] = zscore_dev

        overshoot = pd.Series(0.0, index=close.index)
        overshoot = overshoot.mask(out["feature_reversion_rsi"] > 70, out["feature_reversion_rsi"] - 70)
        overshoot = overshoot.mask(out["feature_reversion_rsi"] < 30, out["feature_reversion_rsi"] - 30)
        out["feature_reversion_rsi_overshoot"] = overshoot

        returns = close.pct_change()
        out["feature_reversion_impulse"] = -returns * zscore_dev.abs()

        return out
