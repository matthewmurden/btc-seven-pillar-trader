"""Momentum feature group computations."""

from __future__ import annotations

import numpy as np
import pandas as pd


class MomentumFeatures:
    """Compute momentum-related primitives."""

    def __init__(
        self,
        roc_windows: tuple[int, int, int] = (5, 10, 20),
        mom_windows: tuple[int, int] = (3, 7),
        zscore_window: int = 20,
        obv_slope_window: int = 20,
    ) -> None:
        self.roc_windows = roc_windows
        self.mom_windows = mom_windows
        self.zscore_window = zscore_window
        self.obv_slope_window = obv_slope_window

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
        volume = out["volume"].fillna(0)
        returns = close.pct_change()

        for window in self.roc_windows:
            out[f"feature_momentum_roc_{window}"] = close.pct_change(window)

        out["feature_momentum_mom_short"] = close.diff(self.mom_windows[0])
        out["feature_momentum_mom_medium"] = close.diff(self.mom_windows[1])

        rolling_returns = returns.rolling(self.zscore_window, min_periods=self.zscore_window)
        out["feature_momentum_zscore_returns"] = (
            (returns - rolling_returns.mean())
            / (rolling_returns.std(ddof=0) + 1e-9)
        )

        direction = np.sign(close.diff()).fillna(0)
        obv = (direction * volume).cumsum()
        out["feature_momentum_obv"] = obv
        out["feature_momentum_obv_slope"] = self._rolling_slope(
            obv, self.obv_slope_window
        )

        return out
