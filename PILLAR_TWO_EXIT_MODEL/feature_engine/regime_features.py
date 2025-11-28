"""Regime feature group computations."""

from __future__ import annotations

import numpy as np
import pandas as pd


class RegimeFeatures:
    """Compute regime-related primitives."""

    def __init__(
        self,
        vol_regime_window: int = 200,
        trend_threshold: float = 1.0,
    ) -> None:
        self.vol_regime_window = vol_regime_window
        self.trend_threshold = trend_threshold

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if "feature_volatility_realized" in out.columns:
            realized = out["feature_volatility_realized"]
        else:
            returns = out["close"].pct_change()
            realized = (
                returns.pow(2)
                .rolling(20, min_periods=20)
                .mean()
                .pow(0.5)
            )

        low_thresh = realized.rolling(self.vol_regime_window, min_periods=10).quantile(0.33)
        high_thresh = realized.rolling(self.vol_regime_window, min_periods=10).quantile(0.66)

        regime_volatility = np.select(
            [realized <= low_thresh, realized >= high_thresh],
            [-1, 1],
            default=0,
        )
        out["feature_regime_volatility"] = regime_volatility

        trend_strength = out.get("feature_trend_strength")
        if trend_strength is None:
            trend_strength = out["close"].diff().rolling(20, min_periods=20).mean()

        trend_vs_range = (trend_strength.abs() > self.trend_threshold).astype(int)
        out["feature_regime_trend_vs_range"] = trend_vs_range

        atr_expansion = out.get("feature_volatility_atr_expansion")
        if atr_expansion is None:
            atr_expansion = realized.pct_change()

        breakout_compression = (
            (regime_volatility == -1)
            & (atr_expansion < 0)
        ).astype(int)
        out["feature_regime_breakout_compression"] = breakout_compression

        return out
