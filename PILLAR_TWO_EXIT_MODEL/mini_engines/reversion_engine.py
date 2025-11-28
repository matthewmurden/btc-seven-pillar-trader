"""Mean reversion mini engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


class ReversionEngine:
    """Compute reversion_score from reversion primitives."""

    def __init__(self, acceleration_window: int = 5, vol_window: int = 20) -> None:
        self.acceleration_window = acceleration_window
        self.vol_window = vol_window

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        close = out["close"]

        returns = close.pct_change()
        vol = (
            returns.pow(2)
            .rolling(self.vol_window, min_periods=self.vol_window)
            .mean()
            .pow(0.5)
        )

        rsi = out.get("feature_reversion_rsi", pd.Series(np.nan, index=out.index))
        zdev = out.get("feature_reversion_zscore_dev", pd.Series(np.nan, index=out.index))
        overshoot = out.get("feature_reversion_rsi_overshoot", pd.Series(0, index=out.index))
        impulse = out.get("feature_reversion_impulse", pd.Series(0, index=out.index))

        mr_acceleration = returns.diff(self.acceleration_window)

        components = [
            (50 - rsi) / 50.0,
            -zdev,
            -overshoot / 100.0,
            -impulse,
            -mr_acceleration.fillna(0),
            -(returns / (vol + 1e-9)).fillna(0),
        ]

        combined = sum(comp.fillna(0) for comp in components) / len(components)
        out["reversion_score"] = np.tanh(combined)
        return out
