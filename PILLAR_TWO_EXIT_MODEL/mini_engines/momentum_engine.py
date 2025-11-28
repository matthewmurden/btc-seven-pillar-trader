"""Momentum mini engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


class MomentumEngine:
    """Compute momentum_score from momentum-related primitives."""

    def __init__(
        self,
        vol_window: int = 20,
        volume_htf_window: int = 60,
    ) -> None:
        self.vol_window = vol_window
        self.volume_htf_window = volume_htf_window

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        close = out["close"]
        volume = out["volume"]

        returns = close.pct_change()
        realized_vol = (
            returns.pow(2)
            .rolling(self.vol_window, min_periods=self.vol_window)
            .mean()
            .pow(0.5)
        )
        vol_adjusted_momentum = returns / (realized_vol + 1e-9)

        mom_short = out.get("feature_momentum_mom_short")
        mom_acceleration = mom_short.diff() if mom_short is not None else pd.Series(
            np.nan, index=out.index
        )

        htf_volume = volume.rolling(self.volume_htf_window, min_periods=1).mean()
        htf_volume_alignment = (volume / (htf_volume + 1e-9)).fillna(0)

        components = [
            out.get("feature_momentum_roc_5", pd.Series(0, index=out.index)),
            out.get("feature_momentum_roc_10", pd.Series(0, index=out.index)),
            out.get("feature_momentum_roc_20", pd.Series(0, index=out.index)),
            out.get("feature_momentum_zscore_returns", pd.Series(0, index=out.index)),
            out.get("feature_momentum_obv_slope", pd.Series(0, index=out.index)),
            vol_adjusted_momentum,
            mom_acceleration,
            htf_volume_alignment,
        ]

        combined = sum(comp.fillna(0) for comp in components) / len(components)
        out["momentum_score"] = np.tanh(combined)

        return out
