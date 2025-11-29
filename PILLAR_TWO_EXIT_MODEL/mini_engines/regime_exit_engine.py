"""Regime Exit deterministic engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


class RegimeExitEngine:
    """Classifies exit regime based on volatility/trend/breakout states."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        vol_regime = out.get("feature_regime_volatility", pd.Series(0, index=out.index))
        trend_vs_range = out.get("feature_regime_trend_vs_range", pd.Series(0, index=out.index))
        breakout_regime = out.get("feature_regime_breakout_compression", pd.Series(0, index=out.index))

        exit_regime = np.where(
            breakout_regime > 0, 2,
            np.where(trend_vs_range > 0, 1, 0)
        )
        out["exit_regime"] = exit_regime.astype(int)
        return out
