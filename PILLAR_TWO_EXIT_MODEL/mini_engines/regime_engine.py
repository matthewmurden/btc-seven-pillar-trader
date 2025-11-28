"""Regime mini engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


class RegimeEngine:
    """Compute regime_state (and optional confidence) from regime primitives."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        vol_regime = out.get("feature_regime_volatility", pd.Series(0, index=out.index))
        trend_vs_range = out.get("feature_regime_trend_vs_range", pd.Series(0, index=out.index))
        breakout_comp = out.get("feature_regime_breakout_compression", pd.Series(0, index=out.index))

        # Basic mapping: trend_vs_range drives base state, breakout overrides.
        regime_state = np.where(
            breakout_comp > 0, 2,
            np.where(trend_vs_range > 0, 1, 0)
        )

        # Confidence from agreement of signals.
        confidence = (
            trend_vs_range.astype(float)
            + breakout_comp.astype(float) * 1.5
            + vol_regime.astype(float) * 0.5
        )
        confidence = 1 / (1 + np.exp(-confidence))

        out["regime_state"] = regime_state.astype(int)
        out["regime_confidence"] = confidence
        return out
