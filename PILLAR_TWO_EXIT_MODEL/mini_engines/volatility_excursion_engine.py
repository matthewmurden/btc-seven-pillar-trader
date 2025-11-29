"""Volatility–Excursion deterministic engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


class VolatilityExcursionEngine:
    """Scores excursion potential based on volatility context."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        atr_pct = out.get("feature_exit_vol_percentile", pd.Series(0, index=out.index)) / 100.0
        vol_compression = out.get("feature_exit_vol_compression", pd.Series(0, index=out.index))
        vol_slope = out.get("feature_exit_atr_slope", pd.Series(0, index=out.index))
        breakout_flag = out.get("breakout_label", pd.Series(0, index=out.index))

        score = (
            0.5 * (1 - vol_compression.astype(float))
            + 0.3 * atr_pct.fillna(0)
            + 0.1 * np.tanh(vol_slope.fillna(0))
            + 0.1 * breakout_flag.astype(float)
        )
        out["excursion_score"] = np.clip(score, 0, 1)
        return out
