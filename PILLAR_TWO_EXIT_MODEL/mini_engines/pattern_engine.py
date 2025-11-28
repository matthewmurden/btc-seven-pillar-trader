"""Micro-pattern mini engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


class PatternEngine:
    """Compute pattern_score from pattern primitives."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        pin_bar = out.get("feature_pattern_pin_bar", pd.Series(0, index=out.index))
        engulfing = out.get("feature_pattern_engulfing", pd.Series(0, index=out.index))
        momentum_bar = out.get("feature_pattern_momentum_bar", pd.Series(0, index=out.index))
        wick_signal = out.get("feature_pattern_wick_signal", pd.Series(0, index=out.index))
        pattern_score = out.get("feature_pattern_score", pd.Series(0, index=out.index))

        cluster = pin_bar + engulfing + momentum_bar + wick_signal + pattern_score
        max_possible = 5  # simple normalization bound
        out["pattern_score"] = (cluster / max_possible).clip(lower=0, upper=1)
        return out
