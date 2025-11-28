"""Pattern feature group computations."""

from __future__ import annotations

import numpy as np
import pandas as pd


class PatternFeatures:
    """Compute micro-pattern primitives."""

    def __init__(self, momentum_ratio_threshold: float = 0.7) -> None:
        self.momentum_ratio_threshold = momentum_ratio_threshold

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        open_ = out["open"]
        close = out["close"]
        high = out["high"]
        low = out["low"]

        body = (close - open_).abs()
        candle_range = (high - low).replace(0, np.nan)
        wick_top = out.get("wick_top", pd.Series(0, index=out.index))
        wick_bottom = out.get("wick_bottom", pd.Series(0, index=out.index))

        pin_bar = (
            (wick_top > 0.5 * candle_range) | (wick_bottom > 0.5 * candle_range)
        ) & (body < 0.3 * candle_range)
        out["feature_pattern_pin_bar"] = pin_bar.astype(int)

        prev_open = open_.shift(1)
        prev_close = close.shift(1)
        engulfing = (
            (body > body.shift(1))
            & (
                ((close > open_) & (prev_close < prev_open) & (close > prev_open) & (open_ < prev_close))
                | ((close < open_) & (prev_close > prev_open) & (close < prev_open) & (open_ > prev_close))
            )
        )
        out["feature_pattern_engulfing"] = engulfing.astype(int)

        momentum_bar = (body / (candle_range + 1e-9)) > self.momentum_ratio_threshold
        out["feature_pattern_momentum_bar"] = momentum_bar.astype(int)

        wick_signal = (
            (wick_top / (candle_range + 1e-9) > 0.4)
            | (wick_bottom / (candle_range + 1e-9) > 0.4)
        )
        out["feature_pattern_wick_signal"] = wick_signal.astype(int)

        out["feature_pattern_score"] = (
            out["feature_pattern_pin_bar"]
            + out["feature_pattern_engulfing"]
            + out["feature_pattern_momentum_bar"]
            + out["feature_pattern_wick_signal"]
        )

        return out
