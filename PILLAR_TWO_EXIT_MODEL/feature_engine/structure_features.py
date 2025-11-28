"""Structure feature group computations."""

from __future__ import annotations

import numpy as np
import pandas as pd


class StructureFeatures:
    """Compute market structure primitives."""

    def __init__(self, swing_lookback: int = 2) -> None:
        self.swing_lookback = swing_lookback

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        high = out["high"]
        low = out["low"]
        close = out["close"]

        swing_high = (
            (high > high.shift(1))
            & (high > high.shift(-1))
            & (high > high.shift(self.swing_lookback))
        )
        swing_low = (
            (low < low.shift(1))
            & (low < low.shift(-1))
            & (low < low.shift(self.swing_lookback))
        )

        out["feature_structure_swing_high"] = swing_high.astype(int)
        out["feature_structure_swing_low"] = swing_low.astype(int)

        last_swing_high = high.where(swing_high).ffill()
        last_swing_low = low.where(swing_low).ffill()

        bos_up = close > last_swing_high.shift(1)
        bos_down = close < last_swing_low.shift(1)

        out["feature_structure_bos_up"] = bos_up.astype(int)
        out["feature_structure_bos_down"] = bos_down.astype(int)

        bos_down_shifted = bos_down.shift(1)
        bos_down_shifted = pd.Series(
            np.where(bos_down_shifted.isna(), False, bos_down_shifted),
            index=bos_down_shifted.index,
        ).astype(bool)

        bos_up_shifted = bos_up.shift(1)
        bos_up_shifted = pd.Series(
            np.where(bos_up_shifted.isna(), False, bos_up_shifted),
            index=bos_up_shifted.index,
        ).astype(bool)

        choch_up = bos_up & bos_down_shifted
        choch_down = bos_down & bos_up_shifted

        out["feature_structure_choch_up"] = choch_up.astype(int)
        out["feature_structure_choch_down"] = choch_down.astype(int)

        out["feature_structure_signal"] = (
            bos_up.astype(int)
            - bos_down.astype(int)
            + (choch_up.astype(int) * 2)
            - (choch_down.astype(int) * 2)
        )

        return out
