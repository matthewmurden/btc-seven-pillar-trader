"""Orderflow feature group computations."""

from __future__ import annotations

import numpy as np
import pandas as pd


class OrderflowFeatures:
    """Compute orderflow-related primitives."""

    def __init__(
        self,
        imbalance_window: int = 20,
        sweep_threshold: float = 0.6,
    ) -> None:
        self.imbalance_window = imbalance_window
        self.sweep_threshold = sweep_threshold

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        volume = out["volume"].replace(0, np.nan)
        delta = out["delta"]

        out["feature_orderflow_raw_imbalance"] = delta / (volume + 1e-9)

        delta_std = delta.rolling(self.imbalance_window, min_periods=self.imbalance_window).std(ddof=0)
        out["feature_orderflow_norm_imbalance"] = delta / (delta_std + 1e-9)

        candle_range = (out["high"] - out["low"]).replace(0, np.nan)
        wick_top_ratio = out.get("wick_top", pd.Series(0, index=out.index)) / candle_range
        wick_bottom_ratio = out.get("wick_bottom", pd.Series(0, index=out.index)) / candle_range

        out["feature_orderflow_sweep_up"] = (
            (wick_bottom_ratio.fillna(0) > self.sweep_threshold)
        ).astype(int)
        out["feature_orderflow_sweep_down"] = (
            (wick_top_ratio.fillna(0) > self.sweep_threshold)
        ).astype(int)

        return out
