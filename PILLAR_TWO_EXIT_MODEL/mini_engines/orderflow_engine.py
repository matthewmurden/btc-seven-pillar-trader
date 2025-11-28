"""Orderflow mini engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


class OrderflowEngine:
    """Compute orderflow_score from orderflow primitives."""

    def __init__(self, cumulative_window: int = 20, imbalance_change_window: int = 5) -> None:
        self.cumulative_window = cumulative_window
        self.imbalance_change_window = imbalance_change_window

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        delta = out["delta"]
        raw_imb = out.get("feature_orderflow_raw_imbalance", pd.Series(0, index=out.index))
        norm_imb = out.get("feature_orderflow_norm_imbalance", pd.Series(0, index=out.index))

        sweep_up = out.get("feature_orderflow_sweep_up", pd.Series(0, index=out.index))
        sweep_down = out.get("feature_orderflow_sweep_down", pd.Series(0, index=out.index))

        imbalance_change = norm_imb.diff(self.imbalance_change_window)
        cumulative_imbalance = delta.rolling(self.cumulative_window, min_periods=1).sum()

        components = [
            norm_imb,
            raw_imb,
            cumulative_imbalance / (self.cumulative_window + 1e-9),
            imbalance_change.fillna(0),
            (sweep_up.astype(float) - sweep_down.astype(float)),
        ]

        combined = sum(comp.fillna(0) for comp in components) / len(components)
        out["orderflow_score"] = np.tanh(combined)
        return out
