"""Final exit decision layer (Part 10)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd


REQUIRED_COLS = [
    "entry_signal",
    "direction",
    "entry_price",
    "entry_confidence",
    "mae_pred",
    "mfe_pred",
    "exit_time_pred",
    "optimal_tp_pred",
    "optimal_sl_pred",
    "tp_first_pred",
    "sl_first_pred",
    "asymmetry_ratio",
    "reward_risk_prediction",
    "survival_confidence",
    "volatility_after_entry_pred",
    "exit_direction_consistency",
    "exit_quality_score",
    "recommended_tp_pct",
    "recommended_sl_pct",
]


@dataclass
class FinalExitDecision:
    """Compute final trading decision based on prior pipeline outputs."""

    quality_threshold: float = 0.25
    min_tp: float = 0.05
    max_mae: float = 0.02

    def _validate_row(self, row: pd.Series) -> None:
        missing = [c for c in REQUIRED_COLS if c not in row]
        if missing:
            raise ValueError(f"Missing required columns for decision: {missing}")

    def decide(self, row: pd.Series) -> Dict[str, Any]:
        self._validate_row(row)

        take_trade = True
        position_size_factor = 1.0

        if not row["entry_signal"]:
            take_trade = False
        if row["exit_quality_score"] < self.quality_threshold:
            take_trade = False
        if row["recommended_tp_pct"] < self.min_tp:
            take_trade = False

        if abs(row["mae_pred"]) > self.max_mae:
            position_size_factor = 0.5

        if not take_trade:
            return {
                "take_trade": False,
                "direction": 0,
                "entry_price": row.get("entry_price", np.nan),
                "tp_pct": 0.0,
                "sl_pct": 0.0,
                "expected_exit_time": 0.0,
                "position_size_factor": 0.0,
            }

        return {
            "take_trade": True,
            "direction": int(row["direction"]),
            "entry_price": float(row["entry_price"]),
            "tp_pct": float(row["recommended_tp_pct"]),
            "sl_pct": float(row["recommended_sl_pct"]),
            "expected_exit_time": float(row["exit_time_pred"]),
            "position_size_factor": float(position_size_factor),
        }

    def run_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for batch decision: {missing}")

        decisions = df.apply(self.decide, axis=1, result_type="expand")
        decisions.columns = [
            "take_trade",
            "final_direction",
            "entry_price",
            "final_tp_pct",
            "final_sl_pct",
            "final_expected_exit_time",
            "final_position_size_factor",
        ]
        return pd.concat([df.reset_index(drop=True), decisions], axis=1)
