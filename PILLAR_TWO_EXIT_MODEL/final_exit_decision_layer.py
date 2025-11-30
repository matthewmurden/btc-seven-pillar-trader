"""
Final Exit Decision Layer for Pillar Two.

Determines tradable exit actions from calibrated signals, meta/decay signals,
and optimisation outputs. Supports both single-row and vectorised batch mode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

QUALITY_BASE_THRESHOLD = 0.52
VOL_Z_FACTOR = 0.02
REGIME_PENALTY_FACTOR = 0.01


REQUIRED_COLUMNS = [
    "entry_signal",
    "entry_direction",
    "entry_price",
    "tp_first_calibrated",
    "sl_first_calibrated",
    "exit_quality_calibrated",
    "recommended_tp_pct",
    "recommended_sl_pct",
    "expected_exit_time",
    "decayed_mae_risk",
    "decayed_mfe_potential",
    "survival_confidence_filtered",
    "meta_exit_strength",
    "final_exit_strength",
    "regime",
    "volatility_score",
]


@dataclass
class ExitDecision:
    decision: str
    direction: Optional[str]
    entry_price: Optional[float]
    tp_price: Optional[float]
    sl_price: Optional[float]
    tp_pct: Optional[float]
    sl_pct: Optional[float]
    expected_exit_time: Optional[float]
    position_size_multiplier: float
    exit_quality: float
    expectancy: float
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "decision": self.decision,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "tp_price": self.tp_price,
            "sl_price": self.sl_price,
            "tp_pct": self.tp_pct,
            "sl_pct": self.sl_pct,
            "expected_exit_time": self.expected_exit_time,
            "position_size_multiplier": self.position_size_multiplier,
            "exit_quality": self.exit_quality,
            "expectancy": self.expectancy,
            "reason": self.reason,
        }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _vol_z(vol_score: float, series_mean: float, series_std: float) -> float:
    std = series_std if series_std > 1e-12 else 1.0
    return (vol_score - series_mean) / std


def _validate_row(row: pd.Series) -> Optional[str]:
    if row["entry_signal"] != 1:
        return "entry_signal==0"
    if not np.isfinite(row["entry_price"]) or row["entry_price"] <= 0:
        return "invalid_entry_price"
    if not np.isfinite(row["recommended_tp_pct"]) or not np.isfinite(row["recommended_sl_pct"]):
        return "invalid_tp_sl_pct"
    return None


def _compute_threshold(base: float, vol_z: float, regime: float) -> float:
    return base + VOL_Z_FACTOR * vol_z + REGIME_PENALTY_FACTOR * regime


def _position_multiplier(mae_risk: float) -> float:
    if mae_risk > 0.7:
        return 0.0
    if mae_risk > 0.5:
        return 0.4
    if mae_risk > 0.3:
        return 0.7
    return 1.0


def _compute_tp_sl(entry_price: float, tp_pct: float, sl_pct: float, direction: int) -> tuple[float, float]:
    if direction == 1:
        tp_price = entry_price * (1.0 + tp_pct)
        sl_price = entry_price * (1.0 - sl_pct)
    else:
        tp_price = entry_price * (1.0 - tp_pct)
        sl_price = entry_price * (1.0 + sl_pct)
    return tp_price, sl_price


def apply_single_decision(row: pd.Series, vol_mean: float, vol_std: float) -> ExitDecision:
    """Compute a single exit decision from one row of signals."""
    reason = []

    # Validation
    invalid_reason = _validate_row(row)
    if invalid_reason:
        return ExitDecision(
            decision="NO_TRADE",
            direction=None,
            entry_price=None,
            tp_price=None,
            sl_price=None,
            tp_pct=None,
            sl_pct=None,
            expected_exit_time=None,
            position_size_multiplier=0.0,
            exit_quality=0.0,
            expectancy=0.0,
            reason=invalid_reason,
        )

    direction = "LONG" if row["entry_direction"] == 1 else "SHORT"
    expectancy = float(row["tp_first_calibrated"] - row["sl_first_calibrated"])

    # Thresholds
    vol_z = _vol_z(float(row["volatility_score"]), vol_mean, vol_std)
    threshold = _compute_threshold(QUALITY_BASE_THRESHOLD, vol_z, float(row["regime"]))

    # Rules
    if row["exit_quality_calibrated"] < threshold:
        reason.append("low_exit_quality")
    if expectancy < -0.1:
        reason.append("negative_expectancy")
    if row["recommended_tp_pct"] < 0.05 or row["recommended_sl_pct"] < 0.02:
        reason.append("untradable_tp_sl")

    pos_mult = _position_multiplier(float(row["decayed_mae_risk"]))
    if pos_mult == 0.0:
        reason.append("mae_risk_high")

    if reason:
        return ExitDecision(
            decision="NO_TRADE",
            direction=direction,
            entry_price=float(row["entry_price"]),
            tp_price=None,
            sl_price=None,
            tp_pct=None,
            sl_pct=None,
            expected_exit_time=None,
            position_size_multiplier=pos_mult,
            exit_quality=float(row["exit_quality_calibrated"]),
            expectancy=expectancy,
            reason=";".join(reason),
        )

    tp_price, sl_price = _compute_tp_sl(
        float(row["entry_price"]), float(row["recommended_tp_pct"]), float(row["recommended_sl_pct"]), int(row["entry_direction"])
    )

    if tp_price <= 0 or sl_price <= 0 or not np.isfinite(tp_price) or not np.isfinite(sl_price):
        return ExitDecision(
            decision="NO_TRADE",
            direction=direction,
            entry_price=float(row["entry_price"]),
            tp_price=None,
            sl_price=None,
            tp_pct=None,
            sl_pct=None,
            expected_exit_time=None,
            position_size_multiplier=pos_mult,
            exit_quality=float(row["exit_quality_calibrated"]),
            expectancy=expectancy,
            reason="invalid_tp_sl_prices",
        )

    return ExitDecision(
        decision="TAKE_TRADE",
        direction=direction,
        entry_price=float(row["entry_price"]),
        tp_price=tp_price,
        sl_price=sl_price,
        tp_pct=float(row["recommended_tp_pct"]),
        sl_pct=float(row["recommended_sl_pct"]),
        expected_exit_time=float(row["expected_exit_time"]),
        position_size_multiplier=pos_mult,
        exit_quality=float(row["exit_quality_calibrated"]),
        expectancy=expectancy,
        reason="ok",
    )


def apply_final_exit_decisions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised application of the final exit decision logic.
    Returns a DataFrame with decision outputs appended.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for final decision: {missing}")

    working = df.copy()
    # Clean inputs
    for idx, dtype in enumerate(working.dtypes):
        if dtype.kind in ("i", "u", "f"):
            col_data = working.iloc[:, idx].to_numpy()
            col_data = np.where(np.isinf(col_data), np.nan, col_data)
            col_data = np.where(np.isnan(col_data), 0.0, col_data)
            working.iloc[:, idx] = col_data

    vol_mean = float(np.nanmean(working["volatility_score"]))
    vol_std = float(np.nanstd(working["volatility_score"]) + 1e-12)

    decisions = [apply_single_decision(row, vol_mean, vol_std).to_dict() for _, row in working.iterrows()]
    decisions_df = pd.DataFrame(decisions, index=working.index)

    out = pd.concat([working, decisions_df], axis=1)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sample = pd.DataFrame({col: [0] for col in REQUIRED_COLUMNS})
    result = apply_final_exit_decisions(sample)
    print(result.head())
