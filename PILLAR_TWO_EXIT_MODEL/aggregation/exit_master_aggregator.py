"""
Exit Master Aggregator
----------------------
Combines meta, decay, optimisation, and mini-engine signals into a unified
exit_quality_score in [0, 1].
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Required inputs
# ---------------------------------------------------------------------
REQUIRED_COLUMNS: List[str] = [
    "asymmetry_ratio",
    "reward_risk_prediction",
    "survival_confidence",
    "volatility_after_entry_pred",
    "exit_direction_consistency",
    "expected_exit_time",
    "decayed_mfe_potential",
    "decayed_mae_risk",
    "survival_confidence_filtered",
    "integrity_score",
    "valid_signal_flag",
    "meta_exit_strength",
    "final_exit_strength",
    "tp_first_proba",
    "sl_first_proba",
    "recommended_tp_pct",
    "recommended_sl_pct",
    "path_entropy_score",
    "trend_continuation_score",
    "volatility_score",
    "exit_regime",
    "mae_risk_score",
]


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _zscore(arr: np.ndarray) -> np.ndarray:
    mean = np.nanmean(arr)
    std = np.nanstd(arr) + 1e-12
    return (arr - mean) / std


def _ensure_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for aggregation: {missing}")


def _compute_entry_confidence(df: pd.DataFrame) -> np.ndarray:
    """Derived entry confidence when not provided."""
    trend = df["trend_continuation_score"].to_numpy(dtype=np.float64)
    mae = df["mae_risk_score"].to_numpy(dtype=np.float64)
    return _sigmoid(0.5 * trend - 0.5 * mae)


# ---------------------------------------------------------------------
# Main aggregation engine
# ---------------------------------------------------------------------
def apply_exit_master_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    _ensure_columns(df, REQUIRED_COLUMNS)

    # Copy and sanitize numeric columns ONLY
    working = df.copy()

    for idx in range(working.shape[1]):
        col = working.columns[idx]
        series = working.iloc[:, idx]

        # Only clean numeric types (float, int)
        if series.dtype.kind in ("i", "u", "f"):
            arr = series.to_numpy(dtype=np.float64)
            arr = np.where(np.isinf(arr), np.nan, arr)
            arr = np.where(np.isnan(arr), 0.0, arr)
            working.iloc[:, idx] = arr

    # ---------------------------------------------------------------
    # Build component signals
    # ---------------------------------------------------------------
    expectancy = (
        working["tp_first_proba"].to_numpy(dtype=np.float64)
        - working["sl_first_proba"].to_numpy(dtype=np.float64)
    )
    expectancy_signal = _sigmoid(expectancy * 3.0)

    vol_z = _zscore(working["volatility_score"].to_numpy(dtype=np.float64))
    volatility_factor = np.exp(-np.abs(vol_z))

    asymmetry_factor = _sigmoid(
        working["asymmetry_ratio"].to_numpy(dtype=np.float64) / 2.0
    )
    survival_factor = np.power(
        working["survival_confidence_filtered"].to_numpy(dtype=np.float64), 1.5
    )
    integrity_penalty = _sigmoid(
        working["integrity_score"].to_numpy(dtype=np.float64) / 2.0
    )

    # Regime logic
    regime = working["exit_regime"].astype(int).to_numpy()
    regime_factor = np.where(
        regime == 0,
        1.0,
        np.where(
            regime == 1,
            0.7,
            np.where(regime == 2, 0.85, np.where(regime == 3, 0.9, 0.8)),
        ),
    )

    # Entry confidence (either provided or derived)
    if "entry_model_confidence" in working.columns:
        entry_confidence = working["entry_model_confidence"].to_numpy(
            dtype=np.float64
        )
    else:
        entry_confidence = _compute_entry_confidence(working)

    # ---------------------------------------------------------------
    # Weighted aggregation
    # ---------------------------------------------------------------
    exit_quality_raw = (
        0.25 * asymmetry_factor
        + 0.25 * expectancy_signal
        + 0.20 * survival_factor
        + 0.15 * volatility_factor
        + 0.10 * entry_confidence
    )

    exit_quality_raw *= integrity_penalty
    exit_quality_raw *= regime_factor
    exit_quality_raw *= working["valid_signal_flag"].to_numpy(dtype=np.float64)

    exit_quality_score = np.clip(
        _sigmoid(exit_quality_raw), 0.0, 1.0
    ).astype(np.float32)

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------
    if not np.isfinite(exit_quality_score).all():
        raise ValueError("Non-finite values detected in exit_quality_score.")

    working["exit_quality_score"] = exit_quality_score

    # Validate ALL numeric columns
    numeric_cols = []
    for idx, dtype in enumerate(working.dtypes):
        if dtype.kind in ("i", "u", "f"):
            numeric_cols.append(working.iloc[:, idx].to_numpy())

    if numeric_cols:
        merged_num = np.column_stack(numeric_cols)
        if not np.isfinite(merged_num).all():
            raise ValueError("Non-finite values detected in aggregated output.")

    return working


# ---------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    base_dir = Path(__file__).resolve().parents[1]

    meta_path = base_dir / "data" / "processed" / "meta_exit_features.parquet"
    decay_path = base_dir / "data" / "processed" / "exit_decay_filtered.parquet"
    opt_path = base_dir / "data" / "processed" / "exit_optimised.parquet"
    mini_path = base_dir / "data" / "processed" / "exit_features_with_mini_engines.parquet"

    merged = pd.concat(
        [
            pd.read_parquet(meta_path).reset_index(drop=True),
            pd.read_parquet(decay_path).reset_index(drop=True),
            pd.read_parquet(opt_path).reset_index(drop=True),
            pd.read_parquet(mini_path).reset_index(drop=True),
        ],
        axis=1,
    )

    out = apply_exit_master_aggregation(merged)

    out_path = base_dir / "data" / "processed" / "exit_master.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    logger.info("Saved exit_master to %s", out_path)