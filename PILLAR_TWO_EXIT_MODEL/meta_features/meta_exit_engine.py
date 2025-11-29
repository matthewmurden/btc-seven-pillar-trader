"""Meta exit feature engine (Part 5)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLS: Iterable[str] = [
    "mae_pred",
    "mfe_pred",
    "optimal_tp_pred",
    "optimal_sl_pred",
    "tp_first_pred",
    "sl_first_pred",
    "exit_time_pred",
    "volatility_score",
    "regime_state",
]

META_COLS = [
    "asymmetry_ratio",
    "reward_risk_prediction",
    "survival_confidence",
    "volatility_after_entry_pred",
    "exit_direction_consistency",
    "expected_exit_time",
]

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "meta_features.parquet"


def build_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute meta-level exit features from model predictions."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for meta features: {missing}")

    out = df.copy()
    mae = out["mae_pred"].astype(float)
    mfe = out["mfe_pred"].astype(float)
    tp = out["optimal_tp_pred"].astype(float)
    sl = out["optimal_sl_pred"].astype(float)
    tp_first = out["tp_first_pred"].astype(float)
    sl_first = out["sl_first_pred"].astype(float)
    exit_time = out["exit_time_pred"].astype(float)
    vol_score = out["volatility_score"].astype(float)

    out["asymmetry_ratio"] = mfe / np.maximum(np.abs(mae), 1e-12)
    out["reward_risk_prediction"] = tp / np.maximum(np.abs(sl), 1e-12)
    out["survival_confidence"] = np.exp(-exit_time / (1 + vol_score))
    out["volatility_after_entry_pred"] = vol_score * (1 + np.abs(mae) + np.abs(mfe))
    out["exit_direction_consistency"] = tp_first - sl_first
    out["expected_exit_time"] = exit_time

    # enforce no NaNs
    for col in META_COLS:
        if out[col].isna().any() or np.isinf(out[col].values).any():
            raise ValueError(f"Invalid values in meta feature {col}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out[META_COLS].to_parquet(OUTPUT_PATH)
    return out
