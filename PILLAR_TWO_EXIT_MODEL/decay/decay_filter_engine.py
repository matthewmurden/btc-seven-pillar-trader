"""Decay and validity filter engine (Part 7)."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLS = [
    "mfe_potential_score",
    "mae_risk_score",
    "survival_confidence",
    "exit_regime",
    "momentum_score",
    "trend_score",
    "volatility_score",
    "structure_score",
]


def _bars_since_last_peak(series: pd.Series) -> pd.Series:
    """Return bars since last peak for a positive series (vectorised)."""
    values = series.to_numpy()
    cummax = np.maximum.accumulate(values)
    # index of last max
    peak_idx = np.zeros_like(values, dtype=int)
    last_peak = 0
    for i in range(len(values)):
        if values[i] >= cummax[i]:
            last_peak = i
        peak_idx[i] = last_peak
    bars_since = np.arange(len(values)) - peak_idx
    return pd.Series(bars_since, index=series.index, dtype=float)


def apply_decay_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic decay filters to exit signals."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for decay filtering: {missing}")

    out = df.copy()

    bars_since_peak_mfe = _bars_since_last_peak(out["mfe_potential_score"])
    bars_since_peak_mae = _bars_since_last_peak(out["mae_risk_score"])

    out["mfe_potential_score_decayed"] = out["mfe_potential_score"] * np.clip(
        1 - bars_since_peak_mfe / 10.0, 0.0, 1.0
    )
    out["mae_risk_score_decayed"] = out["mae_risk_score"] * np.exp(-0.4 * bars_since_peak_mae)

    regime_change = out["exit_regime"] != out["exit_regime"].shift(1)
    out["survival_confidence_decayed"] = out["survival_confidence"].where(~regime_change, 0)

    validity = (
        (out["mfe_potential_score_decayed"] > 0)
        | (out["mae_risk_score_decayed"] > 0)
        | (out["survival_confidence_decayed"] > 0)
    ).astype(int)
    out["valid_exit_signal"] = validity

    return out
