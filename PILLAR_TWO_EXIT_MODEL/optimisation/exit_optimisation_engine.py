"""
Exit Optimisation Engine
------------------------
Regime-aware, penalty-adjusted TP/SL selection to maximise expectancy.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

EPS = 1e-8
TP_STEPS = 20
SL_STEPS = 20


# ---------------------------------------------------------------------------
# Grid builders
# ---------------------------------------------------------------------------
def _tp_grid(atr: float, expected_exit_time: float) -> np.ndarray:
    """Build TP grid using ATR and expected exit time regime."""
    if atr <= 0:
        return np.array([1.0], dtype=np.float64)
    tp_min = 0.5 * atr
    if expected_exit_time < 2:
        tp_max = 2.0 * atr
    elif expected_exit_time < 5:
        tp_max = 3.0 * atr
    else:
        tp_max = 4.0 * atr
    return np.linspace(tp_min, tp_max, TP_STEPS, dtype=np.float64)


def _sl_grid(atr: float, path_entropy: float) -> np.ndarray:
    """Build SL grid using ATR and path entropy."""
    if atr <= 0:
        return np.array([1.0], dtype=np.float64)
    entropy = np.clip(path_entropy, 0.0, 1.0)
    sl_min = (0.5 + 0.2 * entropy) * atr  # 0.5*ATR (smooth) to 0.7*ATR (choppy)
    sl_max = 3.0 * atr
    return np.linspace(sl_min, sl_max, SL_STEPS, dtype=np.float64)


# ---------------------------------------------------------------------------
# Regime adjustment
# ---------------------------------------------------------------------------
def _regime_adjustment(tp_pct: np.ndarray, sl_pct: np.ndarray, atr_slope: float, regime: int, volatility_score: float) -> np.ndarray:
    """
    Compute multiplicative regime weight for expectancy.
    - Breakout: bonus increases with larger TP.
    - Trend: bonus around mid TP.
    - Range: penalise large TP, favour tighter SL.
    Includes small adjustments from ATR slope and volatility score.
    """
    tp_norm = tp_pct / (tp_pct.max() + EPS)
    sl_norm = sl_pct / (sl_pct.max() + EPS)

    if regime == 2:
        weight = 1.10 + 0.10 * tp_norm  # +10–20% for large TP
    elif regime == 1:
        mid_bonus = 1 - np.abs(tp_norm - 0.5) * 2  # peak at mid
        weight = 1.05 + 0.05 * mid_bonus  # +5–10% at mid TP
    else:
        weight = 1.0 - 0.15 * tp_norm + 0.05 * (1 - sl_norm)  # penalise large TP, prefer tighter SL
        weight = np.clip(weight, 0.7, 1.05)

    weight *= 1 + 0.05 * np.tanh(atr_slope)
    weight *= 1 - 0.05 * np.tanh(volatility_score)
    return weight


# ---------------------------------------------------------------------------
# Core optimiser
# ---------------------------------------------------------------------------
def _advanced_expectancy(
    tp_prob: float,
    sl_prob: float,
    tp_grid: np.ndarray,
    sl_grid: np.ndarray,
    survival: float,
    vol_after_entry_z: float,
    mae_risk: float,
    atr_slope: float,
    regime: int,
    volatility_score: float,
) -> Tuple[float, float, float]:
    """
    Compute best TP/SL using vectorised grid for a single row.
    Returns best_tp_pct, best_sl_pct, best_expectancy.
    """
    tp_mat = tp_grid[:, None]  # (TP_STEPS,1)
    sl_mat = sl_grid[None, :]  # (1,SL_STEPS)

    base = (tp_prob * tp_mat) - (sl_prob * sl_mat)
    survival_penalty = float(np.clip(survival, 0.0, 1.0))
    vol_penalty = float(np.exp(-float(vol_after_entry_z) ** 2 / 2.0))
    mae_penalty = float(np.exp(-float(mae_risk) / 3.0))

    regime_w = _regime_adjustment(tp_mat, sl_mat, atr_slope, regime, volatility_score)

    expectancy = base * survival_penalty * vol_penalty * mae_penalty * regime_w

    idx = np.argmax(expectancy)
    tp_idx, sl_idx = np.unravel_index(idx, expectancy.shape)
    return float(tp_mat[tp_idx, 0]), float(sl_mat[0, sl_idx]), float(expectancy[tp_idx, sl_idx])


def optimise_exits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimise TP/SL for each trade entry.

    Required columns:
        exit_direction_consistency, survival_confidence, volatility_after_entry_pred,
        volatility_after_entry_z, mae_risk_score, ATR_14, expected_exit_time,
        exit_regime, ATR_slope, volatility_score, path_entropy_score
    Optionally uses:
        decayed_mfe_potential
        decayed_mae_risk
        survival_confidence_filtered
    """
    required = [
        "exit_direction_consistency",
        "survival_confidence",
        "volatility_after_entry_pred",
        "volatility_after_entry_z",
        "mae_risk_score",
        "ATR_14",
        "expected_exit_time",
        "exit_regime",
        "ATR_slope",
        "volatility_score",
        "path_entropy_score",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    # Hook for decayed fields
    if "decayed_mfe_potential" in df.columns:
        df["volatility_after_entry_pred"] = df["decayed_mfe_potential"]
    if "decayed_mae_risk" in df.columns:
        df["mae_risk_score"] = df["decayed_mae_risk"]
    if "survival_confidence_filtered" in df.columns:
        df["survival_confidence"] = df["survival_confidence_filtered"]

    # Reconstruct probabilities from exit_direction_consistency
    exit_cons = df["exit_direction_consistency"].astype(float).to_numpy()
    tp_prob_arr = np.clip((exit_cons + 1.0) / 2.0, 0.0, 1.0)
    sl_prob_arr = 1.0 - tp_prob_arr

    n = len(df)
    best_tp = np.zeros(n, dtype=np.float64)
    best_sl = np.zeros(n, dtype=np.float64)

    for i in range(n):
        atr = float(df["ATR_14"].iloc[i])
        exit_time = float(df["expected_exit_time"].iloc[i])
        path_entropy = float(df["path_entropy_score"].iloc[i])
        regime = int(df["exit_regime"].iloc[i]) if not pd.isna(df["exit_regime"].iloc[i]) else 0
        atr_slope = float(df["ATR_slope"].iloc[i])
        vol_score = float(df["volatility_score"].iloc[i])

        tp_grid = _tp_grid(atr, exit_time)
        sl_grid = _sl_grid(atr, path_entropy)

        tp_pct, sl_pct, _ = _advanced_expectancy(
            tp_prob=tp_prob_arr[i],
            sl_prob=sl_prob_arr[i],
            tp_grid=tp_grid,
            sl_grid=sl_grid,
            survival=float(df["survival_confidence"].iloc[i]),
            vol_after_entry_z=float(df["volatility_after_entry_z"].iloc[i]),
            mae_risk=float(df["mae_risk_score"].iloc[i]),
            atr_slope=atr_slope,
            regime=regime,
            volatility_score=vol_score,
        )

        best_tp[i] = tp_pct if atr > 0 else 1.0
        best_sl[i] = sl_pct if atr > 0 else 1.0

    expected_exit_time = np.maximum(df["expected_exit_time"].to_numpy(dtype=np.float64), 0.0)

    out = pd.DataFrame(
        {
            "recommended_tp_pct": best_tp.astype(np.float32),
            "recommended_sl_pct": best_sl.astype(np.float32),
            "expected_exit_time": expected_exit_time.astype(np.float32),
        },
        index=df.index,
    )

    if (out <= 0).any().any():
        raise ValueError("Non-positive TP/SL detected in optimisation output.")
    if out.isna().any().any():
        raise ValueError("NaNs detected in optimisation output.")
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError("Non-finite values detected in optimisation output.")

    return out
