"""
Decay and validity filtering for exit meta-signals with adaptive, volatility-aware logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def _zscore(arr: np.ndarray) -> np.ndarray:
    mean = np.nanmean(arr)
    std = np.nanstd(arr) + 1e-12
    return (arr - mean) / std


def apply_decay_and_validity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply adaptive decay and validity filtering to meta signals.

    Expects columns:
        mfe_potential_score, mae_risk_score, survival_confidence, exit_regime, regime,
        ATR_14, asymmetry_ratio_z, reward_risk_z, exit_direction_z,
        volatility_score, path_entropy_score.

    Returns columns:
        decayed_mfe_potential
        decayed_mae_risk
        survival_confidence_filtered
        integrity_score
        valid_signal_flag
        meta_exit_strength
        final_exit_strength
    """
    required = [
        "mfe_potential_score",
        "mae_risk_score",
        "survival_confidence",
        "exit_regime",
        "regime",
        "ATR_14",
        "asymmetry_ratio_z",
        "reward_risk_z",
        "exit_direction_z",
        "volatility_score",
        "path_entropy_score",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for decay/validity: {missing}")

    # Extract and clean to numpy
    arr = {col: np.nan_to_num(df[col].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0) for col in required}
    n = len(df)
    bars_elapsed = np.arange(n, dtype=np.float64)

    # Adaptive decay speeds
    vol_z = _zscore(arr["volatility_score"])
    decay_speed = 1.0 + _softplus(vol_z)
    mfe_decay = np.exp(-bars_elapsed / (25.0 / decay_speed))
    mae_decay = np.exp(-bars_elapsed / (8.0 / decay_speed))

    # Directional consistency boost
    exit_dir_z = np.clip(arr["exit_direction_z"], -1.0, 1.0)
    dir_consistency = pd.Series(exit_dir_z).rolling(window=5, min_periods=1).mean().to_numpy()
    dir_consistency = np.clip(dir_consistency, -1.0, 1.0)

    decayed_mfe = np.clip(arr["mfe_potential_score"] * mfe_decay, 0.0, 1.0)
    decayed_mfe *= (1.0 + 0.15 * dir_consistency)

    decayed_mae = np.clip(arr["mae_risk_score"] * mae_decay, 0.0, 1.0)

    # MAE panic spike handling
    atr = arr["ATR_14"]
    atr_series = pd.Series(atr)
    rolling_min = atr_series.rolling(window=20, min_periods=1).min().to_numpy()
    rolling_max = atr_series.rolling(window=20, min_periods=1).max().to_numpy()
    spike = atr > (rolling_max * 1.1)
    decayed_mae = np.where(spike, np.maximum(decayed_mae, arr["mae_risk_score"] * 1.2), decayed_mae)

    # Regime transition penalty
    regime_change = arr["regime"] != arr["exit_regime"]
    regime_transition_penalty = 1.0 - 0.4 * regime_change

    # Survival confidence filtering
    path_entropy_z = _zscore(arr["path_entropy_score"])
    survival = arr["survival_confidence"]
    survival_filtered = np.where(
        (vol_z > 2.5) | (np.abs(path_entropy_z) > 2.0),
        survival * 0.05,
        survival * (1 - _softplus(vol_z) / (1 + _softplus(vol_z))),
    )

    invalid = (vol_z > 2.5) | (np.abs(path_entropy_z) > 2.0) | regime_change
    valid_flag = (~invalid).astype(np.int8)

    # Integrity score
    integrity = (
        0.3 * arr["asymmetry_ratio_z"]
        + 0.3 * arr["reward_risk_z"]
        + 0.2 * exit_dir_z
        - 0.2 * path_entropy_z
    )
    integrity = np.clip(integrity, -4.0, 4.0)

    # Meta exit strength (legacy)
    meta_exit_strength = (
        0.35 * arr["asymmetry_ratio_z"]
        + 0.25 * arr["reward_risk_z"]
        + 0.20 * exit_dir_z
        + 0.20 * decayed_mfe
    )
    meta_exit_strength = np.clip(meta_exit_strength, -5.0, 5.0)

    # Final exit strength
    final_exit_strength = (
        0.30 * meta_exit_strength
        + 0.25 * survival_filtered
        + 0.20 * decayed_mfe
        - 0.10 * decayed_mae
        + 0.15 * integrity
    )
    final_exit_strength *= regime_transition_penalty
    final_exit_strength = np.clip(final_exit_strength, -6.0, 6.0)

    out = pd.DataFrame(
        {
            "decayed_mfe_potential": decayed_mfe.astype(np.float32),
            "decayed_mae_risk": decayed_mae.astype(np.float32),
            "survival_confidence_filtered": survival_filtered.astype(np.float32),
            "integrity_score": integrity.astype(np.float32),
            "valid_signal_flag": valid_flag.astype(np.int8),
            "meta_exit_strength": meta_exit_strength.astype(np.float32),
            "final_exit_strength": final_exit_strength.astype(np.float32),
        },
        index=df.index,
    )

    if out.isna().any().any():
        raise ValueError("NaNs detected in decay/validity output.")
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError("Non-finite values detected in decay/validity output.")

    return out
