"""Exit master aggregator (Part 8)."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLS = [
    "asymmetry_ratio",
    "reward_risk_prediction",
    "survival_confidence_decayed",
    "volatility_after_entry_pred",
    "exit_direction_consistency",
    "expected_exit_time",
    "valid_exit_signal",
    "momentum_score",
    "trend_score",
    "volatility_score",
    "structure_score",
    "entry_probability",
]


def _minmax(series: pd.Series) -> pd.Series:
    s_min = series.min()
    s_max = series.max()
    if np.isclose(s_max, s_min):
        return pd.Series(0.0, index=series.index)
    return (series - s_min) / (s_max - s_min)


def build_exit_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for exit quality aggregation: {missing}")

    out = df.copy()
    asym = _minmax(out["asymmetry_ratio"])
    rr = _minmax(out["reward_risk_prediction"])
    surv = _minmax(out["survival_confidence_decayed"])
    dir_cons = _minmax(out["exit_direction_consistency"])
    vol_after = _minmax(out["volatility_after_entry_pred"])
    entry_prob = _minmax(out["entry_probability"])
    mom = _minmax(out["momentum_score"])
    trend = _minmax(out["trend_score"])
    vol = _minmax(out["volatility_score"])
    struct = _minmax(out["structure_score"])

    expectancy_strength = out["asymmetry_ratio"] * out["reward_risk_prediction"]
    expectancy_strength_norm = _minmax(expectancy_strength)

    raw_score = (
        0.30 * expectancy_strength_norm
        + 0.20 * surv
        + 0.15 * dir_cons
        + 0.10 * vol_after
        + 0.10 * entry_prob
        + 0.05 * mom
        + 0.05 * trend
        + 0.03 * vol
        + 0.02 * struct
    )

    raw_score = raw_score.clip(0, 1)
    out["exit_quality_score"] = raw_score * out["valid_exit_signal"]
    return out
