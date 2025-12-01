"""
engine_utils.py
Utility helpers + deterministic engine orchestrator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pct_rank(x: pd.Series) -> pd.Series:
    """Percentile rank with neutral fill."""
    return x.rank(pct=True).fillna(0.5).clip(0, 1)


def normalize(x: pd.Series) -> pd.Series:
    """
    Z-score then squash with tanh to keep outputs bounded to [-1, 1].
    Falls back to zeros on degenerate input.
    """
    x = x.astype(float)
    mu = x.mean()
    sd = x.std()
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=x.index)
    z = (x - mu) / sd
    return np.tanh(z).fillna(0.0)


def get_series(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    """Fetch a column by name or return a constant-filled series."""
    if name in df.columns:
        return df[name].astype(float).fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def run_all_engines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute all deterministic engines and return a unified feature frame.
    Each engine returns only its own columns; we concat them.
    """
    from .trend_engine import compute_trend_engine
    from .chop_engine import compute_chop_engine
    from .volatility_engine import compute_volatility_engine
    from .liquidity_engine import compute_liquidity_engine
    from .mean_reversion_engine import compute_mean_reversion_engine

    outputs = [
        compute_trend_engine(df),
        compute_chop_engine(df),
        compute_volatility_engine(df),
        compute_liquidity_engine(df),
        compute_mean_reversion_engine(df),
    ]

    combined = pd.concat(outputs, axis=1)
    return combined.fillna(0)
