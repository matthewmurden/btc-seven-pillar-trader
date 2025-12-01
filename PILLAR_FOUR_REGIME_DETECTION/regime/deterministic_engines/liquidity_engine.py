"""
liquidity_engine.py
Detects liquidity drought/expansion via volume intensity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine_utils import get_series, normalize


def compute_liquidity_engine(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    volume = None
    for col in ("volume", "vol"):
        if col in df.columns:
            volume = df[col].astype(float)
            break
    if volume is None:
        volume = pd.Series(0.0, index=df.index)

    delta_intensity = get_series(df, "delta_intensity", default=0.0012)

    rolling_vol_mean = volume.rolling(window=20, min_periods=10).mean().fillna(volume.mean())
    volume_expansion_ratio = (volume / rolling_vol_mean).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    regime = np.full(len(df), 0, dtype=np.int8)  # 0 = normal
    regime = np.where(volume < rolling_vol_mean * 0.6, 1, regime)  # drought
    regime = np.where(volume > rolling_vol_mean * 1.3, 2, regime)  # expansion

    liquidity_score = normalize(volume_expansion_ratio - 1.0).clip(-1, 1).fillna(0.0)

    out["liquidity_regime"] = pd.Series(regime, index=df.index, dtype=int)
    out["liquidity_score"] = liquidity_score
    return out
