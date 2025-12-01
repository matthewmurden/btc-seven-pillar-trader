"""
mean_reversion_engine.py
Flags sweep/overextension events for mean reversion setups.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine_utils import get_series, normalize


def compute_mean_reversion_engine(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    sweep_cols = [c for c in ("sweep_flag", "sweep_signal", "sweep_up", "sweep_down") if c in df.columns]
    if sweep_cols:
        sweep_flag = (df[sweep_cols].astype(float).sum(axis=1) > 0).astype(int)
    else:
        sweep_flag = pd.Series(0, index=df.index, dtype=int)

    dist = get_series(df, "mean_reversion_distance", default=0.0)

    # Deviation from VWAP/mean as additional overextension proxy
    deviation = get_series(df, "deviation_from_mean", default=0.0)
    price_col = "close" if "close" in df.columns else ("price" if "price" in df.columns else None)
    if deviation.eq(0).all() and price_col:
        price = df[price_col].astype(float)
        if "vwap" in df.columns:
            vwap = df["vwap"].astype(float).replace(0, np.nan)
            deviation = ((price - vwap) / vwap).abs()
        else:
            rolling_mean = price.rolling(window=50, min_periods=10).mean()
            deviation = (price - rolling_mean).abs() / rolling_mean.clip(lower=1e-9)
        deviation = deviation.fillna(0.0)

    mean_reversion_flag = ((sweep_flag == 1) | (dist > 500) | (deviation > 0.003)).astype(int)
    raw_score = np.log1p(dist.clip(lower=0)) + 0.5 * normalize(deviation).abs()
    mean_reversion_score = normalize(raw_score).clip(-1, 1).fillna(0.0)

    out["mean_reversion_flag"] = mean_reversion_flag
    out["mean_reversion_score"] = mean_reversion_score
    return out
