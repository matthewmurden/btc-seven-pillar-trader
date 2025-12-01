"""
volatility_engine.py
Classifies compression/normal/expansion from realized volatility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine_utils import get_series, normalize


def _fallback_realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """Fallback realized vol from rolling std of returns."""
    returns = close.astype(float).pct_change().fillna(0.0)
    return returns.rolling(window=window, min_periods=max(5, window // 2)).std()


def _get_realized(df: pd.DataFrame, candidates: list[str], close: pd.Series | None) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return df[col].astype(float)
    if close is not None:
        return _fallback_realized_vol(close)
    return pd.Series(0.0, index=df.index)


def compute_volatility_engine(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    price_col = "close" if "close" in df.columns else ("price" if "price" in df.columns else None)
    close = df[price_col] if price_col else None

    # Prefer the populated "volatility" column; fall back if it is missing or flat
    if "volatility" in df.columns and df["volatility"].std() > 0:
        rv1 = df["volatility"].astype(float)
    else:
        rv1 = _get_realized(df, ["realized_vol_1", "realized_vol_x", "realized_vol"], close)

    rv5 = _get_realized(df, ["realized_vol_5"], close).fillna(rv1.rolling(5, min_periods=3).mean())
    rv15 = _get_realized(df, ["realized_vol_15"], close).fillna(rv1.rolling(15, min_periods=5).mean())

    breakout_flag = get_series(df, "volatility_breakout_flag", default=0.0)
    vol_percentile = get_series(df, "vol_percentile", default=np.nan)
    range_comp = get_series(df, "range_compression", default=0.0)

    compression = ((rv1 < 0.001) & (range_comp > 0.003)).astype(int)
    expansion = ((rv1 > 0.002) | (breakout_flag == 1) | (vol_percentile > 0.8)).astype(int)

    volatility_regime = np.where(expansion == 1, 2, 1)  # default normal
    volatility_regime = np.where(compression == 1, 0, volatility_regime)

    # If rv1 is flat, try using short/medium realized vols to derive a gradient
    if rv1.std() == 0 and (rv5.std() > 0 or rv15.std() > 0):
        rv1 = rv5.combine_first(rv15)

    vol_score = normalize(rv1).clip(-1, 1).fillna(0.0)

    out["volatility_regime"] = pd.Series(volatility_regime, index=df.index, dtype=int)
    out["volatility_score"] = vol_score
    return out
