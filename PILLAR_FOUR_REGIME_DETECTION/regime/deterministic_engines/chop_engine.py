"""
chop_engine.py
Detects choppy/rangy conditions via noise-to-signal and hurst.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine_utils import get_series, pct_rank


def compute_chop_engine(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    noise = get_series(df, "noise_to_signal", default=0.0012)
    hurst = get_series(df, "hurst_exponent", default=0.36)

    # Direction alternation proxy if price exists
    alt = pd.Series(0.0, index=df.index)
    price_col = "close" if "close" in df.columns else ("price" if "price" in df.columns else None)
    if price_col:
        returns = df[price_col].astype(float).diff().fillna(0.0)
        alt = (np.sign(returns).diff().abs() > 0).rolling(window=10, min_periods=5).mean().fillna(0.0)

    chop_strength = (0.7 * pct_rank(noise) + 0.3 * alt).clip(0, 1).fillna(0.0)
    chop_flag = ((noise > 0.0025) | (hurst < 0.25)).astype(int)

    out["chop_strength_score"] = chop_strength
    out["chop_flag"] = chop_flag
    return out
