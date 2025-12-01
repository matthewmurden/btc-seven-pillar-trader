# ============================================
# liquidity_engine.py
# ============================================

import numpy as np
import pandas as pd
from .engine_utils import pct_rank


def compute_liquidity_engine(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    vol = df["volume"]
    orderflow = df["orderflow_score"]
    delta_int = df["delta_intensity"]

    # Reliably identifies thin markets in your dataset
    out["liquidity_score"] = (
        -0.4 * pct_rank(vol) +
        -0.4 * pct_rank(orderflow) +
        -0.2 * pct_rank(delta_int)
    ).fillna(0)

    reg = np.full(len(df), 1, dtype=np.int8)  # normal
    reg[(vol < vol.quantile(0.25)) & (orderflow < -0.4)] = 0  # drought
    reg[(vol > vol.quantile(0.75)) & (delta_int > 0.6)] = 2  # expansion (optional)

    out["liquidity_regime"] = reg
    return out