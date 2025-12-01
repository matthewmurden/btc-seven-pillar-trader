# ============================================
# volatility_engine.py
# ============================================

import numpy as np
import pandas as pd
from .engine_utils import pct_rank


def compute_volatility_engine(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    vol_s  = df["volatility_score"]
    atr_exp = df["atr_expansion"]
    bb_w = df["bb_width"]

    # Smooth combined score
    out["volatility_regime_score"] = (
        0.5 * pct_rank(vol_s) +
        0.3 * pct_rank(atr_exp) +
        0.2 * pct_rank(bb_w)
    ).fillna(0)

    # Multi-state regime: compression, normal, expansion
    reg = np.full(len(df), 1, dtype=np.int8)  # default normal

    reg[(vol_s > 0.55) & (atr_exp > 0.35) & (bb_w > bb_w.quantile(0.70))] = 2  # expansion
    reg[(vol_s < -0.35) & (bb_w < bb_w.quantile(0.20))] = 0  # compression

    out["volatility_regime"] = reg
    return out