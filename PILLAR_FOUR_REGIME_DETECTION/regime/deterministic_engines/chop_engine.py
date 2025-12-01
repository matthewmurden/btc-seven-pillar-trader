# ============================================
# chop_engine.py
# ============================================

import numpy as np
import pandas as pd
from .engine_utils import pct_rank


def compute_chop_engine(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    noise = df["noise_to_signal"]
    chop_index = df["chop_index"]
    trend_s = df["trend_score"]

    # Chop strength: more noise + bar alternation, less trend
    out["chop_strength_score"] = (
        0.5 * pct_rank(noise) +
        0.4 * pct_rank(chop_index) +
        0.1 * (1 - pct_rank(abs(trend_s)))
    ).fillna(0)

    # Flag
    out["chop_flag"] = (
        (noise > noise.quantile(0.75)) |
        (chop_index > 0.5) |
        ((abs(trend_s) < 0.1) & (noise > noise.median()))
    ).astype(int)

    return out