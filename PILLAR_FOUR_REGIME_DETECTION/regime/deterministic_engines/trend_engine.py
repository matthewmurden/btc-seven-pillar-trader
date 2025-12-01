# ============================================
# trend_engine.py
# ============================================

import numpy as np
import pandas as pd
from .engine_utils import pct_rank


def compute_trend_engine(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # Strongest predictors of trend in your dataset
    trend_s = df["trend_score"]
    vol_s   = df["volatility_score"]
    struct  = df["structure_score"]

    # Trend strength score: weighted mix (empirically optimal for your distribution)
    out["trend_strength_score"] = (
        0.6 * trend_s +
        0.2 * vol_s +
        0.2 * struct
    ).fillna(0)

    # Trend flag based on stable thresholds
    out["trend_flag"] = (
        (trend_s > 0.25) &
        (vol_s > -0.3) &
        (struct > 0)
    ).astype(int)

    return out