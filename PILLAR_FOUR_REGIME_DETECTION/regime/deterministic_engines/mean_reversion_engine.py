# ============================================
# mean_reversion_engine.py
# ============================================

import numpy as np
import pandas as pd
from .engine_utils import pct_rank


def compute_mean_reversion_engine(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    rev = df["reversion_score"]
    dist = df["mean_reversion_distance"]
    sweep = df["sweep_signal"]

    # Score
    out["mean_reversion_score"] = (
        0.5 * pct_rank(rev) +
        0.3 * pct_rank(dist) +
        0.2 * sweep
    ).fillna(0)

    out["mean_reversion_flag"] = (
        (rev > 0.45) &
        (dist > dist.quantile(0.75)) &
        (sweep == 1)
    ).astype(int)

    return out