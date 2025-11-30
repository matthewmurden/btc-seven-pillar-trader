"""
Chop Engine
Detects sideways, low-directional, mean-reverting chop conditions.
"""

import pandas as pd
import numpy as np

def apply_chop_engine(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    entropy = out["micro_trend_entropy"]
    bar_cluster = out["bar_clustering_score"]
    alt_ratio = out["chop_index"]  # alternation of candle direction
    wick_ratio = (
        out["upper_wick_body"] + out["lower_wick_body"]
    ) / 2

    chop_score = (
        0.3 * entropy +
        0.25 * bar_cluster +
        0.25 * alt_ratio +
        0.2 * wick_ratio
    )

    chop_flag = (chop_score > 0.55).astype(int)

    out["chop_strength_score"] = chop_score.clip(0, 1)
    out["chop_flag"] = chop_flag

    return out