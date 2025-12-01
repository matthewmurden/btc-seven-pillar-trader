# ============================================
# engine_utils.py
# Utility functions only — NO ENGINE IMPORTS
# ============================================

import pandas as pd

def pct_rank(x: pd.Series) -> pd.Series:
    """Percentile-rank a series (0–1)."""
    return x.rank(pct=True).fillna(0.5)


def normalize(x: pd.Series) -> pd.Series:
    """Z-score normalization with fallback."""
    mu = x.mean()
    sd = x.std()
    if sd == 0:
        return pd.Series(0, index=x.index)
    return (x - mu) / sd