# ===========================================================
# fusion_rules_deterministic.py
# Deterministic-only regime fusion logic (Part 4A)
# ===========================================================

import pandas as pd

# Final deterministic regime mapping
REGIME_MAP = {
    0: "trend",
    1: "range",
    2: "chop",
    3: "vol_expansion",
    4: "vol_compression",
    5: "liquidity_drought",
    7: "mean_reversion"
}


def fuse_deterministic_row(row):
    """
    Fusion priority (deterministic-only):
        1) Liquidity drought
        2) Volatility expansion
        3) Volatility compression
        4) Trend
        5) Chop
        6) Mean reversion
        7) Else → Range
    """

    # 1. Liquidity drought
    if row["liquidity_regime"] == 0:
        return 5

    # 2. Vol expansion
    if row["volatility_regime"] == 2:
        return 3

    # 3. Vol compression
    if row["volatility_regime"] == 0:
        return 4

    # 4. Trend
    if row["trend_flag"] == 1:
        return 0

    # 5. Chop
    if row["chop_flag"] == 1:
        return 2

    # 6. Mean reversion
    if row["mean_reversion_flag"] == 1:
        return 7

    # 7. Default: range
    return 1


def fuse_deterministic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["regime_label"] = df.apply(fuse_deterministic_row, axis=1)
    return df