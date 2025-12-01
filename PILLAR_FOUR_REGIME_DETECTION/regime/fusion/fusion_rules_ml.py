# ===========================================================
# fusion_rules_ml.py
# ML + deterministic fusion logic (Part 4B)
# ===========================================================

import numpy as np

# Deterministic mapping
DET_MAP = {
    0: "trend",
    1: "range",
    2: "chop",
    3: "vol_expansion",
    4: "vol_compression",
    5: "liquidity_drought",
    7: "mean_reversion",
}


def fuse_row_ml(row):
    """
    Decision logic combining ML + deterministic.
    Priority:
        1. Liquidity drought always overrides.
        2. Vol expansion overrides other states.
        3. If ML confidence low → fallback to deterministic.
        4. Else → ML regime.
    """

    ml_pred = int(row["ml_regime"])
    ml_conf = float(row.get("ml_confidence", 0.0))
    det_pred = int(row["det_regime"])

    # Liquidity drought always overrides
    # Note: liquidity_regime=1 indicates drought in current engine logic.
    if row.get("liquidity_regime", 0) == 1:
        return 5

    # Vol expansion overrides
    if row.get("volatility_regime", 1) == 2:
        return 3

    # Low ML confidence → deterministic fallback
    if ml_conf < 0.60:
        return det_pred

    # Otherwise → ML prediction
    return ml_pred
