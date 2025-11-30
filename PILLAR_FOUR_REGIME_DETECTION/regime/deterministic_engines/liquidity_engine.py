"""
Liquidity Engine
Detects liquidity drought/normal/expansion states using orderflow proxies.
"""

import pandas as pd
import numpy as np

def apply_liquidity_engine(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    delta = out["delta_intensity"]
    imb = out["imbalance_persistence"]
    cvd = out["cvd_slope"]
    vol_exp = out["volume_expansion_20"]

    drought_score = (
        0.4 * (1 - delta) +
        0.3 * (1 - imb) +
        0.2 * (1 - cvd) +
        0.1 * (1 - vol_exp)
    )

    # Liquidity regime
    # 0 = drought
    # 1 = normal
    # 2 = expansion
    regime = np.where(vol_exp > 0.6, 2,
                np.where(drought_score > 0.6, 0, 1))

    out["liquidity_score"] = 1 - drought_score.clip(0, 1)
    out["liquidity_regime"] = regime.astype(int)

    return out