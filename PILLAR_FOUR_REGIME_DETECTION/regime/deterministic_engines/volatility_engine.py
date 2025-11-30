"""
Volatility Regime Engine
Classifies compression → normal → expansion using multi-horizon vol curve.
"""

import pandas as pd
import numpy as np

def apply_volatility_engine(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    vol1 = out["realized_vol_1"]
    vol5 = out["realized_vol_5"]
    vol15 = out["realized_vol_15"]
    vol30 = out["realized_vol_30"]

    compression = out["vol_compression_score"]
    expansion = out["vol_expansion_score"]
    atrp = out["atr_percentile"]

    # Composite volatility level
    base_vol = (vol1 + vol5 + vol15 + vol30) / 4

    # Score: expansion > normal > compression
    vol_score = (
        0.4 * base_vol +
        0.3 * expansion +
        0.3 * atrp
    )

    # Discrete volatility regime
    # 0 = compression
    # 1 = normal
    # 2 = expansion
    regime = np.where(expansion > 0.55, 2,
                np.where(compression > 0.55, 0, 1))

    out["volatility_regime_score"] = vol_score.clip(0, 1)
    out["volatility_regime"] = regime.astype(int)

    return out