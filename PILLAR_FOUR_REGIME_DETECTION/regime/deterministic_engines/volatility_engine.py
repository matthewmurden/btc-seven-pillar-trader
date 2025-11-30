"""
Deterministic volatility regime engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime.deterministic_engines.engine_utils import sigmoid, zscore


def apply_volatility_engine(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "realized_vol_5",
        "realized_vol_15",
        "realized_vol_30",
        "atr_percentile",
        "vol_compression_score",
        "vol_expansion_score",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for volatility engine: {missing}")

    vol_exp_norm = zscore(df["vol_expansion_score"].to_numpy(dtype=np.float64))
    vol_comp_norm = zscore(df["vol_compression_score"].to_numpy(dtype=np.float64))
    atr_pct = df["atr_percentile"].to_numpy(dtype=np.float64)

    vol_score = sigmoid(
        0.5 * vol_exp_norm
        - 0.5 * vol_comp_norm
        + atr_pct
    )

    vol_regime = np.full(len(df), 1, dtype=np.int8)
    comp_mask = (df["vol_compression_score"] > 1.5) & (df["atr_percentile"] < 0.3)
    exp_mask = (df["vol_expansion_score"] > 1.0) | (df["atr_percentile"] > 0.7)
    vol_regime = np.where(comp_mask, 0, vol_regime)
    vol_regime = np.where(exp_mask, 2, vol_regime)

    return pd.DataFrame(
        {
            "volatility_regime_score": vol_score.astype(np.float32),
            "volatility_regime": vol_regime.astype(np.int8),
        },
        index=df.index,
    )
