"""
Deterministic liquidity engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime.deterministic_engines.engine_utils import sigmoid, zscore


def apply_liquidity_engine(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "delta_intensity",
        "imbalance_persistence",
        "volume_expansion_5",
        "volume_expansion_20",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for liquidity engine: {missing}")

    delta_norm = zscore(df["delta_intensity"].to_numpy(dtype=np.float64))
    imbalance_norm = zscore(df["imbalance_persistence"].to_numpy(dtype=np.float64))
    ve5_norm = zscore(df["volume_expansion_5"].to_numpy(dtype=np.float64))
    ve20_norm = zscore(df["volume_expansion_20"].to_numpy(dtype=np.float64))

    liquidity_score = sigmoid(
        -delta_norm
        + imbalance_norm
        + 0.5 * (ve5_norm + ve20_norm)
    )

    regime = np.full(len(df), 1, dtype=np.int8)
    drought = df["volume_expansion_20"] < 0.5
    expansion = df["volume_expansion_20"] > 1.3
    regime = np.where(drought, 0, regime)
    regime = np.where(expansion, 2, regime)

    return pd.DataFrame(
        {
            "liquidity_score": liquidity_score.astype(np.float32),
            "liquidity_regime": regime.astype(np.int8),
        },
        index=df.index,
    )
