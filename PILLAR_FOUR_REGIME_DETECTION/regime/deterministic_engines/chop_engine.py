"""
Deterministic chop engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime.deterministic_engines.engine_utils import sigmoid, zscore


def apply_chop_engine(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "micro_trend_entropy",
        "bar_clustering_score",
        "chop_index",
        "upper_wick_body",
        "lower_wick_body",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for chop engine: {missing}")

    entropy_norm = zscore(df["micro_trend_entropy"].to_numpy(dtype=np.float64))
    chop_norm = zscore(df["chop_index"].to_numpy(dtype=np.float64))
    uwb = df["upper_wick_body"].to_numpy(dtype=np.float64)
    lwb = df["lower_wick_body"].to_numpy(dtype=np.float64)

    score = sigmoid(
        entropy_norm
        + chop_norm
        + 0.5 * (uwb + lwb)
    )
    chop_flag = (score > 0.6).astype(np.float32)

    return pd.DataFrame(
        {
            "chop_strength_score": score.astype(np.float32),
            "chop_flag": chop_flag,
        },
        index=df.index,
    )
