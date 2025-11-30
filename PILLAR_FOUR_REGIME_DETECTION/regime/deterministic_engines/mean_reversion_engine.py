"""
Deterministic mean reversion engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime.deterministic_engines.engine_utils import sigmoid, zscore


def apply_mean_reversion_engine(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "sweep_flags",
        "mean_reversion_distance",
        "noise_to_signal",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for mean reversion engine: {missing}")

    sweep = df["sweep_flags"].to_numpy(dtype=np.float64)
    mrd_norm = zscore(df["mean_reversion_distance"].to_numpy(dtype=np.float64))
    nts_norm = zscore(df["noise_to_signal"].to_numpy(dtype=np.float64))

    mr_score = sigmoid(
        sweep
        + np.abs(mrd_norm)
        + nts_norm
    )
    mr_flag = (mr_score > 0.55).astype(np.float32)

    return pd.DataFrame(
        {
            "mean_reversion_score": mr_score.astype(np.float32),
            "mean_reversion_flag": mr_flag,
        },
        index=df.index,
    )
