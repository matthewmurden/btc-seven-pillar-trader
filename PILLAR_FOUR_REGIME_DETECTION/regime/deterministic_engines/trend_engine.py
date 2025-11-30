"""
Deterministic trend engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime.deterministic_engines.engine_utils import sigmoid, zscore


def apply_trend_engine(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "slope_5",
        "lr_grad_fast",
        "trend_strength",
        "higher_high_score",
        "higher_low_score",
        "htf_trend_5m",
        "htf_trend_15m",
        "htf_trend_1h",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for trend engine: {missing}")

    slope_norm = zscore(df["slope_5"].to_numpy(dtype=np.float64))
    strength_norm = zscore(df["trend_strength"].to_numpy(dtype=np.float64))
    htf_alignment = np.nanmean(
        np.column_stack(
            [
                np.sign(df["htf_trend_5m"]),
                np.sign(df["htf_trend_15m"]),
                np.sign(df["htf_trend_1h"]),
            ]
        ),
        axis=1,
    )

    hh = df["higher_high_score"].to_numpy(dtype=np.float64)
    hl = df["higher_low_score"].to_numpy(dtype=np.float64)

    score = sigmoid(
        0.6 * slope_norm
        + 0.6 * strength_norm
        + 0.4 * ((hh + hl) / 2.0)
        + 0.4 * htf_alignment
    )
    trend_flag = (score > 0.65).astype(np.float32)

    out = pd.DataFrame(
        {
            "trend_strength_score": score.astype(np.float32),
            "trend_flag": trend_flag,
        },
        index=df.index,
    )
    return out
