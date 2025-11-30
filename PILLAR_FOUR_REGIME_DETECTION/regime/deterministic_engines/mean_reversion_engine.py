"""
Mean Reversion Engine
Detects pullback → reversion conditions using sweep flags,
bias distance, and reversion intensity.
"""

import pandas as pd
import numpy as np

def apply_mean_reversion_engine(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    sweeps = out["sweep_flags"]
    dist = out["mean_reversion_distance"]
    tight = out["range_tightness_score"]

    mr_score = (
        0.5 * sweeps +
        0.3 * dist +
        0.2 * tight
    )

    mr_flag = (mr_score > 0.55).astype(int)

    out["mean_reversion_score"] = mr_score.clip(0, 1)
    out["mean_reversion_flag"] = mr_flag

    return out