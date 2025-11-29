"""Tests for Part 5 meta exit features."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from meta_features.meta_exit_engine import build_meta_features, META_COLS, OUTPUT_PATH  # noqa: E402


def _sample_df() -> pd.DataFrame:
    data = {
        "mae_pred": [-0.01, -0.02, -0.005],
        "mfe_pred": [0.015, 0.03, 0.01],
        "optimal_tp_pred": [0.02, 0.04, 0.015],
        "optimal_sl_pred": [-0.01, -0.015, -0.008],
        "tp_first_pred": [0.6, 0.7, 0.5],
        "sl_first_pred": [0.4, 0.3, 0.5],
        "exit_time_pred": [10, 20, 5],
        "volatility_score": [0.2, 0.5, 0.8],
        "regime_state": [0, 1, 2],
    }
    return pd.DataFrame(data)


def main() -> None:
    df = _sample_df()
    out = build_meta_features(df)

    missing = [c for c in META_COLS if c not in out.columns]
    if missing:
        raise AssertionError(f"Missing meta columns: {missing}")

    if len(out) != len(df):
        raise AssertionError("Row count changed in meta features")

    # survival_confidence bounds
    surv = out["survival_confidence"]
    if not ((surv >= 0).all() and (surv <= 1).all()):
        raise AssertionError("survival_confidence out of bounds")

    # asymmetry/reward risk finite
    for col in ("asymmetry_ratio", "reward_risk_prediction"):
        vals = out[col].values
        if np.isnan(vals).any() or np.isinf(vals).any():
            raise AssertionError(f"{col} contains NaN/inf")

    # direction consistency
    if not np.allclose(
        out["exit_direction_consistency"],
        df["tp_first_pred"] - df["sl_first_pred"],
    ):
        raise AssertionError("exit_direction_consistency incorrect")

    if not OUTPUT_PATH.exists():
        raise AssertionError("Meta features parquet not written")

    reloaded = pd.read_parquet(OUTPUT_PATH)
    if len(reloaded) == 0:
        raise AssertionError("Reloaded meta features empty")

    print("Part 5 meta features test passed")
    print(out[META_COLS].head())


if __name__ == "__main__":
    main()
