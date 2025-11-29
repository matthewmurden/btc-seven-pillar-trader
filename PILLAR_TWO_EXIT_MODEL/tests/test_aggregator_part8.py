"""Tests for Part 8 exit master aggregator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aggregator.exit_master_aggregator import build_exit_quality_score  # noqa: E402


def dummy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asymmetry_ratio": [0.5, 1.0, 2.0],
            "reward_risk_prediction": [1.0, 1.5, 2.0],
            "survival_confidence_decayed": [0.5, 0.6, 0.7],
            "volatility_after_entry_pred": [0.2, 0.3, 0.4],
            "exit_direction_consistency": [0.1, 0.2, 0.3],
            "expected_exit_time": [10, 15, 20],
            "valid_exit_signal": [1, 1, 0],
            "momentum_score": [0.1, 0.2, 0.3],
            "trend_score": [0.2, 0.3, 0.4],
            "volatility_score": [0.3, 0.4, 0.5],
            "structure_score": [0.4, 0.5, 0.6],
            "entry_probability": [0.6, 0.7, 0.8],
        }
    )


def main() -> None:
    df = dummy_df()
    out = build_exit_quality_score(df)

    if "exit_quality_score" not in out.columns:
        raise AssertionError("Missing exit_quality_score")

    if out["exit_quality_score"].isna().any():
        raise AssertionError("NaNs in exit_quality_score")

    if not ((out["exit_quality_score"] >= 0).all() and (out["exit_quality_score"] <= 1).all()):
        raise AssertionError("exit_quality_score out of bounds")

    # validity mask
    if out.loc[2, "exit_quality_score"] != 0:
        raise AssertionError("valid_exit_signal not applied")

    # monotonic: increasing asymmetry should lift score
    if out.loc[2, "exit_quality_score"] < out.loc[0, "exit_quality_score"]:
        raise AssertionError("Score not monotonic with asymmetry")

    print("Part 8 aggregator test passed")
    print(out[["exit_quality_score"]].head())


if __name__ == "__main__":
    main()
