"""Tests for Part 7 decay filters."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decay.decay_filter_engine import apply_decay_filters  # noqa: E402


def dummy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mfe_potential_score": [0.5, 0.6, 0.4, 0.3],
            "mae_risk_score": [0.2, 0.1, 0.3, 0.05],
            "survival_confidence": [0.8, 0.7, 0.6, 0.5],
            "exit_regime": [0, 0, 1, 1],
            "momentum_score": [0.1, 0.2, 0.3, 0.4],
            "trend_score": [0.1, 0.2, 0.3, 0.4],
            "volatility_score": [0.1, 0.2, 0.3, 0.4],
            "structure_score": [0.1, 0.2, 0.3, 0.4],
        }
    )


def main() -> None:
    df = dummy_df()
    out = apply_decay_filters(df)

    expected_cols = [
        "mfe_potential_score_decayed",
        "mae_risk_score_decayed",
        "survival_confidence_decayed",
        "valid_exit_signal",
    ]
    for col in expected_cols:
        if col not in out.columns:
            raise AssertionError(f"Missing {col}")
        if out[col].isna().any():
            raise AssertionError(f"NaNs found in {col}")

    # validity should go to 0 if all decayed terms zero
    zero_df = df.copy()
    zero_df[["mfe_potential_score", "mae_risk_score", "survival_confidence"]] = 0
    zero_out = apply_decay_filters(zero_df)
    if zero_out["valid_exit_signal"].sum() != 0:
        raise AssertionError("Validity flag not zeroed when all signals are zero")

    # regime change should zero survival at that bar
    if zero_out.loc[1, "survival_confidence_decayed"] != 0:
        raise AssertionError("Regime change did not zero survival_confidence_decayed")

    print("Part 7 decay filters test passed")
    print(out[expected_cols].head())


if __name__ == "__main__":
    main()
