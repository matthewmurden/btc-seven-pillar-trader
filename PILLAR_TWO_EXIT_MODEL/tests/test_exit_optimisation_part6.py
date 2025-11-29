"""Tests for Part 6 exit optimisation engine."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimisation.exit_optimisation_engine import optimise_exit, OUTPUT_PATH  # noqa: E402


def synthetic_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "atr": [100, 100, 100],
            "tp_first_pred": [0.6, 0.6, 0.6],
            "sl_first_pred": [0.4, 0.4, 0.4],
            "survival_confidence": [0.9, 0.9, 0.9],
            "volatility_score": [0.2, 0.9, 0.2],
            "regime_state": [0, 1, 2],
            "exit_time_pred": [10, 10, 10],
        }
    )


def main() -> None:
    df = synthetic_df()
    out = optimise_exit(df)

    if len(out) != len(df):
        raise AssertionError("Row count changed in optimisation")

    for col in ["recommended_tp_pct", "recommended_sl_pct"]:
        if col not in out.columns:
            raise AssertionError(f"Missing column {col}")
        if out[col].isna().any():
            raise AssertionError(f"NaNs in {col}")
        if not (out[col] > 0).all():
            raise AssertionError(f"Non-positive values in {col}")

    tp_grid = np.linspace(0.5, 4.0, 20) * df["atr"].iloc[0]
    sl_grid = np.linspace(0.5, 3.0, 20) * df["atr"].iloc[0]

    if not out["recommended_tp_pct"].between(tp_grid.min(), tp_grid.max()).all():
        raise AssertionError("TP outside grid bounds")
    if not out["recommended_sl_pct"].between(sl_grid.min(), sl_grid.max()).all():
        raise AssertionError("SL outside grid bounds")

    # Trend regime should lift TP vs range
    if out.loc[1, "recommended_tp_pct"] <= out.loc[0, "recommended_tp_pct"]:
        raise AssertionError("Trend regime did not increase TP as expected")
    # Range regime should reduce TP vs trend
    if out.loc[2, "recommended_tp_pct"] >= out.loc[1, "recommended_tp_pct"]:
        raise AssertionError("Range/breakout adjustment not applied as expected")
    # High vol increases SL (penalty encourages cautious sizing)
    if out.loc[1, "recommended_sl_pct"] <= out.loc[0, "recommended_sl_pct"]:
        raise AssertionError("High volatility did not increase SL")

    if not OUTPUT_PATH.exists():
        raise AssertionError("Optimisation parquet not written")
    reloaded = pd.read_parquet(OUTPUT_PATH)
    if len(reloaded) == 0:
        raise AssertionError("Reloaded optimisation parquet empty")

    print("Part 6 exit optimisation test passed")
    print(out[["recommended_tp_pct", "recommended_sl_pct"]].head())


if __name__ == "__main__":
    main()
