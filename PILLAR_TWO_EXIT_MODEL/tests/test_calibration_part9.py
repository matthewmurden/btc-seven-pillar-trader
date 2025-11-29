"""Tests for Part 9 calibration layer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration.calibration_engine import ExitCalibrator  # noqa: E402


def dummy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tp_first_pred": [0.2, 0.4, 0.6, 0.8],
            "sl_first_pred": [0.8, 0.6, 0.4, 0.2],
            "exit_quality_score": [0.3, 0.5, 0.7, 0.9],
            "tp_first_label": [0, 0, 1, 1],
            "sl_first_label": [1, 1, 0, 0],
        }
    )


def main() -> None:
    df = dummy_df()
    calib = ExitCalibrator()
    calib.fit(df)
    out = calib.predict(df)

    for col in ["tp_first_calibrated", "sl_first_calibrated", "exit_quality_calibrated"]:
        if col not in out.columns:
            raise AssertionError(f"Missing calibrated column {col}")
        vals = out[col].to_numpy()
        if not ((vals >= 0).all() and (vals <= 1).all()):
            raise AssertionError(f"Calibrated values out of [0,1] in {col}")

    # Temperature scaling should preserve ordering
    if not np.array_equal(
        np.argsort(df["exit_quality_score"].to_numpy()),
        np.argsort(out["exit_quality_calibrated"].to_numpy()),
    ):
        raise AssertionError("Temperature scaling changed ordering")

    # Isotonic should be monotone and likely alter shape
    if np.allclose(out["tp_first_calibrated"], df["tp_first_pred"]):
        raise AssertionError("Isotonic calibration did not change tp_first_pred")

    print("Part 9 calibration test passed")
    print(out[["tp_first_calibrated", "sl_first_calibrated", "exit_quality_calibrated"]].head())


if __name__ == "__main__":
    main()
