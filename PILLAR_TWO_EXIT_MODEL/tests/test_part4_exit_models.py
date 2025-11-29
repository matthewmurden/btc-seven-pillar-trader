"""Tests for Part 4 exit ML models."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.build_exit_training_set import build_exit_training_dataset  # noqa: E402
from ml_models.model_training_exit_master import train_all_exit_models  # noqa: E402
from ml_models.model_prediction_exit_master import ExitModelPredictionMaster  # noqa: E402


TRAINING_PATH = PROJECT_ROOT / "data" / "processed" / "exit_training.parquet"

PRED_COLS = [
    "mae_pred",
    "mfe_pred",
    "exit_time_pred",
    "optimal_tp_pred",
    "optimal_sl_pred",
    "tp_first_pred",
    "sl_first_pred",
]


def main() -> None:
    # Ensure training dataset exists
    build_exit_training_dataset()
    train_all_exit_models()

    df = pd.read_parquet(TRAINING_PATH)
    predictor = ExitModelPredictionMaster()
    out = predictor.run_all(df, save=False)

    if out.shape[0] != df.shape[0]:
        raise AssertionError("Prediction output length mismatch")

    for col in PRED_COLS:
        if col not in out.columns:
            raise AssertionError(f"Missing prediction column: {col}")
        series = out[col]
        if series.isna().any():
            raise AssertionError(f"NaNs found in prediction column {col}")

    print("Part 4 exit models test passed")
    print(out[PRED_COLS].head())


if __name__ == "__main__":
    main()
