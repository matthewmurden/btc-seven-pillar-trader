"""Run full exit pipeline on test set (Parts 5-10)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_models.model_predict_testset import generate_test_predictions
from meta_features.meta_exit_engine import build_meta_features
from optimisation.exit_optimisation_engine import optimise_exit
from decay.decay_filter_engine import apply_decay_filters
from aggregator.exit_master_aggregator import build_exit_quality_score
from calibration.calibration_engine import ExitCalibrator
from exit_decision.decision_layer import FinalExitDecision
from backtesting.load_entry_signals import load_entry_signals_for_test


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TEST_PREDS_PATH = DATA_DIR / "exit_test_with_preds.parquet"
META_PATH = DATA_DIR / "meta_features.parquet"
OPT_PATH = DATA_DIR / "exit_optimisation.parquet"
DECAY_PATH = DATA_DIR / "exit_decay.parquet"
AGG_PATH = DATA_DIR / "exit_agg.parquet"
DECISION_PATH = DATA_DIR / "exit_decisions_test.parquet"


def run_exit_pipeline_test() -> Path:
    if not TEST_PREDS_PATH.exists():
        generate_test_predictions()
    df = pd.read_parquet(TEST_PREDS_PATH)

    # Attach entry columns if missing
    entry_cols = ["entry_signal", "direction", "entry_price", "entry_confidence"]
    if any(c not in df.columns for c in entry_cols):
        entry_df = load_entry_signals_for_test(len(df))
        for col in entry_cols:
            if col in entry_df.columns:
                df[col] = entry_df[col].values[: len(df)]
            else:
                # default fallbacks
                if col == "entry_signal":
                    df[col] = 1
                elif col == "direction":
                    df[col] = 1
                elif col == "entry_price":
                    df[col] = df.get("close", pd.Series(0, index=df.index))
                elif col == "entry_confidence":
                    df[col] = 1.0

    # Part 5
    df = build_meta_features(df)
    df.to_parquet(META_PATH)
    # Part 6
    df = optimise_exit(df)
    df.to_parquet(OPT_PATH)
    # Part 7
    df = apply_decay_filters(df)
    df.to_parquet(DECAY_PATH)
    # Part 8
    df = build_exit_quality_score(df)
    df.to_parquet(AGG_PATH)
    # Part 9 calibration
    calibrator = ExitCalibrator()
    # Use available labels if present; otherwise fall back to dummy labels
    if {"tp_first_label", "sl_first_label"}.issubset(df.columns):
        calib_df = df
    else:
        calib_df = df.copy()
        calib_df["tp_first_label"] = (calib_df["tp_first_pred"] > 0.5).astype(int)
        calib_df["sl_first_label"] = (calib_df["sl_first_pred"] > 0.5).astype(int)
    calibrator.fit(calib_df)
    df = calibrator.predict(df)

    # Part 10 decision
    decider = FinalExitDecision()
    df = decider.run_batch(df)
    # remove duplicate columns before saving
    df = df.loc[:, ~df.columns.duplicated(keep="last")]
    df.to_parquet(DECISION_PATH)
    return DECISION_PATH


if __name__ == "__main__":
    run_exit_pipeline_test()
