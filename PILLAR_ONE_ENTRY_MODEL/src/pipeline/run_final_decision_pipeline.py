"""
Pipeline to apply calibration, consensus fail-safe, and final decision.
Input: data/processed/features_with_raw_confidence.parquet
Output: data/processed/features_with_final_signal.parquet
"""

from pathlib import Path

import pandas as pd

from src.calibration.calibration_layer import CalibrationLayer
from src.calibration.calibration_config import CalibrationConfig
from src.failsafe.consensus_failsafe import apply_consensus_failsafe
from src.failsafe.consensus_failsafe_config import DEFAULT_CONSENSUS_FAILSAFE_CONFIG
from src.decision.final_decision_layer import compute_final_signal
from src.decision.final_decision_config import DEFAULT_FINAL_DECISION_CONFIG


RAW_PATH = Path("data/processed/features_with_raw_confidence.parquet")
CALIBRATED_PATH = Path("data/processed/features_with_entry_probability.parquet")
FAILSAFE_PATH = Path("data/processed/features_with_failsafe_probability.parquet")
FINAL_PATH = Path("data/processed/features_with_final_signal.parquet")
CALIBRATOR_FILE = Path("models/calibrator.pkl")


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Input dataset not found: {RAW_PATH}")
    if not CALIBRATOR_FILE.exists():
        raise FileNotFoundError("Calibrator missing; run training for calibration first.")

    df = pd.read_parquet(RAW_PATH).ffill().bfill()
    if "raw_entry_confidence" not in df.columns:
        raise ValueError("raw_entry_confidence missing; run aggregator first.")

    # Calibration
    calib_cfg = CalibrationConfig(model_path=str(CALIBRATOR_FILE))
    calibrator = CalibrationLayer.load(calib_cfg.model_path)
    df["entry_probability"] = calibrator.predict(df, calib_cfg)
    df = df.ffill().bfill()
    CALIBRATED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CALIBRATED_PATH, index=False)

    # Fail-safe
    df = apply_consensus_failsafe(df, DEFAULT_CONSENSUS_FAILSAFE_CONFIG)
    df = df.ffill().bfill()
    if "entry_probability_final" not in df.columns:
        raise ValueError("Fail-safe did not produce entry_probability_final.")
    FAILSAFE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FAILSAFE_PATH, index=False)

    # Final decision
    df = compute_final_signal(df, DEFAULT_FINAL_DECISION_CONFIG)
    df = df.ffill().bfill()
    if "final_signal" not in df.columns:
        raise ValueError("Final decision did not produce final_signal.")

    print("Signal counts:")
    print(df["final_signal"].value_counts())
    print("Entry probability summary:")
    print(df[["entry_probability", "entry_probability_final"]].describe())

    FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FINAL_PATH, index=False)
    print("Saved final signals to", FINAL_PATH)


if __name__ == "__main__":
    main()
