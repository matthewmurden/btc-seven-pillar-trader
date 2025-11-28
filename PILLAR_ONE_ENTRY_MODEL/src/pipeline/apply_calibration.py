"""
Step 4 — Apply calibration to raw_entry_confidence.
Loads raw confidence, applies trained calibrator, saves entry_probability.
"""

from pathlib import Path

import pandas as pd

from src.calibration.calibration_layer import CalibrationLayer
from src.calibration.calibration_config import CalibrationConfig


INPUT_PATH = Path("data/processed/features_with_raw_confidence.parquet")
OUTPUT_PATH = Path("data/processed/features_with_entry_probability.parquet")
CALIBRATOR_PATH = Path("models/calibrator.pkl")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_PATH}")
    if not CALIBRATOR_PATH.exists():
        raise FileNotFoundError(f"Calibrator not found: {CALIBRATOR_PATH}")

    df = pd.read_parquet(INPUT_PATH).ffill().bfill()
    if "raw_entry_confidence" not in df.columns:
        raise ValueError("Missing raw_entry_confidence column. Run aggregator first.")
    if df["raw_entry_confidence"].isna().any():
        raise ValueError("NaNs detected in raw_entry_confidence.")

    config = CalibrationConfig(model_path=str(CALIBRATOR_PATH))
    calibrator = CalibrationLayer.load(config.model_path)
    probs = calibrator.predict(df, config)
    df["entry_probability"] = probs.clip(config.clip_min, config.clip_max)
    df = df.ffill().bfill()

    print("Calibration applied. Shape:", df.shape)
    print("Entry probability summary:")
    print(df["entry_probability"].describe())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print("Saved calibrated probabilities to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
