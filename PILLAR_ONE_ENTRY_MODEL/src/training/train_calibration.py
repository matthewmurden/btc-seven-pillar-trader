"""
Step 5 — Train calibration layer:
Maps raw_entry_confidence → true probability.
"""

from pathlib import Path
import pandas as pd

from src.calibration.calibration_layer import CalibrationLayer
from src.calibration.calibration_config import CalibrationConfig, DEFAULT_CALIBRATION_CONFIG
from src.models.label_builder import build_labels

INPUT_PATH = Path("data/processed/features_with_raw_confidence.parquet")
CALIBRATOR_PATH = Path("models/calibrator.pkl")

# Choose the REAL truth horizon for your trading logic
TRUTH_LABEL = "next_5m_up"   # <<< CHANGE HERE if needed


def main():
    print("Loading dataset:", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH).ffill().bfill()

    # -----------------------------------------------------
    # Ensure labels exist (clean, deterministic rebuild)
    # -----------------------------------------------------
    if TRUTH_LABEL not in df.columns:
        print(f"{TRUTH_LABEL} missing — rebuilding labels...")
        df = build_labels(df).ffill().bfill()

    # -----------------------------------------------------
    # Ensure aggregator result exists
    # -----------------------------------------------------
    if "raw_entry_confidence" not in df.columns:
        raise ValueError("Missing raw_entry_confidence — run aggregator first.")

    # -----------------------------------------------------
    # Filter rows
    # -----------------------------------------------------
    df = df.dropna(subset=[TRUTH_LABEL, "raw_entry_confidence"])
    df[TRUTH_LABEL] = df[TRUTH_LABEL].astype(int)

    print(f"Calibration rows: {len(df)}")
    print("Truth label distribution:", df[TRUTH_LABEL].mean())

    # -----------------------------------------------------
    # Build config
    # -----------------------------------------------------
    config = CalibrationConfig(
        method="isotonic",  # preserve distribution shape; avoid over-compression
        clip_min=DEFAULT_CALIBRATION_CONFIG.clip_min,
        clip_max=DEFAULT_CALIBRATION_CONFIG.clip_max,
        model_path=str(CALIBRATOR_PATH),
    )

    # -----------------------------------------------------
    # Train calibrator
    # -----------------------------------------------------
    calibrator = CalibrationLayer(method=config.method)
    calibrator.fit(df.assign(target=df[TRUTH_LABEL]), config)
    calibrator.save(config.model_path)

    # -----------------------------------------------------
    # Diagnostic output
    # -----------------------------------------------------
    probs = calibrator.predict(df, config)
    print("\nCalibrated probability summary:")
    print(probs.describe())

    print("\nSaved calibrator to:", CALIBRATOR_PATH)


if __name__ == "__main__":
    main()
