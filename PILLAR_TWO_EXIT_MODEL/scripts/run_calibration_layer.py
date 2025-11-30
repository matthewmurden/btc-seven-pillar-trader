"""
Train and apply calibration layer for exit probabilities and scores.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from calibration.calibration_layer import apply_calibration, train_calibration_layer


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
MASTER_PATH = BASE_DIR / "data" / "processed" / "exit_master.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "exit_master_calibrated.parquet"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Missing exit_master parquet: {MASTER_PATH}")

    logger.info("Loading master file: %s", MASTER_PATH)
    df = pd.read_parquet(MASTER_PATH)

    logger.info("Training calibration layer...")
    calibrated_train = train_calibration_layer(df)

    logger.info("Applying calibration...")
    calibrated_apply = apply_calibration(df)

    # Merge calibrated outputs into one result (apply output keeps original columns)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    calibrated_apply.to_parquet(OUTPUT_PATH, index=False)

    logger.info("Saved calibrated master to %s", OUTPUT_PATH)
    logger.info("Head:\n%s", calibrated_apply.head())
    logger.info("Summary:\n%s", calibrated_apply.describe())
    logger.info("NaN counts:\n%s", calibrated_apply.isna().sum())
    logger.info(
        "exit_quality_calibrated range: min=%.6f max=%.6f",
        calibrated_apply["exit_quality_calibrated"].min(),
        calibrated_apply["exit_quality_calibrated"].max(),
    )


if __name__ == "__main__":
    main()
