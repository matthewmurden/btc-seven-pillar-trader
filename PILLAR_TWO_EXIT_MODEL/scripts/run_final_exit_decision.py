"""
Run the final exit decision layer on exit_master_calibrated.parquet.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from final_exit_decision_layer import apply_final_exit_decisions


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "processed" / "exit_master_calibrated.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "final_exit_decisions.parquet"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    logger.info("Loading calibrated master file from %s", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)

    logger.info("Applying final exit decision layer...")
    decisions = apply_final_exit_decisions(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_parquet(OUTPUT_PATH, index=False)

    logger.info("Saved final decisions to %s", OUTPUT_PATH)
    logger.info("Head:\n%s", decisions.head())
    logger.info("Summary:\n%s", decisions.describe())
    logger.info("Decision counts:\n%s", decisions["decision"].value_counts())


if __name__ == "__main__":
    main()
