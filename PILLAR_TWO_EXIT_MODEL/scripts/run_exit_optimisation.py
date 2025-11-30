"""
Run the exit optimisation engine on merged meta and exit feature datasets.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from optimisation.exit_optimisation_engine import optimise_exits


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
META_PATH = BASE_DIR / "data" / "processed" / "meta_exit_features.parquet"
EXIT_WITH_MINI_PATH = BASE_DIR / "data" / "processed" / "exit_features_with_mini_engines.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "exit_optimised.parquet"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing meta features file: {META_PATH}")
    if not EXIT_WITH_MINI_PATH.exists():
        raise FileNotFoundError(f"Missing exit features with mini engines file: {EXIT_WITH_MINI_PATH}")

    logger.info("Loading meta features from %s", META_PATH)
    meta = pd.read_parquet(META_PATH)
    logger.info("Loading exit + mini features from %s", EXIT_WITH_MINI_PATH)
    exit_feat = pd.read_parquet(EXIT_WITH_MINI_PATH)

    merged = pd.concat([meta.reset_index(drop=True), exit_feat.reset_index(drop=True)], axis=1)
    logger.info("Merged dataset shape: %s", merged.shape)

    result = optimise_exits(merged)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)

    logger.info("Saved optimisation output to %s", OUTPUT_PATH)
    logger.info("Summary:\n%s", result.describe())
    logger.info("NaN counts:\n%s", result.isna().sum())


if __name__ == "__main__":
    main()
