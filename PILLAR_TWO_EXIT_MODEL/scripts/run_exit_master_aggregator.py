"""
Run the Exit Master Aggregator to produce exit_quality_score.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from aggregation.exit_master_aggregator import apply_exit_master_aggregation

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
META_PATH = BASE_DIR / "data" / "processed" / "meta_exit_features.parquet"
DECAY_PATH = BASE_DIR / "data" / "processed" / "exit_decay_filtered.parquet"
OPT_PATH = BASE_DIR / "data" / "processed" / "exit_optimised.parquet"
MINI_PATH = BASE_DIR / "data" / "processed" / "exit_features_with_mini_engines.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "exit_master.parquet"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    for path in [META_PATH, DECAY_PATH, OPT_PATH, MINI_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    logger.info("Loading meta features (REQUIRED_COLUMNS source)")
    meta = pd.read_parquet(META_PATH)

    logger.info("Loading decay-filtered features")
    decay = pd.read_parquet(DECAY_PATH)

    logger.info("Loading optimisation outputs")
    opt = pd.read_parquet(OPT_PATH)

    logger.info("Loading raw mini-engine features")
    mini = pd.read_parquet(MINI_PATH)

    # Simple merge without dropping required columns
    merged = pd.concat(
        [
            meta.reset_index(drop=True),
            decay.reset_index(drop=True),
            opt.reset_index(drop=True),
            mini.reset_index(drop=True),
        ],
        axis=1,
    )
    merged = merged.loc[:, ~merged.columns.duplicated()]  # keep first occurrence (meta is authoritative)
    logger.info("Merged shape: %s", merged.shape)

    aggregated = apply_exit_master_aggregation(merged)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_parquet(OUTPUT_PATH, index=False)

    logger.info("Saved exit master output to %s", OUTPUT_PATH)
    logger.info("Head:\n%s", aggregated.head())
    logger.info("Summary:\n%s", aggregated.describe())
    logger.info("NaN counts:\n%s", aggregated.isna().sum())
    logger.info(
        "exit_quality_score range: min=%.4f max=%.4f",
        aggregated["exit_quality_score"].min(),
        aggregated["exit_quality_score"].max(),
    )


if __name__ == "__main__":
    main()
