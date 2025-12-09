"""
Pipeline entrypoint for Pillar 1 — Label Construction.

Uses the feature-level dataset (with primitives + mini-engines) to build
all labels required by the multi-horizon ML models (Section 3.4):

  - y_dir_1_2
  - y_dir_5_10
  - y_dir_15_30
  - y_breakout
  - y_mean_reversion

Input:
  PILLAR_ONE_ENTRY_MODEL/data/processed/features_pillar1_with_engines.parquet

Output:
  PILLAR_ONE_ENTRY_MODEL/data/processed/labels_pillar1.parquet

Run:
  python -m src.pipeline.build_pillar1_labels
"""

import logging
from pathlib import Path

import pandas as pd

from src.labels.pillar1_labels import (
    Pillar1LabelConfig,
    build_pillar1_labels,
)

BASE_DIR = Path(__file__).resolve().parents[3]
P1_BASE = BASE_DIR / "PILLAR_ONE_ENTRY_MODEL"

FEATURES_PATH = P1_BASE / "data" / "processed" / "features_pillar1_with_engines.parquet"
OUTPUT_PATH = P1_BASE / "data" / "processed" / "labels_pillar1.parquet"


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature parquet not found: {FEATURES_PATH}")

    logging.info("Loading features for label construction: %s", FEATURES_PATH)
    feats = pd.read_parquet(FEATURES_PATH)
    _log_mem(feats, "[P1-Labels] Loaded features_with_engines")

    # Build labels
    cfg = Pillar1LabelConfig()
    logging.info("Building Pillar 1 labels with config: %s", cfg)
    labels_df = build_pillar1_labels(feats, cfg)
    _log_mem(labels_df, "[P1-Labels] Built labels")

    # Basic sanity diagnostics
    logging.info("Label value distributions (non-NaN only):")
    for col in ["y_dir_1_2", "y_dir_5_10", "y_dir_15_30", "y_breakout", "y_mean_reversion"]:
        vc = labels_df[col].value_counts(dropna=True, normalize=True).sort_index()
        logging.info("  %s:\n%s", col, vc.to_string())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Saved Pillar 1 labels -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()