from __future__ import annotations

"""
Live helper to build Pillar 1 labels using the existing label pipeline,
but wired to the live processed inputs under RUN_LAYER/data/processed.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Paths
THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parents[2]
RUN_PROCESSED = BASE_DIR / "RUN_LAYER" / "data" / "processed"

P1_PROCESSED = RUN_PROCESSED / "PILLAR_ONE_ENTRY_MODEL"
FEATURES_PATH = P1_PROCESSED / "features_pillar1_with_engines.parquet"
OUTPUT_PATH = P1_PROCESSED / "labels_pillar1.parquet"

# Pillar 1 src
P1_SRC = (BASE_DIR / "PILLAR_ONE_ENTRY_MODEL" / "src")
if str(P1_SRC) not in sys.path:
    sys.path.append(str(P1_SRC))

from labels.pillar1_labels import (  # type: ignore
    Pillar1LabelConfig,
    build_pillar1_labels,
)


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature parquet not found: {FEATURES_PATH}")

    logging.info("Loading features_with_engines (live) from %s", FEATURES_PATH)
    feats = pd.read_parquet(FEATURES_PATH)
    _log_mem(feats, "[P1-Labels] Loaded features_with_engines")

    cfg = Pillar1LabelConfig()
    logging.info("Building Pillar 1 labels (live) with config: %s", cfg)
    labels_df = build_pillar1_labels(feats, cfg)
    _log_mem(labels_df, "[P1-Labels] Built labels")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Saved Pillar 1 labels (live) -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
