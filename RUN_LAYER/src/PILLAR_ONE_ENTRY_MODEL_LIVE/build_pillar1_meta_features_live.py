from __future__ import annotations

"""
Live helper to compute Pillar 1 meta-features using existing code.

Inputs:
  RUN_LAYER/data/processed/PILLAR_ONE_ENTRY_MODEL/features_pillar1_with_ml.parquet

Output:
  RUN_LAYER/data/processed/PILLAR_ONE_ENTRY_MODEL/features_pillar1_with_meta.parquet
"""

import logging
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parents[2]
RUN_PROCESSED = BASE_DIR / "RUN_LAYER" / "data" / "processed"

P1_PROCESSED = RUN_PROCESSED / "PILLAR_ONE_ENTRY_MODEL"
INPUT_PATH = P1_PROCESSED / "features_pillar1_with_ml.parquet"
OUTPUT_PATH = P1_PROCESSED / "features_pillar1_with_meta.parquet"

# Pillar 1 src
P1_SRC = BASE_DIR / "PILLAR_ONE_ENTRY_MODEL" / "src"
if str(P1_SRC) not in sys.path:
    sys.path.append(str(P1_SRC))

from features.pillar1_meta_features import (  # type: ignore
    MetaFeatureConfig,
    build_pillar1_meta_features,
    summarize_meta_features,
)


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input features parquet not found: {INPUT_PATH}")

    logging.info("Loading features_with_ml (live) from %s", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)
    _log_mem(df, "[Meta] Loaded features_with_ml")

    if "open_time_ms" in df.columns:
        df = df.sort_values("open_time_ms").reset_index(drop=True)

    cfg = MetaFeatureConfig()
    logging.info("Building Pillar 1 meta-features (live) with config: %s", cfg)
    df_out = build_pillar1_meta_features(df, cfg)

    _log_mem(df_out, "[Meta] After meta-feature construction")
    summarize_meta_features(df_out)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Saved Pillar 1 features with meta-features (live) -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
