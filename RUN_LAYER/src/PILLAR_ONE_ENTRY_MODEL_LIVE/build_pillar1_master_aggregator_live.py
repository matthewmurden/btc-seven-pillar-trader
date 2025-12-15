from __future__ import annotations

"""
Live helper to run the Pillar 1 regime-weighted master aggregator.

Inputs:
  RUN_LAYER/data/processed/PILLAR_ONE_ENTRY_MODEL/features_pillar1_with_meta.parquet

Outputs:
  RUN_LAYER/data/processed/PILLAR_ONE_ENTRY_MODEL/features_pillar1_with_agg.parquet
"""

import logging
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parents[2]
RUN_PROCESSED = BASE_DIR / "RUN_LAYER" / "data" / "processed"

P1_PROCESSED = RUN_PROCESSED / "PILLAR_ONE_ENTRY_MODEL"
INPUT_PATH = P1_PROCESSED / "features_pillar1_with_meta.parquet"
OUTPUT_PATH = P1_PROCESSED / "features_pillar1_with_agg.parquet"

# Pillar 1 src
P1_SRC = BASE_DIR / "PILLAR_ONE_ENTRY_MODEL" / "src"
if str(P1_SRC) not in sys.path:
    sys.path.append(str(P1_SRC))

from models.pillar1_master_aggregator import (  # type: ignore
    MasterAggregatorConfig,
    compute_master_aggregator,
)


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input parquet not found: {INPUT_PATH}")

    logging.info("Loading features_with_meta (live) from %s", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)
    _log_mem(df, "[Agg] Loaded features_with_meta")

    cfg = MasterAggregatorConfig()
    logging.info("Computing regime-weighted master aggregation (live) with config: %s", cfg)

    df_out = compute_master_aggregator(df, cfg=cfg)
    _log_mem(df_out, "[Agg] After master aggregation")

    if "raw_entry_confidence" not in df_out.columns:
        raise RuntimeError("raw_entry_confidence not found after aggregation.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Saved Pillar 1 features with master aggregator (live) -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
