from __future__ import annotations

"""
Live helper for Pillar 2 — deterministic exit profile.

Input:
  RUN_LAYER/data/processed/PILLAR_TWO_EXIT_MODEL/exit_with_mini_engines.parquet

Output:
  RUN_LAYER/data/processed/PILLAR_TWO_EXIT_MODEL/exit_with_deterministic.parquet
"""

import gc
import logging
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parents[2]
RUN_PROCESSED = BASE_DIR / "RUN_LAYER" / "data" / "processed"

P2_PROCESSED = RUN_PROCESSED / "PILLAR_TWO_EXIT_MODEL"
INPUT_PATH = P2_PROCESSED / "exit_with_mini_engines.parquet"
OUTPUT_PATH = P2_PROCESSED / "exit_with_deterministic.parquet"

# Pillar 2 src
P2_BASE = BASE_DIR / "PILLAR_TWO_EXIT_MODEL"
P2_SRC = P2_BASE / "src"
for p in (P2_BASE, P2_SRC):
    ps = str(p)
    if ps not in sys.path:
        sys.path.append(ps)

from exit.exit_deterministic import ExitDeterministicConfig, build_exit_deterministic_profile  # type: ignore


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024**2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Mini-engines parquet not found: {INPUT_PATH}")

    logging.info("Loading exit_with_mini_engines (live): %s", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)
    _log_mem(df, "[ExitDeterministic] Loaded mini-engines parquet")

    cfg = ExitDeterministicConfig()
    df_det = build_exit_deterministic_profile(df, cfg=cfg)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_det.to_parquet(OUTPUT_PATH, index=False, compression="zstd")
    logging.info("Saved Pillar 2 deterministic exit profile (live) -> %s", OUTPUT_PATH)

    del df, df_det
    gc.collect()


if __name__ == "__main__":
    main()
