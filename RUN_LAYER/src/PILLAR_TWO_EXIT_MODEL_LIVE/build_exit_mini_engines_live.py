from __future__ import annotations

"""
Live helper for Pillar 2 — deterministic mini-engines.

Input:
  RUN_LAYER/data/processed/PILLAR_TWO_EXIT_MODEL/exit_raw.parquet

Output:
  RUN_LAYER/data/processed/PILLAR_TWO_EXIT_MODEL/exit_with_mini_engines.parquet
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
EXIT_RAW_PATH = P2_PROCESSED / "exit_raw.parquet"
OUTPUT_PATH = P2_PROCESSED / "exit_with_mini_engines.parquet"

# Pillar 2 src
P2_BASE = BASE_DIR / "PILLAR_TWO_EXIT_MODEL"
P2_SRC = P2_BASE / "src"
for p in (P2_BASE, P2_SRC):
    ps = str(p)
    if ps not in sys.path:
        sys.path.append(ps)

from exit.exit_mini_engines import ExitMiniEngineConfig, build_exit_mini_engines  # type: ignore


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024**2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not EXIT_RAW_PATH.exists():
        raise FileNotFoundError(f"exit_raw parquet not found: {EXIT_RAW_PATH}")

    logging.info("Loading exit_raw (live): %s", EXIT_RAW_PATH)
    exit_raw = pd.read_parquet(EXIT_RAW_PATH)
    _log_mem(exit_raw, "[ExitMini] Loaded exit_raw")

    cfg = ExitMiniEngineConfig()
    exit_with_engines = build_exit_mini_engines(exit_raw, cfg=cfg)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    exit_with_engines.to_parquet(OUTPUT_PATH, index=False, compression="zstd")
    logging.info("Saved Pillar 2 mini-engine outputs (live) -> %s", OUTPUT_PATH)

    del exit_raw, exit_with_engines
    gc.collect()


if __name__ == "__main__":
    main()
