# src/pipeline/build_exit_mini_engines.py
"""
Pipeline entrypoint for Pillar 2 — Mini-Engines (Deterministic Exit Engines).

Loads:
  - data/processed/exit_raw.parquet

Applies deterministic mini-engines:
  - Excursion Engine
  - Path Shape Engine
  - Entropy Engine
  - Volatility Dynamics Engine
  - Reversion Distance Engine

Writes:
  - data/processed/exit_with_mini_engines.parquet

Run:
  python -m src.pipeline.build_exit_mini_engines
"""

import gc
import logging
from pathlib import Path

import pandas as pd

from src.exit.exit_mini_engines import (
    ExitMiniEngineConfig,
    build_exit_mini_engines,
)

BASE_DIR = Path(__file__).resolve().parents[3]
P2_BASE = BASE_DIR / "PILLAR_TWO_EXIT_MODEL"

EXIT_RAW_PATH = P2_BASE / "data" / "processed" / "exit_raw.parquet"
OUTPUT_PATH = P2_BASE / "data" / "processed" / "exit_with_mini_engines.parquet"


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024**2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not EXIT_RAW_PATH.exists():
        raise FileNotFoundError(f"exit_raw parquet not found: {EXIT_RAW_PATH}")

    logging.info("Loading exit_raw features: %s", EXIT_RAW_PATH)
    exit_raw = pd.read_parquet(EXIT_RAW_PATH)
    _log_mem(exit_raw, "[ExitMini] Loaded exit_raw")

    cfg = ExitMiniEngineConfig()

    exit_with_engines = build_exit_mini_engines(exit_raw, cfg=cfg)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    exit_with_engines.to_parquet(OUTPUT_PATH, index=False, compression="zstd")

    logging.info(
        "Saved Pillar 2 features with mini-engines -> %s",
        OUTPUT_PATH,
    )

    del exit_raw, exit_with_engines
    gc.collect()


if __name__ == "__main__":
    main()