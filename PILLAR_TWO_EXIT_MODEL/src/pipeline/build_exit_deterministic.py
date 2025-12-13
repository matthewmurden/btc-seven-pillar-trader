"""
Pipeline entrypoint for Pillar 2 — Deterministic Exit Engines (Part 2).

Loads:
  - data/processed/exit_with_mini_engines.parquet

Applies deterministic exit engines to produce a candidate exit profile:
  - deterministic_tp
  - deterministic_sl
  - deterministic_tp_first_probability
  - deterministic_exit_time_estimate

Writes:
  - data/processed/exit_with_deterministic.parquet

Run:
  python -m src.pipeline.build_exit_deterministic
"""

import gc
import logging
from pathlib import Path

import pandas as pd

from src.exit.exit_deterministic import (
    ExitDeterministicConfig,
    build_exit_deterministic_profile,
)

BASE_DIR = Path(__file__).resolve().parents[3]
P2_BASE = BASE_DIR / "PILLAR_TWO_EXIT_MODEL"

INPUT_PATH = P2_BASE / "data" / "processed" / "exit_with_mini_engines.parquet"
OUTPUT_PATH = P2_BASE / "data" / "processed" / "exit_with_deterministic.parquet"


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Mini-engines parquet not found: {INPUT_PATH}")

    logging.info("Loading exit_with_mini_engines: %s", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)
    _log_mem(df, "[ExitDeterministic] Loaded mini-engines parquet")

    cfg = ExitDeterministicConfig()
    df_det = build_exit_deterministic_profile(df, cfg=cfg)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_det.to_parquet(OUTPUT_PATH, index=False, compression="zstd")

    logging.info(
        "Saved Pillar 2 deterministic exit profile -> %s",
        OUTPUT_PATH,
    )

    del df, df_det
    gc.collect()


if __name__ == "__main__":
    main()