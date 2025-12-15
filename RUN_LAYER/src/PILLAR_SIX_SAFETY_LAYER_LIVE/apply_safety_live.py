from __future__ import annotations

"""
Live helper for Pillar 6 — Safety / Filters.

Inputs:
  RUN_LAYER/data/processed/PILLAR_TWO_EXIT_MODEL/exit_final_targets.parquet

Outputs:
  RUN_LAYER/data/processed/PILLAR_SIX_SAFETY_LAYER/exit_with_safety.parquet

This mirrors the offline `apply_safety_to_df` workflow and does not
maintain live safety state (loss streaks, cooldowns, etc.). It simply
applies the structural/system filters to the Pillar 2 outputs.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parents[2]
RUN_PROCESSED = BASE_DIR / "RUN_LAYER" / "data" / "processed"

P2_PROCESSED = RUN_PROCESSED / "PILLAR_TWO_EXIT_MODEL"
P6_PROCESSED = RUN_PROCESSED / "PILLAR_SIX_SAFETY_LAYER"

INPUT_PATH = P2_PROCESSED / "exit_final_targets.parquet"
OUTPUT_PATH = P6_PROCESSED / "exit_with_safety.parquet"

# Pillar 6 src
P6_BASE = BASE_DIR / "PILLAR_SIX_SAFETY_LAYER"
P6_SRC = P6_BASE / "src"
for p in (P6_BASE, P6_SRC):
    ps = str(p)
    if ps not in sys.path:
        sys.path.append(ps)

from safety.safety_engine import SafetyConfig, apply_safety_to_df  # type: ignore


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Pillar 2 final targets not found: {INPUT_PATH}")

    logging.info("Loading Pillar 2 final targets (live): %s", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)
    _log_mem(df, "[SafetyLive] Loaded exit_final_targets")

    cfg = SafetyConfig()
    df_safe = apply_safety_to_df(df, cfg=cfg)
    _log_mem(df_safe, "[SafetyLive] After safety filter")

    P6_PROCESSED.mkdir(parents=True, exist_ok=True)
    df_safe.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Saved Pillar 6 safety outputs (live) -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
