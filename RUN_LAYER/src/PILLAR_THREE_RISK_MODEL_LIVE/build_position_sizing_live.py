from __future__ import annotations

"""
Live helper for Pillar 3 — Risk Model (Position Sizing).

Inputs:
  RUN_LAYER/data/processed/PILLAR_TWO_EXIT_MODEL/exit_final_targets.parquet

Output:
  RUN_LAYER/data/processed/PILLAR_TWO_EXIT_MODEL/exit_with_sizing.parquet

This mirrors src/pipeline/build_position_sizing.py but simply applies the
existing risk surface logic to live Pillar 2 outputs (no training).
"""

import gc
import logging
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parents[2]
RUN_PROCESSED = BASE_DIR / "RUN_LAYER" / "data" / "processed"

# Input/Output within the live processed tree
P2_PROCESSED = RUN_PROCESSED / "PILLAR_TWO_EXIT_MODEL"
TRADES_PATH = P2_PROCESSED / "exit_final_targets.parquet"
OUTPUT_PATH = P2_PROCESSED / "exit_with_sizing.parquet"

# Pillar 3 source
P3_BASE = BASE_DIR / "PILLAR_THREE_RISK_MODEL"
P3_SRC = P3_BASE / "src"
for p in (P3_BASE, P3_SRC):
    ps = str(p)
    if ps not in sys.path:
        sys.path.append(ps)

from risk.risk_surface import RiskSurfaceConfig, compute_risk_surface_and_sizing  # type: ignore


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not TRADES_PATH.exists():
        raise FileNotFoundError(f"Final TP/SL targets parquet not found: {TRADES_PATH}")

    logging.info("Loading Pillar 2 final targets (live): %s", TRADES_PATH)
    trades_df = pd.read_parquet(TRADES_PATH)
    _log_mem(trades_df, "[P3] Loaded exit_final_targets (live)")

    cfg = RiskSurfaceConfig()
    logging.info("[P3] Computing risk surface + sizing (live) with config: %s", cfg)
    sized_df = compute_risk_surface_and_sizing(trades_df, cfg=cfg)
    _log_mem(sized_df, "[P3] After sizing")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sized_df.to_parquet(OUTPUT_PATH, index=False, compression="zstd")
    logging.info("Saved Pillar 3 risk surface + position sizing (live) -> %s", OUTPUT_PATH)

    del trades_df, sized_df
    gc.collect()


if __name__ == "__main__":
    main()
