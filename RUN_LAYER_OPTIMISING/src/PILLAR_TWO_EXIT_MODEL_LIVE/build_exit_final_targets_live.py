from __future__ import annotations

"""
Live helper for Pillar 2 — Final TP/SL generator.

Input:
  RUN_LAYER/data/processed/PILLAR_TWO_EXIT_MODEL/exit_with_fusion.parquet

Output:
  RUN_LAYER/data/processed/PILLAR_TWO_EXIT_MODEL/exit_final_targets.parquet
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
INPUT_PATH = P2_PROCESSED / "exit_with_fusion.parquet"
OUTPUT_PATH = P2_PROCESSED / "exit_final_targets.parquet"

P2_BASE = BASE_DIR / "PILLAR_TWO_EXIT_MODEL"
P2_SRC = P2_BASE / "src"
for p in (P2_BASE, P2_SRC):
    ps = str(p)
    if ps not in sys.path:
        sys.path.append(ps)

from exit.exit_final_targets import ExitFinalConfig, build_final_exit_targets  # type: ignore


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024**2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Fused exits parquet not found: {INPUT_PATH}")

    logging.info("Loading exit_with_fusion (live): %s", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)
    _log_mem(df, "[ExitFinal] Loaded fused exits")

    cfg = ExitFinalConfig()
    df_final = build_final_exit_targets(df, cfg=cfg)
    _log_mem(df_final, "[ExitFinal] Final TP/SL targets")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(OUTPUT_PATH, index=False, compression="zstd")
    logging.info("Saved Pillar 2 final TP/SL targets (live) -> %s", OUTPUT_PATH)

    del df, df_final
    gc.collect()


if __name__ == "__main__":
    main()
