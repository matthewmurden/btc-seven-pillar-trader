# src/pipeline/build_exit_fusion.py

"""
Pipeline entrypoint for Pillar 2 — Fusion Layer.

Loads:
  - data/processed/exit_with_ml.parquet

Applies fusion logic combining:
  - deterministic exits
  - ML predictions
  - regime context
  - breakout probabilities

Writes:
  - data/processed/exit_with_fusion.parquet

Run:
  python -m src.pipeline.build_exit_fusion
"""

import gc
import logging
from pathlib import Path

import pandas as pd

from src.exit.exit_fusion import ExitFusionConfig, fuse_exit_signals

BASE_DIR = Path(__file__).resolve().parents[3]
P2_BASE = BASE_DIR / "PILLAR_TWO_EXIT_MODEL"

EXIT_ML_PATH = P2_BASE / "data" / "processed" / "exit_with_ml.parquet"
OUTPUT_PATH = P2_BASE / "data" / "processed" / "exit_with_fusion.parquet"


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not EXIT_ML_PATH.exists():
        raise FileNotFoundError(f"exit_with_ml parquet not found: {EXIT_ML_PATH}")

    logging.info("Loading exit_with_ml: %s", EXIT_ML_PATH)
    df = pd.read_parquet(EXIT_ML_PATH)
    _log_mem(df, "[ExitFusion] Loaded exit_with_ml")

    cfg = ExitFusionConfig()

    df_fused = fuse_exit_signals(df, cfg=cfg)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_fused.to_parquet(OUTPUT_PATH, index=False, compression="zstd")

    logging.info("Saved Pillar 2 fused exits -> %s", OUTPUT_PATH)

    del df, df_fused
    gc.collect()


if __name__ == "__main__":
    main()