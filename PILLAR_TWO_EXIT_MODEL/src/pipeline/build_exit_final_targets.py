# src/pipeline/build_exit_final_targets.py

"""
Pipeline entrypoint for Pillar 2 — Final TP/SL Generator.

Loads:
  - data/processed/exit_with_fusion.parquet

Applies:
  - build_final_exit_targets (uses fused TP/SL, regimes, confidence, etc.)

Writes:
  - data/processed/exit_final_targets.parquet

Run:
  python -m src.pipeline.build_exit_final_targets
"""

import gc
import logging
from pathlib import Path

import pandas as pd

from src.exit.exit_final_targets import ExitFinalConfig, build_final_exit_targets

BASE_DIR = Path(__file__).resolve().parents[3]
P2_BASE = BASE_DIR / "PILLAR_TWO_EXIT_MODEL"

FUSION_PATH = P2_BASE / "data" / "processed" / "exit_with_fusion.parquet"
OUTPUT_PATH = P2_BASE / "data" / "processed" / "exit_final_targets.parquet"


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024**2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not FUSION_PATH.exists():
        raise FileNotFoundError(f"Fused exits parquet not found: {FUSION_PATH}")

    logging.info("Loading exit_with_fusion: %s", FUSION_PATH)
    df = pd.read_parquet(FUSION_PATH)
    _log_mem(df, "[ExitFinal] Loaded fused exits")

    cfg = ExitFinalConfig()

    df_final = build_final_exit_targets(df, cfg=cfg)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(OUTPUT_PATH, index=False, compression="zstd")

    logging.info(
        "Saved Pillar 2 final TP/SL targets -> %s",
        OUTPUT_PATH,
    )

    del df, df_final
    gc.collect()


if __name__ == "__main__":
    main()