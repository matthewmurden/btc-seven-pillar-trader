# src/pipeline/build_exit_labels.py

"""
Pipeline entrypoint for Pillar 2 — TP/SL Label Builder.

Inputs:
  - PILLAR_TWO_EXIT_MODEL/data/processed/exit_raw.parquet
      (one row per active trade, at entry time)
  - PILLAR_ONE_ENTRY_MODEL/data/processed/features_pillar1_with_engines.parquet
      (full OHLC stream with ATR etc.)

Outputs:
  - PILLAR_TWO_EXIT_MODEL/data/processed/exit_labels.parquet

Run:
  python -m src.pipeline.build_exit_labels
"""

import logging
from pathlib import Path

import pandas as pd

from src.exit.exit_labels import ExitLabelConfig, build_exit_labels

BASE_DIR = Path(__file__).resolve().parents[3]

P2_BASE = BASE_DIR / "PILLAR_TWO_EXIT_MODEL"
P1_BASE = BASE_DIR / "PILLAR_ONE_ENTRY_MODEL"

EXIT_RAW_PATH = P2_BASE / "data" / "processed" / "exit_raw.parquet"
P1_FEATURES_PATH = P1_BASE / "data" / "processed" / "features_pillar1_with_engines.parquet"
OUTPUT_PATH = P2_BASE / "data" / "processed" / "exit_labels.parquet"


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not EXIT_RAW_PATH.exists():
        raise FileNotFoundError(f"exit_raw parquet not found: {EXIT_RAW_PATH}")
    if not P1_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Pillar 1 features parquet not found: {P1_FEATURES_PATH}")

    cfg = ExitLabelConfig()

    # 1) Load exit_raw (trade-level features)
    logging.info("[ExitLabelsPipeline] Loading exit_raw: %s", EXIT_RAW_PATH)
    exit_raw = pd.read_parquet(EXIT_RAW_PATH)
    _log_mem(exit_raw, "[ExitLabelsPipeline] Loaded exit_raw")

    # 2) Load just the OHLC + ATR columns from Pillar 1 features
    price_cols = [cfg.col_time, cfg.col_close, cfg.col_high, cfg.col_low]
    # ATR is optional but strongly recommended
    if cfg.col_atr is not None:
        price_cols.append(cfg.col_atr)

    logging.info(
        "[ExitLabelsPipeline] Loading price stream from P1 features: %s (cols=%s)",
        P1_FEATURES_PATH,
        price_cols,
    )
    price_df = pd.read_parquet(P1_FEATURES_PATH, columns=price_cols)
    _log_mem(price_df, "[ExitLabelsPipeline] Loaded price_df (subset OHLC+ATR)")

    # 3) Build labels
    logging.info(
        "[ExitLabelsPipeline] Building exit labels with config: %s",
        cfg,
    )
    labels_df = build_exit_labels(exit_raw=exit_raw, price_df=price_df, cfg=cfg)

    # 4) Save to disk
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_parquet(OUTPUT_PATH, index=False)
    logging.info(
        "[ExitLabelsPipeline] Saved exit labels -> %s",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()