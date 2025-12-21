from __future__ import annotations

"""
Live helper for Pillar 2 — TP/SL label builder.

Inputs:
  - RUN_LAYER/data/processed/PILLAR_TWO_EXIT_MODEL/exit_raw.parquet
  - RUN_LAYER/data/processed/PILLAR_ONE_ENTRY_MODEL/features_pillar1_with_engines.parquet

Output:
  - RUN_LAYER/data/processed/PILLAR_TWO_EXIT_MODEL/exit_labels.parquet

Note: Labels are primarily for diagnostics; live inference does not need them,
but we mirror the offline pipeline for completeness.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parents[2]
RUN_PROCESSED = BASE_DIR / "RUN_LAYER" / "data" / "processed"

P2_PROCESSED = RUN_PROCESSED / "PILLAR_TWO_EXIT_MODEL"
P1_PROCESSED = RUN_PROCESSED / "PILLAR_ONE_ENTRY_MODEL"

EXIT_RAW_PATH = P2_PROCESSED / "exit_raw.parquet"
P1_FEATURES_PATH = P1_PROCESSED / "features_pillar1_with_engines.parquet"
OUTPUT_PATH = P2_PROCESSED / "exit_labels.parquet"

# Pillar 2 src
P2_BASE = BASE_DIR / "PILLAR_TWO_EXIT_MODEL"
P2_SRC = P2_BASE / "src"
for p in (P2_BASE, P2_SRC):
    ps = str(p)
    if ps not in sys.path:
        sys.path.append(ps)

from exit.exit_labels import ExitLabelConfig, build_exit_labels  # type: ignore


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024**2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not EXIT_RAW_PATH.exists():
        raise FileNotFoundError(f"exit_raw parquet not found: {EXIT_RAW_PATH}")
    if not P1_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Pillar 1 features parquet not found: {P1_FEATURES_PATH}")

    cfg = ExitLabelConfig()

    logging.info("[ExitLabelsLive] Loading exit_raw: %s", EXIT_RAW_PATH)
    exit_raw = pd.read_parquet(EXIT_RAW_PATH)
    _log_mem(exit_raw, "[ExitLabelsLive] Loaded exit_raw")

    price_cols = [cfg.col_time, cfg.col_close, cfg.col_high, cfg.col_low]
    if cfg.col_atr is not None:
        price_cols.append(cfg.col_atr)

    logging.info(
        "[ExitLabelsLive] Loading price stream from P1 features: %s (cols=%s)",
        P1_FEATURES_PATH,
        price_cols,
    )
    price_df = pd.read_parquet(P1_FEATURES_PATH, columns=price_cols)
    _log_mem(price_df, "[ExitLabelsLive] Loaded price_df (subset OHLC+ATR)")

    labels_df = build_exit_labels(exit_raw=exit_raw, price_df=price_df, cfg=cfg)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_parquet(OUTPUT_PATH, index=False)
    logging.info("[ExitLabelsLive] Saved exit labels -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
