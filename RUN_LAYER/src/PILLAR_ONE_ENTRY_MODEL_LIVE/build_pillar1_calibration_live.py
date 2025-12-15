from __future__ import annotations

"""
Live helper to apply the Pillar 1 calibration layer using pre-trained
calibration artifacts. No fitting is performed here.

Inputs:
  RUN_LAYER/data/processed/PILLAR_ONE_ENTRY_MODEL/features_pillar1_with_agg.parquet

Model artifact (offline-trained):
  PILLAR_ONE_ENTRY_MODEL/models/pillar1_calibration/pillar1_calibrator.joblib

Output:
  RUN_LAYER/data/processed/PILLAR_ONE_ENTRY_MODEL/features_pillar1_with_calib.parquet
"""

import logging
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parents[2]
RUN_PROCESSED = BASE_DIR / "RUN_LAYER" / "data" / "processed"

P1_PROCESSED = RUN_PROCESSED / "PILLAR_ONE_ENTRY_MODEL"
FEATURES_PATH = P1_PROCESSED / "features_pillar1_with_agg.parquet"
OUTPUT_PATH = P1_PROCESSED / "features_pillar1_with_calib.parquet"

# Pillar 1 src (add both project root and src so pickled modules under "src.*" resolve)
P1_BASE = BASE_DIR / "PILLAR_ONE_ENTRY_MODEL"
P1_SRC = P1_BASE / "src"
for p in (P1_BASE, P1_SRC):
    ps = str(p)
    if ps not in sys.path:
        sys.path.append(ps)

from models.pillar1_calibration import (  # type: ignore
    CalibrationConfig,
    load_calibrator,
    apply_calibration_to_df,
)


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Features parquet not found: {FEATURES_PATH}")

    logging.info("Loading features_with_agg (live) from %s", FEATURES_PATH)
    df = pd.read_parquet(FEATURES_PATH)
    _log_mem(df, "[Calib] Loaded features_with_agg")

    cfg = CalibrationConfig(model_dir=P1_BASE / "models" / "pillar1_calibration")
    calibrator = load_calibrator(cfg=cfg)

    logging.info("Applying pre-trained calibrator to live data...")
    df_out = apply_calibration_to_df(df, calibrator, cfg=cfg)
    _log_mem(df_out, "[Calib] After applying calibration")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Saved calibrated Pillar 1 outputs (live) -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
