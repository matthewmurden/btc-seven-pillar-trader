from __future__ import annotations

"""
Live Pillar 4 Part 4: fuse deterministic and ML regimes using the existing
fusion logic, with live inputs/outputs under RUN_LAYER/data/processed/PILLAR_FOUR_REGIME_DETECTION.

Inputs:
  - deterministic_regime_engines.parquet (live)
  - regime_ml_predictions.parquet (live)

Output:
  - regime_labels_fused.parquet (live)
"""

import logging
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parents[2]
RUN_PROCESSED = BASE_DIR / "RUN_LAYER" / "data" / "processed"
P4_OUT = RUN_PROCESSED / "PILLAR_FOUR_REGIME_DETECTION"

DET_PATH = P4_OUT / "deterministic_regime_engines.parquet"
ML_PATH = P4_OUT / "regime_ml_predictions.parquet"
OUTPUT_PATH = P4_OUT / "regime_labels_fused.parquet"

P4_BASE = BASE_DIR / "PILLAR_FOUR_REGIME_DETECTION"
P4_SRC = P4_BASE / "src"
if str(P4_SRC) not in sys.path:
    sys.path.append(str(P4_SRC))

from part_four import pillar4_part4_fusion_layer as fusion_mod  # type: ignore


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not DET_PATH.exists():
        raise FileNotFoundError(f"Deterministic regimes missing: {DET_PATH}")
    if not ML_PATH.exists():
        raise FileNotFoundError(f"ML regimes missing: {ML_PATH}")

    det = pd.read_parquet(DET_PATH)
    ml = pd.read_parquet(ML_PATH)

    fused_df = fusion_mod.fuse(det, ml)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fused_df.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Saved live fused regimes -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
