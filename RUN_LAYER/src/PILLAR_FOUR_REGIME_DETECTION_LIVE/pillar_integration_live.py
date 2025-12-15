from __future__ import annotations

"""
Live Pillar 4 Integration (Part 6/7):
Integrate fused/stable regimes into a master timeline using the existing
regime_master_api, with inputs/outputs under RUN_LAYER/data/processed/PILLAR_FOUR_REGIME_DETECTION.

Inputs:
  - regime_labels_stable.parquet (live)
  - regime_labels_fused.parquet (live)
  - regime_ml_predictions.parquet (live)
  - deterministic_regime_engines.parquet (live)

Output:
  - regime_timeline.parquet (live) via RegimeMasterAPI
"""

import logging
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parents[2]
RUN_PROCESSED = BASE_DIR / "RUN_LAYER" / "data" / "processed"
P4_OUT = RUN_PROCESSED / "PILLAR_FOUR_REGIME_DETECTION"

STABLE_PATH = P4_OUT / "regime_labels_stable.parquet"
FUSED_PATH = P4_OUT / "regime_labels_fused.parquet"
ML_PATH = P4_OUT / "regime_ml_predictions.parquet"
DET_PATH = P4_OUT / "deterministic_regime_engines.parquet"
OUTPUT_PATH = P4_OUT / "regime_timeline.parquet"

P4_BASE = BASE_DIR / "PILLAR_FOUR_REGIME_DETECTION"
P4_SRC = P4_BASE / "src"
if str(P4_SRC) not in sys.path:
    sys.path.append(str(P4_SRC))

from part_seven.regime_master_api import RegimeMasterAPI  # type: ignore


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    for p in [STABLE_PATH, FUSED_PATH, ML_PATH, DET_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Required input missing: {p}")

    stable = pd.read_parquet(STABLE_PATH)
    fused = pd.read_parquet(FUSED_PATH)
    ml = pd.read_parquet(ML_PATH)
    det = pd.read_parquet(DET_PATH)

    api = RegimeMasterAPI(
        stable_labels=stable,
        fused_labels=fused,
        ml_preds=ml,
        deterministic=det,
    )

    timeline = api.build_timeline()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    timeline.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Saved live regime timeline -> %s (rows=%d)", OUTPUT_PATH, len(timeline))


if __name__ == "__main__":
    main()
