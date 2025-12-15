from __future__ import annotations

"""
Live helper to build Pillar 4 raw regime features using the existing
deterministic_regime_engines logic, but on the run-layer normalised features.

Inputs:
  - RUN_LAYER/data/processed/PRE_PILLARFOUR/normalised_features.parquet

Outputs:
  - RUN_LAYER/data/processed/PILLAR_FOUR_REGIME_DETECTION/raw_regime_features.parquet
"""

import logging
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parents[2]
RUN_PROCESSED = BASE_DIR / "RUN_LAYER" / "data" / "processed"
OUT_DIR = RUN_PROCESSED / "PILLAR_FOUR_REGIME_DETECTION"

NORM_INPUT = RUN_PROCESSED / "PRE_PILLARFOUR" / "normalised_features.parquet"
OUT_PATH = OUT_DIR / "raw_regime_features.parquet"

P4_BASE = BASE_DIR / "PILLAR_FOUR_REGIME_DETECTION"
P4_SRC = P4_BASE / "src"
if str(P4_SRC) not in sys.path:
    sys.path.append(str(P4_SRC))

from build_raw_regime_features import main as build_part_one  # type: ignore


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not NORM_INPUT.exists():
        raise FileNotFoundError(f"Normalised features missing: {NORM_INPUT}")

    # Temporarily patch the paths used inside build_raw_regime_features
    import build_raw_regime_features as brf  # type: ignore

    brf.NORM_INPUT = NORM_INPUT
    brf.OUT = OUT_PATH

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("Building raw regime features from %s", NORM_INPUT)
    build_part_one()
    logging.info("Saved raw regime features -> %s", OUT_PATH)


if __name__ == "__main__":
    main()
