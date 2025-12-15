from __future__ import annotations

"""
Live helper to compute deterministic regime engines (Pillar 4 Part 2) using
the existing deterministic_regime_engines module, with inputs/outputs under
RUN_LAYER/data/processed/PILLAR_FOUR_REGIME_DETECTION.

Input:
  - RUN_LAYER/data/processed/PILLAR_FOUR_REGIME_DETECTION/raw_regime_features.parquet

Output:
  - RUN_LAYER/data/processed/PILLAR_FOUR_REGIME_DETECTION/deterministic_regime_engines.parquet
"""

import logging
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parents[2]
RUN_PROCESSED = BASE_DIR / "RUN_LAYER" / "data" / "processed"
P4_OUT = RUN_PROCESSED / "PILLAR_FOUR_REGIME_DETECTION"

RAW_FEATURES_PATH = P4_OUT / "raw_regime_features.parquet"
OUTPUT_PATH = P4_OUT / "deterministic_regime_engines.parquet"

P4_BASE = BASE_DIR / "PILLAR_FOUR_REGIME_DETECTION"
P4_SRC = P4_BASE / "src"
if str(P4_SRC) not in sys.path:
    sys.path.append(str(P4_SRC))

from deterministic_regime_engines import main as build_det  # type: ignore
import deterministic_regime_engines as det_mod  # type: ignore


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not RAW_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Raw regime features missing: {RAW_FEATURES_PATH}")

    det_mod.INPUT_PATH = RAW_FEATURES_PATH
    det_mod.OUTPUT_PATH = OUTPUT_PATH

    logging.info("Building deterministic regime engines from %s", RAW_FEATURES_PATH)
    build_det()
    logging.info("Saved deterministic regime engines -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
