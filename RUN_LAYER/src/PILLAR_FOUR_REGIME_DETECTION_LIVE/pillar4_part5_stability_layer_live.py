from __future__ import annotations

"""
Live Pillar 4 Part 5: apply stability layer to fused regimes using existing logic,
with inputs/outputs under RUN_LAYER/data/processed/PILLAR_FOUR_REGIME_DETECTION.

Inputs:
  - fused regimes: regime_labels_fused.parquet (live)
  - ML predictions: regime_ml_predictions.parquet (live)
  - deterministic engines: deterministic_regime_engines.parquet (live)

Output:
  - regime_labels_stable.parquet (live)
"""

import logging
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parents[2]
RUN_PROCESSED = BASE_DIR / "RUN_LAYER" / "data" / "processed"
P4_OUT = RUN_PROCESSED / "PILLAR_FOUR_REGIME_DETECTION"

FUSED_PATH = P4_OUT / "regime_labels_fused.parquet"
ML_PATH = P4_OUT / "regime_ml_predictions.parquet"
DET_PATH = P4_OUT / "deterministic_regime_engines.parquet"
OUTPUT_PATH = P4_OUT / "regime_labels_stable.parquet"

P4_BASE = BASE_DIR / "PILLAR_FOUR_REGIME_DETECTION"
P4_SRC = P4_BASE / "src"
if str(P4_SRC) not in sys.path:
    sys.path.append(str(P4_SRC))

from part_five import pillar4_part5_stability_layer as stab_mod  # type: ignore


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    for p in [FUSED_PATH, ML_PATH, DET_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Required input missing: {p}")

    fused = pd.read_parquet(FUSED_PATH)
    ml = pd.read_parquet(ML_PATH)
    det = pd.read_parquet(DET_PATH)

    stable_df = stab_mod.smooth(fused, ml, det)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stable_df.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Saved live stable regimes -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
