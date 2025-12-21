from __future__ import annotations

"""
Live helper to build Pillar 1 feature matrix using the existing
build_pillar1_feature_matrix pipeline, but with inputs under RUN_LAYER/data/processed
and outputs under RUN_LAYER/data/processed/PILLAR_ONE_ENTRY_MODEL.
"""

import logging
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parents[2]
RUN_PROCESSED = BASE_DIR / "RUN_LAYER" / "data" / "processed"

P1_OUT_DIR = RUN_PROCESSED / "PILLAR_ONE_ENTRY_MODEL"
OUT_PATH = P1_OUT_DIR / "features_pillar1_primitives.parquet"

# Inputs (live) mirroring original script
NORM_INPUT = RUN_PROCESSED / "PRE_PILLARFOUR" / "normalised_features.parquet"
P4_BASE_LIVE = RUN_PROCESSED / "PILLAR_FOUR_REGIME_DETECTION"
DET_PATH = P4_BASE_LIVE / "deterministic_regime_engines.parquet"
FUSED_PATH = P4_BASE_LIVE / "regime_labels_fused.parquet"
STABLE_PATH = P4_BASE_LIVE / "regime_labels_stable.parquet"

# Pillar 1 source
P1_BASE = BASE_DIR / "PILLAR_ONE_ENTRY_MODEL"
P1_SRC = P1_BASE / "src"
if str(P1_SRC) not in sys.path:
    sys.path.append(str(P1_SRC))

from pipeline import build_pillar1_feature_matrix as fm  # type: ignore


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not NORM_INPUT.exists():
        raise FileNotFoundError(f"Normalised features missing: {NORM_INPUT}")
    for req in [DET_PATH, FUSED_PATH, STABLE_PATH]:
        if not req.exists():
            raise FileNotFoundError(f"Required Pillar 4 file missing: {req}")

    # Patch paths inside the module to use live inputs/outputs (mirroring original constants)
    fm.NORM_PATH = NORM_INPUT
    fm.DET_PATH = DET_PATH
    fm.FUSED_PATH = FUSED_PATH
    fm.STABLE_PATH = STABLE_PATH
    fm.DEFAULT_OUTPUT = OUT_PATH

    P1_OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(
        "Building Pillar 1 feature matrix (live) from %s, %s, %s, %s",
        NORM_INPUT,
        DET_PATH,
        FUSED_PATH,
        STABLE_PATH,
    )
    fm.build_pillar1_feature_matrix(save_path=OUT_PATH, limit=None)
    logging.info("Saved live Pillar 1 feature matrix -> %s", OUT_PATH)


if __name__ == "__main__":
    main()
