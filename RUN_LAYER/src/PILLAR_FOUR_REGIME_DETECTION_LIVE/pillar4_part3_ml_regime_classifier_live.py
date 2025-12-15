from __future__ import annotations

"""
Live Pillar 4 Part 3: use the pre-trained regime classifier to generate
regime_ml_predictions.parquet on live/raw features.

Inputs:
  - RUN_LAYER/data/processed/PILLAR_FOUR_REGIME_DETECTION/raw_regime_features.parquet
  - RUN_LAYER/data/processed/PILLAR_FOUR_REGIME_DETECTION/deterministic_regime_engines.parquet
  - Trained model: PILLAR_FOUR_REGIME_DETECTION/data/processed/part_three/regime_ml_model.pkl

Output:
  - RUN_LAYER/data/processed/PILLAR_FOUR_REGIME_DETECTION/regime_ml_predictions.parquet
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parents[2]
RUN_PROCESSED = BASE_DIR / "RUN_LAYER" / "data" / "processed"
P4_OUT = RUN_PROCESSED / "PILLAR_FOUR_REGIME_DETECTION"

RAW_FEATURES_PATH = P4_OUT / "raw_regime_features.parquet"
DETERMINISTIC_PATH = P4_OUT / "deterministic_regime_engines.parquet"
OUTPUT_PATH = P4_OUT / "regime_ml_predictions.parquet"

# Pre-trained model in the source Pillar 4 repo
P4_BASE = BASE_DIR / "PILLAR_FOUR_REGIME_DETECTION"
MODEL_PATH = P4_BASE / "data" / "processed" / "part_three" / "regime_ml_model.pkl"

P4_SRC = P4_BASE / "src"
if str(P4_SRC) not in sys.path:
    sys.path.append(str(P4_SRC))

from part_three import pillar4_part3_ml_regime_classifier as p3  # type: ignore


def _load_model() -> object:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Pre-trained regime model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def _prepare_features(raw_feats: pd.DataFrame, det: pd.DataFrame) -> pd.DataFrame:
    # Reuse prepare_features logic from the original script
    return p3.prepare_features(raw_feats, det)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not RAW_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Raw regime features missing: {RAW_FEATURES_PATH}")
    if not DETERMINISTIC_PATH.exists():
        raise FileNotFoundError(f"Deterministic regime engines missing: {DETERMINISTIC_PATH}")

    raw_feats = pd.read_parquet(RAW_FEATURES_PATH)
    det = pd.read_parquet(DETERMINISTIC_PATH)

    df = _prepare_features(raw_feats, det)
    feature_df = df.drop(columns=[c for c in df.columns if c in ["open_time_ms", "open_time"]], errors="ignore")
    feature_df = p3.encode_categoricals(feature_df).fillna(0)

    model = _load_model()
    preds = p3.predict(model, feature_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Saved live regime ML predictions -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
