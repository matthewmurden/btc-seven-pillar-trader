from __future__ import annotations

"""
Live helper for Pillar 2 — ML Exit Models (inference only).

Loads:
  - RUN_LAYER/data/processed/PILLAR_TWO_EXIT_MODEL/exit_with_deterministic.parquet
  - Pre-trained model bundles from PILLAR_TWO_EXIT_MODEL/models/exit_ml/*.joblib

Outputs:
  - RUN_LAYER/data/processed/PILLAR_TWO_EXIT_MODEL/exit_with_ml.parquet
"""

import logging
import sys
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parents[2]
RUN_PROCESSED = BASE_DIR / "RUN_LAYER" / "data" / "processed"

P2_PROCESSED = RUN_PROCESSED / "PILLAR_TWO_EXIT_MODEL"
INPUT_PATH = P2_PROCESSED / "exit_with_deterministic.parquet"
OUTPUT_PATH = P2_PROCESSED / "exit_with_ml.parquet"

P2_BASE = BASE_DIR / "PILLAR_TWO_EXIT_MODEL"
P2_SRC = P2_BASE / "src"
MODEL_DIR = P2_BASE / "models" / "exit_ml"

for p in (P2_BASE, P2_SRC):
    ps = str(p)
    if ps not in sys.path:
        sys.path.append(ps)

from exit.exit_ml_models import (  # type: ignore
    ExitModelSpec,
    default_exit_model_specs,
    _apply_calibrator,
)


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024**2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def _load_bundle(spec: ExitModelSpec):
    path = MODEL_DIR / f"{spec.name}.joblib"
    if not path.exists():
        logging.warning("[ExitML][%s] Model bundle missing: %s", spec.name, path)
        return None
    return joblib.load(path)


def _ensure_feature_cols(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Deterministic exit parquet not found: {INPUT_PATH}")

    logging.info("Loading exit_with_deterministic (live): %s", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)
    _log_mem(df, "[ExitML] Loaded deterministic features")

    specs = default_exit_model_specs()

    for spec in specs:
        bundle = _load_bundle(spec)
        if bundle is None:
            df[spec.pred_col] = np.nan
            continue

        feature_cols = bundle.get("features", [])
        if not feature_cols:
            logging.warning("[ExitML][%s] Bundle missing feature list; skipping.", spec.name)
            df[spec.pred_col] = np.nan
            continue

        df = _ensure_feature_cols(df, feature_cols)
        X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)

        if bundle.get("type") == "regression":
            model = bundle.get("model")
            if model is None:
                df[spec.pred_col] = np.nan
                continue
            y_pred = model.predict(X).astype("float32")
            df[spec.pred_col] = y_pred
        elif bundle.get("type") == "binary":
            model = bundle.get("model")
            calibrator = bundle.get("calibrator")
            if model is None:
                df[spec.pred_col] = np.nan
                continue
            p_raw = model.predict_proba(X)[:, 1]
            p_cal = _apply_calibrator(calibrator, p_raw)
            df[spec.pred_col] = p_cal.astype("float32")
        else:
            logging.warning("[ExitML][%s] Unknown bundle type; filling NaN.", spec.name)
            df[spec.pred_col] = np.nan

    _log_mem(df, "[ExitML] Final df with ML predictions (live)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False, compression="zstd")
    logging.info("Saved Pillar 2 ML predictions (live) -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
