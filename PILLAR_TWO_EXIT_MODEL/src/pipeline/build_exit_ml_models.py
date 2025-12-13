# src/pipeline/build_exit_ml_models.py

"""
Pipeline entrypoint for Pillar 2 — ML Exit Models (Core Predictors).

Loads:
  - data/processed/exit_with_deterministic.parquet
  - data/processed/exit_labels.parquet

Merges them on `open_time_ms`, trains the suite of Pillar 2 exit models,
and writes predictions back to:

  - data/processed/exit_with_ml.parquet

Models (and calibrators) are saved to:
  - models/exit_ml/*.joblib

Run:

  python -m src.pipeline.build_exit_ml_models
"""

import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.exit.exit_ml_models import (
    ExitMLConfig,
    default_exit_model_specs,
    train_and_predict_exit_models,
)

BASE_DIR = Path(__file__).resolve().parents[3]
P2_BASE = BASE_DIR / "PILLAR_TWO_EXIT_MODEL"

EXIT_DET_PATH = P2_BASE / "data" / "processed" / "exit_with_deterministic.parquet"
EXIT_LABELS_PATH = P2_BASE / "data" / "processed" / "exit_labels.parquet"
OUTPUT_PATH = P2_BASE / "data" / "processed" / "exit_with_ml.parquet"
MODEL_DIR = P2_BASE / "models" / "exit_ml"


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024**2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def _downcast_floats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast float64 → float32 to reduce disk usage.
    """
    float_cols = df.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype("float32")
    return df


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not EXIT_DET_PATH.exists():
        raise FileNotFoundError(f"Deterministic exit parquet not found: {EXIT_DET_PATH}")
    if not EXIT_LABELS_PATH.exists():
        raise FileNotFoundError(f"Exit labels parquet not found: {EXIT_LABELS_PATH}")

    logging.info("Loading deterministic exit features: %s", EXIT_DET_PATH)
    det = pd.read_parquet(EXIT_DET_PATH)
    _log_mem(det, "[ExitML] Loaded exit_with_deterministic")

    logging.info("Loading exit labels: %s", EXIT_LABELS_PATH)
    labels = pd.read_parquet(EXIT_LABELS_PATH)
    _log_mem(labels, "[ExitML] Loaded exit_labels")

    if "open_time_ms" not in det.columns:
        raise ValueError("exit_with_deterministic is missing 'open_time_ms'.")
    if "open_time_ms" not in labels.columns:
        raise ValueError("exit_labels is missing 'open_time_ms'.")

    # Ensure chronological order before merge (safety)
    det = det.sort_values("open_time_ms").reset_index(drop=True)
    labels = labels.sort_values("open_time_ms").reset_index(drop=True)

    logging.info("Merging deterministic exit + labels on open_time_ms (inner join)")
    n_det_before = len(det)
    n_lbl_before = len(labels)

    df = det.merge(labels, on="open_time_ms", how="inner", suffixes=("", "_lbl"))
    _log_mem(df, "[ExitML] After merge")

    del det, labels
    gc.collect()

    logging.info(
        "[ExitML] Row counts: det=%d, labels=%d, merged=%d",
        n_det_before,
        n_lbl_before,
        len(df),
    )

    if len(df) < min(n_det_before, n_lbl_before) * 0.5:
        logging.warning(
            "[ExitML] Merged dataset has < 50%% of rows of min(det, labels). "
            "Check alignment and horizons."
        )

    specs = default_exit_model_specs()
    cfg = ExitMLConfig(
        model_dir=MODEL_DIR,
        val_fraction=0.2,
        random_state=42,
        min_train_rows=10_000,
        calibrate_classifiers=True,
        calib_method="auto",
        col_time="open_time_ms",
    )
    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Training Pillar 2 exit ML models...")
    df_with_preds = train_and_predict_exit_models(df, specs=specs, cfg=cfg)

    # Downcast floats before saving to keep file manageable
    df_with_preds = _downcast_floats(df_with_preds)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_with_preds.to_parquet(OUTPUT_PATH, index=False, compression="zstd")
    logging.info("Saved Pillar 2 features + ML exit predictions -> %s", OUTPUT_PATH)

    del df, df_with_preds
    gc.collect()


if __name__ == "__main__":
    main()