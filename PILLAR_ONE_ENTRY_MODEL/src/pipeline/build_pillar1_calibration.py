"""
Pipeline entrypoint for Pillar 1 — Calibration Layer.

Inputs:
  - data/processed/features_pillar1_with_agg.parquet
    (includes raw_entry_confidence from master aggregator)
  - data/processed/labels_pillar1.parquet
    (includes y_dir_5_10 etc.)

Steps:
  1. Merge on open_time_ms (inner join), chronologically sorted.
  2. Fit a 1D calibrator from raw_entry_confidence → P(up) using:
       - Platt scaling
       - Isotonic regression
       - AUTO chooses best by Brier score.
  3. Apply calibrator to all rows to obtain:
       - entry_prob_up
       - entry_prob_down
  4. Save:
       - models/pillar1_calibration/pillar1_calibrator.joblib
       - data/processed/features_pillar1_with_calib.parquet

Run:
  python -m src.pipeline.build_pillar1_calibration
"""

import logging
from pathlib import Path

import pandas as pd

from src.models.pillar1_calibration import (
    CalibrationConfig,
    apply_calibration_to_df,
    fit_calibrator,
    save_calibrator,
)

BASE_DIR = Path(__file__).resolve().parents[3]
P1_BASE = BASE_DIR / "PILLAR_ONE_ENTRY_MODEL"

FEATURES_PATH = P1_BASE / "data" / "processed" / "features_pillar1_with_agg.parquet"
LABELS_PATH = P1_BASE / "data" / "processed" / "labels_pillar1.parquet"
OUTPUT_PATH = P1_BASE / "data" / "processed" / "features_pillar1_with_calib.parquet"
MODEL_DIR = P1_BASE / "models" / "pillar1_calibration"


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Features parquet not found: {FEATURES_PATH}")
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Labels parquet not found: {LABELS_PATH}")

    logging.info("Loading features (with agg): %s", FEATURES_PATH)
    feats = pd.read_parquet(FEATURES_PATH)
    _log_mem(feats, "[Calib] Loaded features_with_agg")

    logging.info("Loading labels: %s", LABELS_PATH)
    labels = pd.read_parquet(LABELS_PATH)
    _log_mem(labels, "[Calib] Loaded labels")

    if "open_time_ms" not in feats.columns:
        raise ValueError("Features parquet is missing 'open_time_ms'.")
    if "open_time_ms" not in labels.columns:
        raise ValueError("Labels parquet is missing 'open_time_ms'.")

    feats = feats.sort_values("open_time_ms").reset_index(drop=True)
    labels = labels.sort_values("open_time_ms").reset_index(drop=True)

    logging.info("Merging features + labels on open_time_ms (inner join)")
    n_feats_before = len(feats)
    n_labels_before = len(labels)

    df = feats.merge(labels, on="open_time_ms", how="inner", suffixes=("", "_lbl"))
    _log_mem(df, "[Calib] After merge")
    logging.info(
        "[Calib] Row counts: features=%d, labels=%d, merged=%d",
        n_feats_before,
        n_labels_before,
        len(df),
    )

    if len(df) < min(n_feats_before, n_labels_before) * 0.5:
        logging.warning(
            "[Calib] Merged dataset has < 50%% of rows of min(features, labels). "
            "Check label alignment and horizons."
        )

    cfg = CalibrationConfig(
        target_label_col="y_dir_5_10",
        method="auto",
        model_dir=MODEL_DIR,
    )

    logging.info("Fitting calibration layer with config: %s", cfg)
    calibrator = fit_calibrator(df, cfg=cfg)
    save_calibrator(calibrator, cfg=cfg)

    logging.info("Applying calibrator to full feature set...")
    df_out = apply_calibration_to_df(df, calibrator, cfg=cfg)
    _log_mem(df_out, "[Calib] Final df with entry probabilities")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Saved features + calibrated entry probabilities -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()