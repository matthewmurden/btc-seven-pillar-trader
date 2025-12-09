"""
Pipeline entrypoint for Pillar 1 — Multi-Horizon ML Predictors.

Loads:
  - data/processed/features_pillar1_with_engines.parquet
  - data/processed/labels_pillar1.parquet

Trains 5 models (short_dir, dir_5_10, dir_15_30, breakout, mean_reversion),
evaluates on a chronological hold-out, saves models, and writes predictions to:
  - data/processed/features_pillar1_with_ml.parquet

Run:
  python -m src.pipeline.build_pillar1_multi_horizon_ml
"""

import logging
from pathlib import Path

import pandas as pd

from src.models.pillar1_multi_horizon_ml import (
    MultiHorizonTrainingConfig,
    default_model_specs,
    train_and_predict,
)

BASE_DIR = Path(__file__).resolve().parents[3]
P1_BASE = BASE_DIR / "PILLAR_ONE_ENTRY_MODEL"

FEATURES_PATH = P1_BASE / "data" / "processed" / "features_pillar1_with_engines.parquet"
LABELS_PATH = P1_BASE / "data" / "processed" / "labels_pillar1.parquet"
OUTPUT_PATH = P1_BASE / "data" / "processed" / "features_pillar1_with_ml.parquet"
MODEL_DIR = P1_BASE / "models" / "pillar1_multi_horizon"


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

    logging.info("Loading features: %s", FEATURES_PATH)
    feats = pd.read_parquet(FEATURES_PATH)
    _log_mem(feats, "[MH] Loaded features")

    logging.info("Loading labels: %s", LABELS_PATH)
    labels = pd.read_parquet(LABELS_PATH)
    _log_mem(labels, "[MH] Loaded labels")

    if "open_time_ms" not in feats.columns:
        raise ValueError("Features parquet is missing 'open_time_ms'.")
    if "open_time_ms" not in labels.columns:
        raise ValueError("Labels parquet is missing 'open_time_ms'.")

    # Ensure chronological order before merge
    feats = feats.sort_values("open_time_ms").reset_index(drop=True)
    labels = labels.sort_values("open_time_ms").reset_index(drop=True)

    logging.info("Merging features and labels on open_time_ms (inner join)")
    n_feats_before = len(feats)
    n_labels_before = len(labels)

    df = feats.merge(labels, on="open_time_ms", how="inner", suffixes=("", "_lbl"))
    _log_mem(df, "[MH] After merge")

    logging.info(
        "[MH] Row counts: features=%d, labels=%d, merged=%d",
        n_feats_before,
        n_labels_before,
        len(df),
    )

    # Basic sanity: if we lost a huge number of rows, warn loudly
    if len(df) < min(n_feats_before, n_labels_before) * 0.5:
        logging.warning(
            "[MH] Merged dataset has < 50%% of rows of min(features, labels). "
            "Check label alignment and horizons."
        )

    # Model specs + training config
    specs = default_model_specs()
    cfg = MultiHorizonTrainingConfig(
        model_dir=MODEL_DIR,
    )

    # Ensure model dir exists before training/saving
    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Training multi-horizon models...")
    df_out = train_and_predict(df, specs=specs, cfg=cfg)

    _log_mem(df_out, "[MH] Final df with predictions")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Saved features + ML predictions -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()