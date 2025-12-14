"""
Pipeline entrypoint for Pillar 1 — Multi-Horizon ML Predictors.

Loads:
  - data/processed/features_pillar1_with_engines.parquet
  - data/processed/labels_pillar1.parquet

Trains 5 models (short_dir, dir_5_10, dir_15_30, breakout, mean_reversion),
evaluates on a chronological hold-out, saves SLIM predictions to:
  - data/processed/features_pillar1_with_ml.parquet

Run:
  python -m src.pipeline.build_pillar1_multi_horizon_ml
"""

import gc
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.models.pillar1_multi_horizon_ml import (
    MultiHorizonTrainingConfig,
    MultiHorizonModelSpec,
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


def _slim_ml_output(df: pd.DataFrame, specs: List[MultiHorizonModelSpec]) -> pd.DataFrame:
    """
    Cut the saved ML file down to what later stages actually need:

    - time index: open_time_ms
    - base OHLCV + key primitives needed for exits
    - label columns (y_*)
    - mini-engine scores
    - regime_* columns
    - orderflow / structure columns used by Pillar 2
    - all model prediction columns (per spec.name)

    This keeps enough context for Pillar 2+ while avoiding dragging thousands
    of raw primitives through every later pillar.
    """
    keep_cols: List[str] = []

    # ---- Time column (always) ----
    if "open_time_ms" in df.columns:
        keep_cols.append("open_time_ms")

    # ---- Base price & core primitives we STILL need downstream (Pillar 2, etc.) ----
    base_cols = [
        "open",
        "close",
        "high",
        "low",
        "volume",
        "vol_atr_14",
        "rev_zscore_50",
    ]
    for col in base_cols:
        if col in df.columns:
            keep_cols.append(col)

    # ---- Orderflow + microstructure columns used in ExitRawConfig ----
    of_struct_cols = [
        "of_delta",
        "of_delta_ema_20",
        "of_imbalance",
        "of_sweep_flag",
        "struct_fair_value",
        "struct_nearest_level_dist",
        "struct_last_sweep_dir",
    ]
    for col in of_struct_cols:
        if col in df.columns:
            keep_cols.append(col)

    # ---- Label columns ----
    label_cols = sorted({spec.label_col for spec in specs if spec.label_col in df.columns})
    keep_cols.extend(label_cols)

    # ---- Mini-engine scores (from pillar1_mini_engines) ----
    for col in [
        "momentum_score",
        "trend_score",
        "volatility_score",
        "reversion_score",
        "orderflow_score",
        "structure_score",
        "pattern_score",
    ]:
        if col in df.columns:
            keep_cols.append(col)

    # ---- Regime-derived features (used across Pillar 1 & 2) ----
    regime_cols = [c for c in df.columns if c.startswith("regime_")]
    keep_cols.extend(regime_cols)

    # ---- All prediction / confidence columns from each model spec ----
    pred_prefixes = [spec.name for spec in specs]
    pred_cols = [c for c in df.columns if any(c.startswith(pfx) for pfx in pred_prefixes)]
    keep_cols.extend(pred_cols)

    # Remove duplicates but keep order, and drop anything that doesn't exist
    seen = set()
    ordered_keep: List[str] = []
    for c in keep_cols:
        if c not in seen and c in df.columns:
            seen.add(c)
            ordered_keep.append(c)

    # Warn if something we expected isn't present (not fatal)
    expected = set(base_cols + of_struct_cols + label_cols)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        logging.warning(
            "[MH] Some expected columns for slim output are missing and will be skipped: %s",
            missing,
        )

    slim = df[ordered_keep].copy()

    # Downcast float64 -> float32 to save disk space
    float_cols = slim.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        slim[float_cols] = slim[float_cols].astype("float32")

    # Downcast small ints where safe (labels, regimes, etc.)
    int_cols = slim.select_dtypes(include=["int64"]).columns
    for col in int_cols:
        col_max = slim[col].max()
        col_min = slim[col].min()
        if col_min >= np.iinfo("int16").min and col_max <= np.iinfo("int16").max:
            slim[col] = slim[col].astype("int16")

    _log_mem(slim, "[MH] Slimmed ML output")
    return slim


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

    # Free raw feats/labels; all info is now in df
    del feats, labels
    gc.collect()

    logging.info(
        "[MH] Row counts: features=%d, labels=%d, merged=%d",
        n_feats_before,
        n_labels_before,
        len(df),
    )

    if len(df) < min(n_feats_before, n_labels_before) * 0.5:
        logging.warning(
            "[MH] Merged dataset has < 50%% of rows of min(features, labels). "
            "Check label alignment and horizons."
        )

    specs = default_model_specs()
    cfg = MultiHorizonTrainingConfig(
        model_dir=MODEL_DIR,
    )
    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Training multi-horizon models (RAM-optimised, class-weighted)...")
    df_with_preds = train_and_predict(df, specs=specs, cfg=cfg)

    # We don't need the full monster file on disk – slim it down
    df_slim = _slim_ml_output(df_with_preds, specs=specs)

    # Save compressed parquet (ZSTD gives good ratio)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_slim.to_parquet(OUTPUT_PATH, index=False, compression="zstd")
    logging.info("Saved SLIM features + ML predictions -> %s", OUTPUT_PATH)

    # Optional: help GC for the rest of the process
    del df, df_with_preds, df_slim
    gc.collect()


if __name__ == "__main__":
    main()