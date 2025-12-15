from __future__ import annotations

"""
Live helper to score Pillar 1 multi-horizon ML models using the already
trained models on disk. No training is performed.

Inputs (live):
  RUN_LAYER/data/processed/PILLAR_ONE_ENTRY_MODEL/features_pillar1_with_engines.parquet

Model artifacts (offline trained):
  PILLAR_ONE_ENTRY_MODEL/models/pillar1_multi_horizon/*.pkl

Output (live):
  RUN_LAYER/data/processed/PILLAR_ONE_ENTRY_MODEL/features_pillar1_with_ml.parquet
"""

import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parents[2]
RUN_PROCESSED = BASE_DIR / "RUN_LAYER" / "data" / "processed"

P1_PROCESSED = RUN_PROCESSED / "PILLAR_ONE_ENTRY_MODEL"
FEATURES_PATH = P1_PROCESSED / "features_pillar1_with_engines.parquet"
OUTPUT_PATH = P1_PROCESSED / "features_pillar1_with_ml.parquet"

# Use the offline training template to preserve column ordering/coverage
P1_BASE = BASE_DIR / "PILLAR_ONE_ENTRY_MODEL"
P1_SRC = P1_BASE / "src"
MODEL_DIR = P1_BASE / "models" / "pillar1_multi_horizon"
TEMPLATE_PATH = P1_BASE / "data" / "processed" / "features_pillar1_with_engines.parquet"

if str(P1_SRC) not in sys.path:
    sys.path.append(str(P1_SRC))

from models.pillar1_multi_horizon_ml import (  # type: ignore
    MultiHorizonTrainingConfig,
    default_model_specs,
    load_models,
    _extract_probs,
)


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024**2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def _load_feature_cols(template_path: Path, exclude: List[str]) -> List[str]:
    if not template_path.exists():
        raise FileNotFoundError(f"Template features parquet missing: {template_path}")
    cols = pd.read_parquet(template_path, columns=None).columns.tolist()
    return [c for c in cols if c not in exclude]


def _align_features(df: pd.DataFrame, feature_cols: List[str], dtype: np.dtype) -> np.ndarray:
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    feat_df = df[feature_cols].copy()
    feat_df = feat_df.select_dtypes(include=[np.number]).astype(dtype, copy=False)
    return feat_df.to_numpy(copy=False)


def _slim_output(df: pd.DataFrame, prefixes: List[str]) -> pd.DataFrame:
    """Keep time, core primitives/labels, mini-engine scores, regimes, and prediction columns."""
    keep: List[str] = []
    if "open_time_ms" in df.columns:
        keep.append("open_time_ms")

    base_cols = [
        "open",
        "close",
        "high",
        "low",
        "volume",
        "vol_atr_14",
        "rev_zscore_50",
    ]
    of_struct_cols = [
        "of_delta",
        "of_delta_ema_20",
        "of_imbalance",
        "of_sweep_flag",
        "struct_fair_value",
        "struct_nearest_level_dist",
        "struct_last_sweep_dir",
    ]
    label_cols = [c for c in df.columns if c.startswith("y_")]
    mini_cols = [
        "momentum_score",
        "trend_score",
        "volatility_score",
        "reversion_score",
        "orderflow_score",
        "structure_score",
        "pattern_score",
    ]
    regime_cols = [c for c in df.columns if c.startswith("regime_")]
    pred_cols = [c for c in df.columns if any(c.startswith(pfx) for pfx in prefixes)]

    for group in [base_cols, of_struct_cols, label_cols, mini_cols, regime_cols, pred_cols]:
        for col in group:
            if col in df.columns and col not in keep:
                keep.append(col)

    slim = df[keep].copy()

    float_cols = slim.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        slim[float_cols] = slim[float_cols].astype("float32")

    int_cols = slim.select_dtypes(include=["int64"]).columns
    for col in int_cols:
        col_max = slim[col].max()
        col_min = slim[col].min()
        if col_min >= np.iinfo("int16").min and col_max <= np.iinfo("int16").max:
            slim[col] = slim[col].astype("int16")

    _log_mem(slim, "[MH] Slimmed ML output")
    return slim


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Features parquet not found: {FEATURES_PATH}")

    specs = default_model_specs()
    label_cols = [s.label_col for s in specs]
    feature_cols = _load_feature_cols(TEMPLATE_PATH, exclude=label_cols + ["open_time_ms"])

    cfg = MultiHorizonTrainingConfig(model_dir=MODEL_DIR)
    models = load_models(cfg, specs)

    logging.info("Loading live features_with_engines from %s", FEATURES_PATH)
    df = pd.read_parquet(FEATURES_PATH)
    if "open_time_ms" in df.columns:
        df = df.sort_values("open_time_ms").reset_index(drop=True)
    _log_mem(df, "[MH] Loaded live features_with_engines")

    X = _align_features(df, feature_cols, cfg.feature_dtype)

    logging.info("Scoring multi-horizon models (pre-trained)")
    for spec in specs:
        model = models[spec.name]
        prob_dict = _extract_probs(model, X, spec)
        for k, v in prob_dict.items():
            df[k] = v

    prefixes = [spec.name for spec in specs]
    df_slim = _slim_output(df, prefixes)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_slim.to_parquet(OUTPUT_PATH, index=False, compression="zstd")
    logging.info("Saved live Pillar 1 ML predictions -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
