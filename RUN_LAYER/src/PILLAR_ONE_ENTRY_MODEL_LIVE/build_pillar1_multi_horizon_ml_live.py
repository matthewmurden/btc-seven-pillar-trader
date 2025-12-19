from __future__ import annotations

"""
Live scorer for Pillar 1 multi-horizon ML models using pre-trained artifacts.

This mirrors the training pipeline logic in:
  - PILLAR_ONE_ENTRY_MODEL/src/pipeline/build_pillar1_multi_horizon_ml.py
  - PILLAR_ONE_ENTRY_MODEL/src/models/pillar1_multi_horizon_ml.py

Differences:
  - No training; models are loaded from disk and used for inference only.
  - Inputs/outputs are the live RUN_LAYER/data/processed parquet files.
"""

import gc
import logging
import sys
from pathlib import Path
from typing import List
import pyarrow.parquet as pq

import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import InconsistentVersionWarning

BASE_DIR = Path(__file__).resolve().parents[3]
RUN_PROCESSED = BASE_DIR / "RUN_LAYER" / "data" / "processed"

P1_PROCESSED = RUN_PROCESSED / "PILLAR_ONE_ENTRY_MODEL"
FEATURES_PATH = P1_PROCESSED / "features_pillar1_with_engines.parquet"
LABELS_PATH = P1_PROCESSED / "labels_pillar1.parquet"
OUTPUT_PATH = P1_PROCESSED / "features_pillar1_with_ml.parquet"

P1_BASE = BASE_DIR / "PILLAR_ONE_ENTRY_MODEL"
P1_SRC = P1_BASE / "src"
MODEL_DIR = P1_BASE / "models" / "pillar1_multi_horizon"

if str(P1_SRC) not in sys.path:
    sys.path.append(str(P1_SRC))

from models.pillar1_multi_horizon_ml import (  # type: ignore
    MultiHorizonTrainingConfig,
    MultiHorizonModelSpec,
    default_model_specs,
    load_models,
    _extract_probs,
)

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Set False to allow scoring when a label column is entirely NaN (small live windows).
# Set True once your live window is large enough to populate all labels.
STRICT_LABEL_CHECK = False

TRAINING_FEATURES_PATH = P1_BASE / "data" / "processed" / "features_pillar1_with_engines.parquet"
TRAINING_LABELS_PATH = P1_BASE / "data" / "processed" / "labels_pillar1.parquet"


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def _slim_ml_output(df: pd.DataFrame, specs: List[MultiHorizonModelSpec]) -> pd.DataFrame:
    """
    Cut the saved ML file down to what later stages actually need.
    This is copied 1:1 from the training pipeline.
    """
    keep_cols: List[str] = []

    if "open_time_ms" in df.columns:
        keep_cols.append("open_time_ms")

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

    label_cols = sorted({spec.label_col for spec in specs if spec.label_col in df.columns})
    keep_cols.extend(label_cols)

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

    regime_cols = [c for c in df.columns if c.startswith("regime_")]
    keep_cols.extend(regime_cols)

    pred_prefixes = [spec.name for spec in specs]
    pred_cols = [c for c in df.columns if any(c.startswith(pfx) for pfx in pred_prefixes)]
    keep_cols.extend(pred_cols)

    seen = set()
    ordered_keep: List[str] = []
    for c in keep_cols:
        if c not in seen and c in df.columns:
            seen.add(c)
            ordered_keep.append(c)

    expected = set(base_cols + of_struct_cols + label_cols)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        logging.warning(
            "[MH] Some expected columns for slim output are missing and will be skipped: %s",
            missing,
        )

    slim = df[ordered_keep].copy()

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


def _zero_pred_columns(
    df: pd.DataFrame,
    spec: MultiHorizonModelSpec,
    dtype: np.dtype,
) -> None:
    """
    Add zero-filled prediction columns for a spec when we skip scoring.
    Mirrors the column names produced by _extract_probs.
    """
    n = len(df)
    prob_dtype = np.dtype(dtype) if dtype is not None else np.float32
    pred_dtype = np.int16

    if spec.problem_type == "binary":
        df[f"{spec.name}_prob_up"] = np.zeros(n, dtype=prob_dtype)
        df[f"{spec.name}_prob_down"] = np.zeros(n, dtype=prob_dtype)
        df[f"{spec.name}_confidence"] = np.zeros(n, dtype=prob_dtype)
    else:
        df[f"{spec.name}_prob_up"] = np.zeros(n, dtype=prob_dtype)
        df[f"{spec.name}_prob_down"] = np.zeros(n, dtype=prob_dtype)
        df[f"{spec.name}_prob_neutral"] = np.zeros(n, dtype=prob_dtype)
        df[f"{spec.name}_confidence"] = np.zeros(n, dtype=prob_dtype)

    df[f"{spec.name}_pred"] = np.zeros(n, dtype=pred_dtype)


def _load_template_feature_list() -> List[str]:
    """
    Read the training-time feature column order (numeric columns only,
    excluding labels and open_time_ms) from the original training parquet
    schema. This gives us an explicit 731-column template to align to.
    """
    if not TRAINING_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Training features parquet not found: {TRAINING_FEATURES_PATH}"
        )
    if not TRAINING_LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Training labels parquet not found: {TRAINING_LABELS_PATH}"
        )

    pf_feat = pq.ParquetFile(TRAINING_FEATURES_PATH)
    pf_lbl = pq.ParquetFile(TRAINING_LABELS_PATH)

    label_cols = set(pf_lbl.schema.names)
    label_cols.discard("open_time_ms")

    template: List[str] = []
    for i, name in enumerate(pf_feat.schema.names):
        if name in label_cols or name == "open_time_ms":
            continue
        ftype = pf_feat.schema_arrow[i].type
        if ftype is None:
            continue
        from pyarrow import types as pat
        if pat.is_integer(ftype) or pat.is_floating(ftype):
            template.append(name)

    logging.info("[MH] Loaded training template feature list (%d cols)", len(template))
    return template


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

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

    feats = feats.sort_values("open_time_ms").reset_index(drop=True)
    labels = labels.sort_values("open_time_ms").reset_index(drop=True)

    logging.info("Merging features and labels on open_time_ms (inner join)")
    n_feats_before = len(feats)
    n_labels_before = len(labels)
    df = feats.merge(labels, on="open_time_ms", how="inner", suffixes=("", "_lbl"))
    _log_mem(df, "[MH] After merge")

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
    cfg = MultiHorizonTrainingConfig(model_dir=MODEL_DIR)
    models = load_models(cfg, specs)

    label_cols = [spec.label_col for spec in specs]
    drop_cols = set(label_cols)
    if "open_time_ms" in df.columns:
        drop_cols.add("open_time_ms")

    # Align to the exact training feature template (731 columns, fixed order)
    template_features = _load_template_feature_list()

    live_numeric = df.drop(columns=list(drop_cols), errors="ignore").select_dtypes(include=[np.number])
    missing = [c for c in template_features if c not in live_numeric.columns]
    extra = [c for c in live_numeric.columns if c not in template_features]
    if missing:
        logging.warning("[MH] Live data missing %d template features; filling with 0.0: %s", len(missing), missing)
    if extra:
        logging.info("[MH] Dropping %d extra numeric columns not in training template: %s", len(extra), extra)

    # Build aligned feature matrix
    aligned = pd.DataFrame(index=df.index)
    for col in template_features:
        if col in live_numeric.columns:
            aligned[col] = live_numeric[col]
        else:
            aligned[col] = 0.0

    logging.info(
        "[MH] Aligned features to training template: %d cols (expected 731)",
        aligned.shape[1],
    )

    if cfg.feature_dtype:
        aligned = aligned.astype(cfg.feature_dtype, copy=False)

    X_all = aligned.to_numpy(copy=False)
    del aligned, live_numeric
    gc.collect()

    logging.info(
        "[MH] Feature matrix built: shape=%s, dtype=%s",
        X_all.shape,
        X_all.dtype,
    )

    results = df

    for spec in specs:
        label_series = results[spec.label_col]
        mask = label_series.notna().to_numpy()
        if not mask.any():
            if STRICT_LABEL_CHECK:
                raise ValueError(f"No non-NaN rows for label {spec.label_col}")
            logging.warning(
                "[MH] Label %s is all NaN; skipping model '%s' and zero-filling preds.",
                spec.label_col,
                spec.name,
            )
            _zero_pred_columns(results, spec, cfg.feature_dtype)
            continue

        idx_all = np.flatnonzero(mask)
        X_label = X_all[idx_all, :]

        model = models[spec.name]
        prob_dict = _extract_probs(model, X_label, spec)

        for k, v in prob_dict.items():
            if np.issubdtype(v.dtype, np.floating):
                v = v.astype(cfg.feature_dtype, copy=False)
            results.loc[idx_all, k] = v

        del X_label, idx_all
        gc.collect()

    pred_prefixes = [spec.name for spec in specs]
    pred_cols = [c for c in results.columns if any(c.startswith(p) for p in pred_prefixes)]
    if pred_cols:
        results[pred_cols] = results[pred_cols].fillna(0.0)

    df_slim = _slim_ml_output(results, specs=specs)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_slim.to_parquet(OUTPUT_PATH, index=False, compression="zstd")
    logging.info("Saved live Pillar 1 ML predictions -> %s", OUTPUT_PATH)

    del df, results, df_slim
    gc.collect()


if __name__ == "__main__":
    main()
