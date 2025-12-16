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
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import InconsistentVersionWarning

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

# Silence sklearn version mismatch chatter (models are already trained)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# -----------------------------------------------------------------------------
# Simple module-level caches to avoid reloading on every call
# -----------------------------------------------------------------------------
_MODEL_CACHE: Optional[Dict[str, object]] = None
_LABELS_CACHE: Optional[pd.DataFrame] = None


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024**2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def _load_feature_cols(template_path: Path, exclude: List[str]) -> List[str]:
    if not template_path.exists():
        raise FileNotFoundError(f"Template features parquet missing: {template_path}")
    cols = pd.read_parquet(template_path, columns=None).columns.tolist()
    return [c for c in cols if c not in exclude]


def _align_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    dtype: np.dtype,
    expected_count: int | None = None,
) -> np.ndarray:
    """
    Align to exactly the columns expected by the model (order matters).
    Missing columns are filled with 0.0; extra columns are ignored.
    All columns are coerced to numeric to avoid drop-by-dtype.
    """
    if expected_count is not None:
        if len(feature_cols) > expected_count:
            logging.warning(
                "[MH] feature_cols (%d) > model expected (%d); trimming to first %d columns",
                len(feature_cols),
                expected_count,
                expected_count,
            )
            feature_cols = feature_cols[:expected_count]
        elif len(feature_cols) < expected_count:
            missing = expected_count - len(feature_cols)
            logging.warning(
                "[MH] feature_cols (%d) < model expected (%d); padding %d zero-columns",
                len(feature_cols),
                expected_count,
                missing,
            )
            feature_cols = feature_cols + [f"__missing_{i}" for i in range(missing)]

    aligned = df.reindex(columns=feature_cols, fill_value=0)
    aligned = aligned.apply(pd.to_numeric, errors="coerce").fillna(0)
    aligned = aligned.astype(dtype, copy=False)
    return aligned.to_numpy(copy=False)


def _load_models_cached(cfg: MultiHorizonTrainingConfig, specs):
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    _MODEL_CACHE = load_models(cfg, specs)
    return _MODEL_CACHE


def _load_labels_cached(label_cols: List[str]) -> Optional[pd.DataFrame]:
    global _LABELS_CACHE
    if _LABELS_CACHE is not None:
        return _LABELS_CACHE
    labels_path = P1_BASE / "data" / "processed" / "labels_pillar1.parquet"
    if not labels_path.exists():
        return None
    df = pd.read_parquet(labels_path)
    if "open_time_ms" not in df.columns:
        return None
    # Keep only needed label columns
    cols = ["open_time_ms"] + [c for c in label_cols if c in df.columns]
    _LABELS_CACHE = df[cols].copy()
    return _LABELS_CACHE


def _slim_output(df: pd.DataFrame, specs) -> pd.DataFrame:
    """
    Produce the exact column set expected downstream (same order as training slim),
    while keeping any extra prediction columns if they exist.
    """
    desired = [
        "open_time_ms",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "vol_atr_14",
        "rev_zscore_50",
        "of_delta",
        "of_delta_ema_20",
        "of_sweep_flag",
        "y_breakout",
        "y_dir_15_30",
        "y_dir_1_2",
        "y_dir_5_10",
        "y_mean_reversion",
        "momentum_score",
        "trend_score",
        "volatility_score",
        "reversion_score",
        "orderflow_score",
        "structure_score",
        "pattern_score",
        "regime_id_stable",
        "regime_confidence",
        "regime_is_trend",
        "regime_is_chop",
        "regime_is_mean_reversion",
        "regime_is_vol_expansion",
        "regime_is_liq_drought",
        "regime_hold_length_norm",
        "regime_chop_mask",
        "mean_reversion_flag",
        "short_dir_prob_up",
        "short_dir_prob_down",
        "short_dir_prob_neutral",
        "short_dir_confidence",
        "short_dir_pred",
        "dir_5_10_prob_up",
        "dir_5_10_prob_down",
        "dir_5_10_prob_neutral",
        "dir_5_10_confidence",
        "dir_5_10_pred",
        "dir_15_30_prob_up",
        "dir_15_30_prob_down",
        "dir_15_30_prob_neutral",
        "dir_15_30_confidence",
        "dir_15_30_pred",
        "breakout_prob_up",
        "breakout_prob_down",
        "breakout_confidence",
        "breakout_pred",
        "mean_reversion_prob_up",
        "mean_reversion_prob_down",
        "mean_reversion_confidence",
        "mean_reversion_pred",
    ]

    # Warn about missing expected columns
    missing = [c for c in desired if c not in df.columns]
    if missing:
        logging.warning("[MH] Missing expected columns (will be skipped): %s", missing)

    # Keep desired in order if present
    keep = [c for c in desired if c in df.columns]

    # Add any extra prediction columns not already included
    pred_prefixes = [spec.name for spec in specs]
    extra_preds = [
        c for c in df.columns if any(c.startswith(pfx) for pfx in pred_prefixes) and c not in keep
    ]
    keep.extend(extra_preds)

    slim = df[keep].copy()

    # Downcast float64 -> float32
    float_cols = slim.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        slim[float_cols] = slim[float_cols].astype("float32")

    # Downcast small ints where safe
    int_cols = slim.select_dtypes(include=["int64"]).columns
    for col in int_cols:
        col_max = slim[col].max()
        col_min = slim[col].min()
        if col_min >= np.iinfo("int16").min and col_max <= np.iinfo("int16").max:
            slim[col] = slim[col].astype("int16")

    _log_mem(slim, "[MH] Slimmed ML output")
    return slim


def _load_existing_ml() -> Optional[pd.DataFrame]:
    """
    Load existing slim ML output if present.
    """
    if not OUTPUT_PATH.exists():
        return None
    df = pd.read_parquet(OUTPUT_PATH)
    if df.empty or "open_time_ms" not in df.columns:
        return None
    return df.sort_values("open_time_ms").reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Features parquet not found: {FEATURES_PATH}")

    specs = default_model_specs()
    label_cols = [s.label_col for s in specs]
    feature_cols = _load_feature_cols(TEMPLATE_PATH, exclude=label_cols + ["open_time_ms"])

    cfg = MultiHorizonTrainingConfig(model_dir=MODEL_DIR)
    models = _load_models_cached(cfg, specs)
    labels_df = _load_labels_cached(label_cols)

    logging.info("Loading live features_with_engines from %s", FEATURES_PATH)
    base_df = pd.read_parquet(FEATURES_PATH)
    if "open_time_ms" in base_df.columns:
        base_df = base_df.sort_values("open_time_ms").reset_index(drop=True)
    _log_mem(base_df, "[MH] Loaded live features_with_engines")

    # Current live window of timestamps
    if "open_time_ms" not in base_df.columns:
        raise RuntimeError("features_pillar1_with_engines.parquet missing 'open_time_ms' column")
    current_times = base_df["open_time_ms"].to_numpy()

    # Load existing slim ML output (if any)
    existing_slim = _load_existing_ml()
    if existing_slim is not None and not existing_slim.empty:
        last_scored_ms = int(existing_slim["open_time_ms"].max())
    else:
        last_scored_ms = -1

    # Determine which rows need (re)scoring: strictly newer than last_scored_ms
    new_mask = base_df["open_time_ms"] > last_scored_ms
    new_df = base_df.loc[new_mask].copy()

    if new_df.empty and existing_slim is not None:
        # No new rows to score; just align existing ML to the current feature window and rewrite
        logging.info(
            "[MH] No new rows to score; aligning existing ML output to current feature window."
        )
        aligned_existing = existing_slim[existing_slim["open_time_ms"].isin(current_times)].copy()
        aligned_existing = aligned_existing.sort_values("open_time_ms").reset_index(drop=True)
        _log_mem(aligned_existing, "[MH] Reused existing ML output (no new rows)")
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        aligned_existing.to_parquet(OUTPUT_PATH, index=False, compression="zstd")
        logging.info(
            "Saved live Pillar 1 ML predictions (reused only) -> %s",
            OUTPUT_PATH,
        )
        return

    if new_df.empty:
        # No existing slim, but somehow nothing to score (should only happen if file just created); fall back to scoring all
        logging.info("[MH] No existing ML output; scoring entire current feature window.")
        new_df = base_df.copy()

    _log_mem(new_df, "[MH] Rows to score (new window subset)")

    logging.info("Scoring multi-horizon models (pre-trained)")

    # If all models share the same feature_names_in_, align once and reuse
    shared_feature_names = None
    shared_expected_count = None
    if all(getattr(models[spec.name], "feature_names_in_", None) is not None for spec in specs):
        first = specs[0].name
        shared_feature_names = list(models[first].feature_names_in_)
        shared_expected_count = getattr(models[first], "n_features_in_", None)
        # sanity: ensure all identical
        for spec in specs[1:]:
            if list(models[spec.name].feature_names_in_) != shared_feature_names:
                shared_feature_names = None
                shared_expected_count = None
                break

    aligned_cache = None
    if shared_feature_names is not None:
        aligned_cache = _align_features(
            new_df,
            shared_feature_names,
            cfg.feature_dtype,
            expected_count=shared_expected_count,
        )

    # Run each spec on the new_df subset
    for spec in specs:
        model = models[spec.name]
        model_feature_cols = shared_feature_names
        expected_count = shared_expected_count

        if model_feature_cols is None:
            model_feature_cols = getattr(model, "feature_names_in_", None)
            if model_feature_cols is not None:
                model_feature_cols = list(model_feature_cols)
                expected_count = getattr(model, "n_features_in_", None)
            else:
                # Fall back to template-based feature list
                model_feature_cols = feature_cols
                expected_count = getattr(model, "n_features_in_", None)

        if aligned_cache is not None and shared_feature_names is not None:
            X = aligned_cache
        else:
            X = _align_features(
                new_df,
                model_feature_cols,
                cfg.feature_dtype,
                expected_count=expected_count,
            )

        prob_dict = _extract_probs(model, X, spec)
        for k, v in prob_dict.items():
            new_df[k] = v

    # Merge labels (if available) for downstream compatibility (only for new rows)
    if labels_df is not None and "open_time_ms" in new_df.columns:
        new_df = new_df.merge(
            labels_df[["open_time_ms"] + label_cols],
            on="open_time_ms",
            how="left",
            suffixes=("", "_lbl"),
        )
        logging.info("[MH] Merged labels into live ML output (new rows only)")

    # Slim just the new rows to the downstream schema
    new_slim = _slim_output(new_df, specs)

    # Combine with existing slim ML, drop duplicates on open_time_ms, and align to current window
    if existing_slim is not None:
        combined = pd.concat([existing_slim, new_slim], ignore_index=True)
    else:
        combined = new_slim

    combined = combined.sort_values("open_time_ms").drop_duplicates(
        subset="open_time_ms", keep="last"
    )

    # Keep only rows that are in the current feature window (so shapes line up downstream)
    combined = combined[combined["open_time_ms"].isin(current_times)].copy()
    combined = combined.sort_values("open_time_ms").reset_index(drop=True)

    _log_mem(combined, "[MH] Final df with ML predictions (live)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False, compression="zstd")
    logging.info("Saved live Pillar 1 ML predictions -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()