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


def _neutralize_label_leaks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove future-looking label columns before feeding the ML models.

    The trained bundles unfortunately include y_* labels in their feature list;
    during live/test runs those labels are computed from future price moves,
    so we must zero them out to avoid leakage while keeping the expected
    column names/dimensions for the loaded model artifacts.
    """
    leak_prefixes = ("y_dir_", "y_breakout", "y_mean_reversion")
    leak_cols = [c for c in df.columns if c.startswith(leak_prefixes)]
    if leak_cols:
        logging.warning("[ExitML] Zeroing label/leak columns for inference: %s", leak_cols)
        df.loc[:, leak_cols] = 0.0
    return df


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


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
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
        "meta_agreement_score",
        "meta_conflict_score",
        "meta_vol_weighted_mom",
        "meta_structure_aligned_mom",
        "meta_regime_trend_confirmation",
        "meta_breakout_alignment",
        "meta_horizon_consistency",
        "meta_directional_persistence",
        "raw_entry_confidence",
        "agg_trend_block",
        "agg_reversion_block",
        "agg_ml_block",
        "agg_breakout_block",
        "agg_meta_block",
        "agg_global_gate",
        "y_dir_1_2_lbl",
        "y_dir_5_10_lbl",
        "y_dir_15_30_lbl",
        "y_breakout_lbl",
        "y_mean_reversion_lbl",
        "entry_prob_up",
        "entry_prob_down",
        "entry_calibration_method",
        "entry_prob_source_label",
        "entry_decision",
        "entry_direction",
        "entry_confidence",
        "signal_metadata",
        "regime_label_stable",
        "regime_label_prev",
        "hold_length",
        "smoothing_mode",
        "exit_rv_5",
        "exit_rv_10",
        "exit_rv_20",
        "exit_rv_30",
        "exit_range_norm_5",
        "exit_range_norm_10",
        "exit_range_norm_20",
        "exit_pre_mfe",
        "exit_pre_mae",
        "exit_path_curvature",
        "exit_path_monotonicity",
        "exit_vol_ratio_short_long",
        "exit_vol_short_trend",
        "exit_reversion_z",
        "exit_reversion_abs_z",
        "exit_of_delta_norm",
        "exit_of_delta_ema_norm",
        "exit_recent_sweep_flag",
        "exit_excursion_net",
        "exit_excursion_ratio",
        "exit_conf_weighted_excursion",
        "exit_path_convexity_score",
        "exit_prob_entropy",
        "exit_regime_complexity",
        "exit_vol_scaled_rv_10",
        "exit_reversion_alignment",
        "deterministic_tp",
        "deterministic_sl",
        "deterministic_tp_first_probability",
        "deterministic_exit_time_estimate",
        "mfe_dir",
        "mae_dir",
        "expected_exit_time",
        "exit_outcome",
        "optimal_tp_distance",
        "optimal_sl_distance",
        "optimal_risk_reward",
        "predicted_tp_distance",
        "predicted_sl_distance",
        "predicted_TP_first_probability",
        "predicted_exit_time",
        "predicted_volatility_effect",
    ]
    # Add any missing expected columns so downstream files always contain the full schema.
    missing = [c for c in desired if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    ordered = [c for c in desired]
    remaining = [c for c in df.columns if c not in ordered]
    if missing:
        logging.warning("[ExitML] Missing expected columns were added as NaN: %s", missing)
    return df[ordered + remaining]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Deterministic exit parquet not found: {INPUT_PATH}")

    logging.info("Loading exit_with_deterministic (live): %s", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)
    _log_mem(df, "[ExitML] Loaded deterministic features")

    # Prevent future-label leakage by zeroing y_* columns before model inference.
    df = _neutralize_label_leaks(df)

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

    # Ensure the full expected schema is present and ordered, even if some columns
    # were not produced upstream (e.g., label-derived metrics on empty trade sets).
    df = _reorder_columns(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False, compression="zstd")
    logging.info("Saved Pillar 2 ML predictions (live) -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
