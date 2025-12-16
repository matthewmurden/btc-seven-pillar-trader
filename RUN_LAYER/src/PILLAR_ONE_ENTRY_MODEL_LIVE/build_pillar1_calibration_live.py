from __future__ import annotations

"""
Live helper to apply the Pillar 1 calibration layer using pre-trained
calibration artifacts. No fitting is performed here.

Inputs:
  RUN_LAYER/data/processed/PILLAR_ONE_ENTRY_MODEL/features_pillar1_with_agg.parquet

Model artifact (offline-trained):
  PILLAR_ONE_ENTRY_MODEL/models/pillar1_calibration/pillar1_calibrator.joblib

Output:
  RUN_LAYER/data/processed/PILLAR_ONE_ENTRY_MODEL/features_pillar1_with_calib.parquet
"""

import logging
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parents[2]
RUN_PROCESSED = BASE_DIR / "RUN_LAYER" / "data" / "processed"

P1_PROCESSED = RUN_PROCESSED / "PILLAR_ONE_ENTRY_MODEL"
FEATURES_PATH = P1_PROCESSED / "features_pillar1_with_agg.parquet"
OUTPUT_PATH = P1_PROCESSED / "features_pillar1_with_calib.parquet"

# Pillar 1 src (add both project root and src so pickled modules under "src.*" resolve)
P1_BASE = BASE_DIR / "PILLAR_ONE_ENTRY_MODEL"
P1_SRC = P1_BASE / "src"
for p in (P1_BASE, P1_SRC):
    ps = str(p)
    if ps not in sys.path:
        sys.path.append(ps)

from models.pillar1_calibration import (  # type: ignore
    CalibrationConfig,
    load_calibrator,
    apply_calibration_to_df,
)


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def _merge_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach training labels (y_*) for downstream consumers that expect *_lbl columns.
    If labels are missing, we warn and return df unchanged.
    """
    labels_path = P1_BASE / "data" / "processed" / "labels_pillar1.parquet"
    if not labels_path.exists():
        logging.warning("Labels file missing; skipping label merge: %s", labels_path)
        return df

    labels_df = pd.read_parquet(labels_path)
    if "open_time_ms" not in labels_df.columns:
        logging.warning("Labels file missing open_time_ms; skipping label merge.")
        return df

    wanted = ["y_dir_1_2", "y_dir_5_10", "y_dir_15_30", "y_breakout", "y_mean_reversion"]
    present = [c for c in wanted if c in labels_df.columns]
    if not present:
        logging.warning("No expected y_* columns found in labels; skipping label merge.")
        return df

    labels_df = labels_df[["open_time_ms"] + present].rename(columns={c: f"{c}_lbl" for c in present})
    merged = df.merge(labels_df, on="open_time_ms", how="left")
    logging.info("Merged labels into calibration output (cols=%s)", list(labels_df.columns))
    return merged


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
    ]

    ordered = [c for c in desired if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered]
    return df[ordered + remaining]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Features parquet not found: {FEATURES_PATH}")

    logging.info("Loading features_with_agg (live) from %s", FEATURES_PATH)
    df = pd.read_parquet(FEATURES_PATH)
    _log_mem(df, "[Calib] Loaded features_with_agg")

    cfg = CalibrationConfig(model_dir=P1_BASE / "models" / "pillar1_calibration")
    calibrator = load_calibrator(cfg=cfg)

    logging.info("Applying pre-trained calibrator to live data...")
    df_out = apply_calibration_to_df(df, calibrator, cfg=cfg)
    _log_mem(df_out, "[Calib] After applying calibration")

    # Attach label columns for downstream compatibility
    df_out = _merge_labels(df_out)
    _log_mem(df_out, "[Calib] After label merge")

    # Reorder columns to expected schema, keep extras appended
    df_out = _reorder_columns(df_out)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Saved calibrated Pillar 1 outputs (live) -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
