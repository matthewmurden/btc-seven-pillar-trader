from __future__ import annotations

"""
Live helper for Pillar 2 — Fusion Layer.

Input:
  RUN_LAYER/data/processed/PILLAR_TWO_EXIT_MODEL/exit_with_ml.parquet

Output:
  RUN_LAYER/data/processed/PILLAR_TWO_EXIT_MODEL/exit_with_fusion.parquet
"""

import gc
import logging
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parents[2]
RUN_PROCESSED = BASE_DIR / "RUN_LAYER" / "data" / "processed"

P2_PROCESSED = RUN_PROCESSED / "PILLAR_TWO_EXIT_MODEL"
INPUT_PATH = P2_PROCESSED / "exit_with_ml.parquet"
OUTPUT_PATH = P2_PROCESSED / "exit_with_fusion.parquet"

P2_BASE = BASE_DIR / "PILLAR_TWO_EXIT_MODEL"
P2_SRC = P2_BASE / "src"
for p in (P2_BASE, P2_SRC):
    ps = str(p)
    if ps not in sys.path:
        sys.path.append(ps)

from exit.exit_fusion import ExitFusionConfig, fuse_exit_signals  # type: ignore
import numpy as np


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024**2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


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
        "fused_tp",
        "fused_sl",
        "fused_tp_first_probability",
        "fused_exit_time",
    ]

    missing = [c for c in desired if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    ordered = desired
    remaining = [c for c in df.columns if c not in ordered]
    if missing:
        logging.warning("[ExitFusion] Missing expected columns added as NaN: %s", missing)
    return df[ordered + remaining]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"exit_with_ml parquet not found: {INPUT_PATH}")

    logging.info("Loading exit_with_ml (live): %s", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)
    _log_mem(df, "[ExitFusion] Loaded exit_with_ml")

    cfg = ExitFusionConfig()
    df_fused = fuse_exit_signals(df, cfg=cfg)
    df_fused = _reorder_columns(df_fused)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_fused.to_parquet(OUTPUT_PATH, index=False, compression="zstd")
    logging.info("Saved Pillar 2 fused exits (live) -> %s", OUTPUT_PATH)

    del df, df_fused
    gc.collect()


if __name__ == "__main__":
    main()
