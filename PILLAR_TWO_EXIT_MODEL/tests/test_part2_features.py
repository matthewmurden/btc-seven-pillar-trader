"""Tests for Part 2 exit feature engine."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_engine.exit_features import ExitFeatureEngine  # noqa: E402

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "exit_raw.parquet"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_part2.parquet"

REQUIRED = [
    "feature_exit_hl2",
    "feature_exit_hlc3",
    "feature_exit_body_pct",
    "feature_exit_wick_top_pct",
    "feature_exit_wick_bottom_pct",
    "feature_exit_range_pct",
    "feature_exit_spread",
    "feature_exit_vol_percentile",
    "feature_exit_distance_to_liquidity",
    "feature_exit_distance_to_vwap",
    "feature_exit_trend_stretch",
    "feature_exit_vol_compression",
    "feature_exit_volume_expansion",
    "feature_exit_exhaustion_score",
    "feature_exit_bar_clustering",
    "feature_exit_micro_entropy",
    "feature_exit_delta_into_entry",
    "feature_exit_absorption_score",
    "feature_exit_imbalance_persistence",
    "feature_exit_distance_htf_swing",
    "feature_exit_bos_distance",
    "feature_exit_liquidity_alignment",
    "feature_exit_wick_imbalance",
    "feature_exit_climax_volume",
    "feature_exit_atr_slope",
    "feature_exit_realised_vol_past_5",
    "feature_exit_realised_vol_past_15",
    "feature_exit_realised_vol_past_30",
    "feature_exit_realised_vol_past_60",
    "feature_exit_garch_var",
]


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing raw exit parquet at {RAW_PATH}")
    raw_df = pd.read_parquet(RAW_PATH)
    engine = ExitFeatureEngine()
    feats = engine.transform(raw_df)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(FEATURES_PATH)

    missing = [c for c in REQUIRED if c not in feats.columns]
    if missing:
        raise AssertionError(f"Missing required feature columns: {missing}")

    buffer_df = feats.iloc[10:]
    if buffer_df.isna().any().any():
        raise AssertionError("NaNs detected after initial buffer in features")

    print("Part 2 features test passed")
    print(buffer_df[REQUIRED[:8]].head())


if __name__ == "__main__":
    main()
