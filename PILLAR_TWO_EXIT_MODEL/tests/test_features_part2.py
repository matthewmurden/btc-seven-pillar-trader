"""Validation for Pillar Two Part 2 feature engine."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_engine.feature_engine_master import FeatureEngineMaster  # noqa: E402


RAW_PATH = PROJECT_ROOT / "data" / "raw" / "exit_raw.parquet"

EXPECTED_GROUP_COLUMNS: Dict[str, List[str]] = {
    "momentum": [
        "feature_momentum_roc_5",
        "feature_momentum_roc_10",
        "feature_momentum_roc_20",
        "feature_momentum_mom_short",
        "feature_momentum_mom_medium",
        "feature_momentum_zscore_returns",
        "feature_momentum_obv",
        "feature_momentum_obv_slope",
    ],
    "trend": [
        "feature_trend_slope",
        "feature_trend_strength",
        "feature_trend_htf_alignment",
    ],
    "volatility": [
        "feature_volatility_atr",
        "feature_volatility_atr_expansion",
        "feature_volatility_realized",
        "feature_volatility_percentile",
    ],
    "reversion": [
        "feature_reversion_rsi",
        "feature_reversion_zscore_dev",
        "feature_reversion_rsi_overshoot",
        "feature_reversion_impulse",
    ],
    "orderflow": [
        "feature_orderflow_raw_imbalance",
        "feature_orderflow_norm_imbalance",
        "feature_orderflow_sweep_up",
        "feature_orderflow_sweep_down",
    ],
    "structure": [
        "feature_structure_swing_high",
        "feature_structure_swing_low",
        "feature_structure_bos_up",
        "feature_structure_bos_down",
        "feature_structure_choch_up",
        "feature_structure_choch_down",
        "feature_structure_signal",
    ],
    "pattern": [
        "feature_pattern_pin_bar",
        "feature_pattern_engulfing",
        "feature_pattern_momentum_bar",
        "feature_pattern_wick_signal",
        "feature_pattern_score",
    ],
    "regime": [
        "feature_regime_volatility",
        "feature_regime_trend_vs_range",
        "feature_regime_breakout_compression",
    ],
}


def assert_numeric_and_finite(df: pd.DataFrame, cols: List[str]) -> None:
    for col in cols:
        series = df[col]
        if not is_numeric_dtype(series):
            raise TypeError(f"Column {col} is not numeric (dtype={series.dtype})")
        finite_mask = np.isfinite(series.dropna())
        if not finite_mask.all():
            raise ValueError(f"Column {col} contains non-finite values")


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw exit parquet missing at {RAW_PATH}")

    raw_df = pd.read_parquet(RAW_PATH)
    engine = FeatureEngineMaster()
    features_df = engine.run_all(raw_df, verbose=False)

    if len(features_df) != len(raw_df):
        raise AssertionError("Feature engine changed dataframe length")

    added_columns = set(features_df.columns) - set(raw_df.columns)
    total_added = len(added_columns)

    for group, expected_cols in EXPECTED_GROUP_COLUMNS.items():
        actual_cols = engine.group_columns.get(group, [])
        if not actual_cols:
            raise AssertionError(f"Feature group '{group}' added no columns")
        missing = [col for col in expected_cols if col not in features_df.columns]
        if missing:
            raise AssertionError(
                f"Feature group '{group}' missing expected columns: {missing}"
            )
        assert_numeric_and_finite(features_df, expected_cols)

    print(f"Total features added: {total_added}")
    for group, cols in engine.group_columns.items():
        print(f"{group}: {len(cols)} features -> {cols}")


if __name__ == "__main__":
    main()
