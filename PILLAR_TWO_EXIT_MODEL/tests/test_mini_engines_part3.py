"""Validation for Pillar Two Part 3 mini engines."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_engine.feature_engine_master import FeatureEngineMaster  # noqa: E402
from mini_engines.mini_engines_master import MiniEnginesMaster  # noqa: E402


RAW_PATH = PROJECT_ROOT / "data" / "raw" / "exit_raw.parquet"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_part2.parquet"

SCORE_COLS = [
    "momentum_score",
    "trend_score",
    "volatility_score",
    "reversion_score",
    "orderflow_score",
    "structure_score",
    "pattern_score",
    "regime_state",
]


def ensure_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Load or build Part 2 features."""
    if FEATURES_PATH.exists():
        return pd.read_parquet(FEATURES_PATH)
    engine = FeatureEngineMaster()
    features_df = engine.run_all(raw_df, verbose=False)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(FEATURES_PATH)
    return features_df


def assert_numeric_and_finite(df: pd.DataFrame, cols) -> None:
    for col in cols:
        series = df[col]
        if not is_numeric_dtype(series):
            raise TypeError(f"Column {col} is not numeric (dtype={series.dtype})")
        mask = np.isfinite(series.dropna())
        if not mask.all():
            raise ValueError(f"Column {col} contains non-finite values")


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw exit parquet missing at {RAW_PATH}")

    raw_df = pd.read_parquet(RAW_PATH)
    features_df = ensure_features(raw_df)

    mini_master = MiniEnginesMaster()
    enriched_df = mini_master.run_all(features_df, verbose=False)

    if len(enriched_df) != len(features_df):
        raise AssertionError("Mini engines changed dataframe length")

    for col in SCORE_COLS:
        if col not in enriched_df.columns:
            raise AssertionError(f"Missing score column: {col}")

    assert_numeric_and_finite(
        enriched_df,
        [
            "momentum_score",
            "trend_score",
            "volatility_score",
            "reversion_score",
            "orderflow_score",
            "structure_score",
            "pattern_score",
        ],
    )

    regime_vals = enriched_df["regime_state"].dropna().unique().tolist()
    invalid = [v for v in regime_vals if v not in (0, 1, 2)]
    if invalid:
        raise AssertionError(f"Unexpected regime_state values: {invalid}")

    # Range checks
    for col in ["momentum_score", "trend_score", "reversion_score", "orderflow_score", "structure_score"]:
        series = enriched_df[col].dropna()
        if not ((series >= -1).all() and (series <= 1).all()):
            raise AssertionError(f"{col} outside [-1, 1]")
    for col in ["volatility_score", "pattern_score"]:
        series = enriched_df[col].dropna()
        if not ((series >= 0).all() and (series <= 1).all()):
            raise AssertionError(f"{col} outside [0, 1]")

    print("Mini Engines Part 3 passed")
    print(enriched_df[SCORE_COLS].head())


if __name__ == "__main__":
    main()
