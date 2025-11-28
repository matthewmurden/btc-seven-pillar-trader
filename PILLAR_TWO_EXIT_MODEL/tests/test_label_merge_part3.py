"""Tests for merging features and labels into training dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_engine.feature_engine_master import FeatureEngineMaster  # noqa: E402
from mini_engines.mini_engines_master import MiniEnginesMaster  # noqa: E402
from label_builder.label_builder import LabelBuilder, build_and_save_labels  # noqa: E402
from data_pipeline.merge_features_and_labels import merge_features_and_labels  # noqa: E402


RAW_PATH = PROJECT_ROOT / "data" / "raw" / "exit_raw.parquet"
FEATURES_PART3_PATH = PROJECT_ROOT / "data" / "processed" / "features_part3.parquet"
LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "labels.parquet"
MERGED_PATH = PROJECT_ROOT / "data" / "processed" / "features_part3_with_labels.parquet"

LABEL_COLS = [
    "forward_returns_1m",
    "forward_returns_5m",
    "forward_returns_15m",
    "breakout_label",
    "reversion_label",
]

FEATURE_COLS = [
    "momentum_score",
    "trend_score",
    "volatility_score",
    "reversion_score",
    "orderflow_score",
    "structure_score",
    "pattern_score",
    "regime_state",
]


def ensure_features_part3() -> pd.DataFrame:
    if FEATURES_PART3_PATH.exists():
        return pd.read_parquet(FEATURES_PART3_PATH)
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data missing at {RAW_PATH}")
    raw_df = pd.read_parquet(RAW_PATH)
    feats = FeatureEngineMaster().run_all(raw_df, verbose=False)
    feats = MiniEnginesMaster().run_all(feats, verbose=False)
    FEATURES_PART3_PATH.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(FEATURES_PART3_PATH)
    return feats


def ensure_labels() -> pd.DataFrame:
    if LABELS_PATH.exists():
        existing = pd.read_parquet(LABELS_PATH)
        if existing.get("reversion_label", pd.Series(dtype=int)).sum() > 0:
            return existing
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data missing at {RAW_PATH}")
    raw_df = pd.read_parquet(RAW_PATH)
    build_and_save_labels(raw_df, LABELS_PATH, LabelBuilder())
    return pd.read_parquet(LABELS_PATH)


def ensure_merged() -> pd.DataFrame:
    if MERGED_PATH.exists():
        return pd.read_parquet(MERGED_PATH)
    ensure_features_part3()
    ensure_labels()
    merge_features_and_labels()
    return pd.read_parquet(MERGED_PATH)


def main() -> None:
    df = ensure_merged()

    if len(df) == 0:
        raise AssertionError("Merged dataframe is empty")

    missing_labels = [c for c in LABEL_COLS if c not in df.columns]
    if missing_labels:
        raise AssertionError(f"Missing label columns in merged df: {missing_labels}")

    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        raise AssertionError(f"Missing feature columns in merged df: {missing_features}")

    print("Merged features and labels preview:")
    print(df[FEATURE_COLS + LABEL_COLS].head())


if __name__ == "__main__":
    main()
