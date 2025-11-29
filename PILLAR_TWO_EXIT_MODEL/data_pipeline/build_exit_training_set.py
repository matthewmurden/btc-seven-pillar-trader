"""Builds the full exit training dataset (features + labels + mini engines)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from label_builder.label_builder import LabelBuilder, build_and_save_labels
from feature_engine.exit_features import ExitFeatureEngine
from mini_engines.exit_mini_engines_master import ExitMiniEnginesMaster


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "exit_raw.parquet"
LABELS_PATH = BASE_DIR / "data" / "processed" / "labels.parquet"
FEATURES_PART2_PATH = BASE_DIR / "data" / "processed" / "features_part2.parquet"
FEATURES_PART3_PATH = BASE_DIR / "data" / "processed" / "features_part3.parquet"
TRAINING_PATH = BASE_DIR / "data" / "processed" / "exit_training.parquet"


def build_exit_training_dataset() -> Path:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw exit data missing at {RAW_PATH}")

    raw_df = pd.read_parquet(RAW_PATH)

    # Build labels
    builder = LabelBuilder()
    labels_path = build_and_save_labels(raw_df, LABELS_PATH, builder)
    labels_df = pd.read_parquet(labels_path)

    # Features (Part 2)
    exit_feat_engine = ExitFeatureEngine()
    features_df = exit_feat_engine.transform(raw_df)
    FEATURES_PART2_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(FEATURES_PART2_PATH)

    # Mini engines (Part 3)
    mini_master = ExitMiniEnginesMaster()
    part3_df = mini_master.run_all(features_df, verbose=False)
    FEATURES_PART3_PATH.parent.mkdir(parents=True, exist_ok=True)
    part3_df.to_parquet(FEATURES_PART3_PATH)

    # Merge with labels
    merged = part3_df.join(labels_df, how="inner")
    TRAINING_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(TRAINING_PATH)
    print(f"Exit training dataset saved to {TRAINING_PATH}")
    return TRAINING_PATH


if __name__ == "__main__":
    build_exit_training_dataset()
