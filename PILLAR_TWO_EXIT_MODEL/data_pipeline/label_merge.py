"""Merge features, labels, and Pillar 1 outputs into aligned training dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .merge_pillar1_data import load_pillar1_outputs, merge_pillar1_and_exit_features

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features_part3.parquet"
LABELS_PATH = BASE_DIR / "data" / "processed" / "labels.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "features_with_labels_and_pillar1.parquet"


def merge_all() -> Path:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Features file missing at {FEATURES_PATH}")
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Labels file missing at {LABELS_PATH}")

    features = pd.read_parquet(FEATURES_PATH)
    labels = pd.read_parquet(LABELS_PATH)
    if "timestamp" not in features.columns or "timestamp" not in labels.columns:
        raise ValueError("Both features and labels must contain timestamp")
    features["timestamp"] = pd.to_datetime(features["timestamp"])
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])
    features = features.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
    labels = labels.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
    merged_features = features.merge(labels, on="timestamp", how="inner")

    pillar1_df = load_pillar1_outputs()
    merged_all = merge_pillar1_and_exit_features(merged_features, pillar1_df)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged_all.to_parquet(OUTPUT_PATH)
    return OUTPUT_PATH


if __name__ == "__main__":
    merge_all()
