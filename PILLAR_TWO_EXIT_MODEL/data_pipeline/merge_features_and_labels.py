"""Merge features and labels into a single training parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features_part3.parquet"
LABELS_PATH = BASE_DIR / "data" / "processed" / "labels.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "features_part3_with_labels.parquet"


def merge_features_and_labels() -> Path:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Features file missing at {FEATURES_PATH}")
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Labels file missing at {LABELS_PATH}")

    features = pd.read_parquet(FEATURES_PATH)
    labels = pd.read_parquet(LABELS_PATH)

    merged = features.join(labels, how="inner")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_PATH)
    print(f"Merged dataset saved: {OUTPUT_PATH.name}")
    return OUTPUT_PATH


if __name__ == "__main__":
    merge_features_and_labels()
