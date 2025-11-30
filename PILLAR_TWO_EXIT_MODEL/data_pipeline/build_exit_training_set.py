"""Build the full exit training dataset with strict timestamp alignment."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from utils.timestamp_utils import normalise_timestamp

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "exit_raw.parquet"
LABELS_PATH = BASE_DIR / "data" / "processed" / "labels.parquet"
FEATURES_WITH_LABELS_PATH = BASE_DIR / "data" / "processed" / "features_part3_with_labels.parquet"


def build_exit_training_dataset() -> Path:
    exit_raw = normalise_timestamp(pd.read_parquet(RAW_PATH))
    labels = normalise_timestamp(pd.read_parquet(LABELS_PATH))

    merged = exit_raw.merge(labels, on="timestamp", how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    FEATURES_WITH_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(FEATURES_WITH_LABELS_PATH, index=False)

    print("Exit training dataset saved:", FEATURES_WITH_LABELS_PATH)
    print("Rows:", len(merged))
    print("Columns:", len(merged.columns))
    print("Timestamp range:", merged['timestamp'].min(), "→", merged['timestamp'].max())
    return FEATURES_WITH_LABELS_PATH


if __name__ == "__main__":
    build_exit_training_dataset()
