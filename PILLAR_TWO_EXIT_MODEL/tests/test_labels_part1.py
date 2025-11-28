"""Tests for label generation (Part 1)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from label_builder.label_builder import LabelBuilder, build_and_save_labels  # noqa: E402


RAW_PATH = PROJECT_ROOT / "data" / "raw" / "exit_raw.parquet"
LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "labels.parquet"
REQUIRED_COLS = [
    "forward_returns_1m",
    "forward_returns_5m",
    "forward_returns_15m",
    "breakout_label",
    "reversion_label",
]


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


def main() -> None:
    labels_df = ensure_labels()

    missing = [c for c in REQUIRED_COLS if c not in labels_df.columns]
    if missing:
        raise AssertionError(f"Missing required label columns: {missing}")

    for col in REQUIRED_COLS:
        series = labels_df[col]
        if series.isna().all():
            raise AssertionError(f"Label column {col} is all NaN")

    for col in ["forward_returns_1m", "forward_returns_5m", "forward_returns_15m"]:
        series = labels_df[col].dropna()
        if not ((series > -0.2) & (series < 0.2)).all():
            raise AssertionError(f"{col} has values outside reasonable bounds [-0.2, 0.2]")

    print("Labels generated and validated:")
    print(labels_df[REQUIRED_COLS].head())


if __name__ == "__main__":
    main()
