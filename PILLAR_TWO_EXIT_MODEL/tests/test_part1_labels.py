"""Strict tests for Part 1 raw exit data + label generation."""

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

REQUIRED = [
    "MFE_raw",
    "MAE_raw",
    "MFE_pct",
    "MAE_pct",
    "tp_first",
    "sl_first",
    "exit_bar",
    "first_touch_tp_distance",
    "first_touch_sl_distance",
    "forward_returns_1m",
    "forward_returns_5m",
    "forward_returns_15m",
    "breakout_label",
    "reversion_label",
    "next_1_to_300_returns",
]


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing raw exit parquet at {RAW_PATH}")

    raw_df = pd.read_parquet(RAW_PATH)
    builder = LabelBuilder()
    labels_path = build_and_save_labels(raw_df, LABELS_PATH, builder)
    labels = pd.read_parquet(labels_path)

    missing = [c for c in REQUIRED if c not in labels.columns]
    if missing:
        raise AssertionError(f"Missing label columns: {missing}")

    if len(labels) != len(raw_df):
        raise AssertionError("Label length mismatch with raw data")

    # No leakage: check that forward returns use future bars (last rows should be NaN)
    tail_returns = labels["forward_returns_15m"].tail(20)
    if tail_returns.isna().sum() == 0:
        raise AssertionError("Expected NaNs at tail for forward returns (no leakage)")

    # Exit bar sanity: must be positive or NaN, and less than horizon+1
    if not labels["exit_bar"].dropna().between(1, builder.horizon).all():
        raise AssertionError("exit_bar out of expected range")

    print("Part 1 labels test passed")
    print(labels.head()[REQUIRED])


if __name__ == "__main__":
    main()
