"""Master runner for meta exit features (Part 5)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .meta_exit_engine import build_meta_features, OUTPUT_PATH


PREDICTIONS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "exit_with_predictions.parquet"


def run_meta_features() -> Path:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Prediction parquet missing at {PREDICTIONS_PATH}")
    df = pd.read_parquet(PREDICTIONS_PATH)
    build_meta_features(df)
    return OUTPUT_PATH


if __name__ == "__main__":
    run_meta_features()
