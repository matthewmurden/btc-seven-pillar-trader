"""Build aligned training set restricted to Pillar 1 entries."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MERGED_PATH = BASE_DIR / "data" / "processed" / "features_with_labels_and_pillar1.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "exit_training_aligned.parquet"

ENTRY_PROBABILITY_THRESHOLD = 0.6
INVALID_REGIME_STATE = -1


def build_aligned_training_set() -> Path:
    if not MERGED_PATH.exists():
        raise FileNotFoundError(f"Merged dataset missing at {MERGED_PATH}")
    df = pd.read_parquet(MERGED_PATH)

    df = df[df["entry_signal"] == 1]
    df = df[df["entry_probability"] >= ENTRY_PROBABILITY_THRESHOLD]
    if "valid_exit_signal" in df.columns:
        df = df[df["valid_exit_signal"] == 1]
    if "regime_state" in df.columns:
        df = df[df["regime_state"] != INVALID_REGIME_STATE]

    # Drop extreme outliers on returns/labels if present
    for col in ["MFE_pct", "MAE_pct", "forward_returns_1m", "forward_returns_5m", "forward_returns_15m"]:
        if col in df.columns:
            df = df[df[col].between(-0.2, 0.2)]

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH)
    return OUTPUT_PATH


if __name__ == "__main__":
    build_aligned_training_set()
