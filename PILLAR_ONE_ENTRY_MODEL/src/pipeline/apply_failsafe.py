"""
Step 5 — Apply consensus fail-safe to entry_probability.
"""

from pathlib import Path

import pandas as pd

from src.failsafe.consensus_failsafe import apply_consensus_failsafe
from src.failsafe.consensus_failsafe_config import DEFAULT_CONSENSUS_FAILSAFE_CONFIG


INPUT_PATH = Path("data/processed/features_with_entry_probability.parquet")
OUTPUT_PATH = Path("data/processed/features_with_failsafe_probability.parquet")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH).ffill().bfill()
    if "entry_probability" not in df.columns:
        raise ValueError("Missing entry_probability column. Run calibration step first.")

    df = apply_consensus_failsafe(df, DEFAULT_CONSENSUS_FAILSAFE_CONFIG)
    df = df.ffill().bfill()

    if "entry_probability_final" not in df.columns:
        raise ValueError("Failsafe did not produce entry_probability_final.")

    print("Fail-safe applied. Shape:", df.shape)
    print("entry_probability_final summary:")
    print(df["entry_probability_final"].describe())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print("Saved fail-safe probabilities to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
