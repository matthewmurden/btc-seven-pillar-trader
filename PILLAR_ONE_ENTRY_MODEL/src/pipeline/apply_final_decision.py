"""
Step 6 — Apply final decision layer to produce final_signal.
"""

from pathlib import Path

import pandas as pd

from src.decision.final_decision_layer import compute_final_signal
from src.decision.final_decision_config import DEFAULT_FINAL_DECISION_CONFIG


INPUT_PATH = Path("data/processed/features_with_failsafe_probability.parquet")
OUTPUT_PATH = Path("data/processed/features_with_final_signal.parquet")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH).ffill().bfill()
    if "entry_probability_final" not in df.columns:
        raise ValueError("Missing entry_probability_final column. Run fail-safe step first.")

    if not df["entry_probability_final"].between(0.0, 1.0).all():
        raise ValueError("entry_probability_final contains values outside [0, 1].")

    df = compute_final_signal(df, DEFAULT_FINAL_DECISION_CONFIG)
    df = df.ffill().bfill()

    if "final_signal" not in df.columns:
        raise ValueError("Final decision layer did not produce final_signal.")

    counts = df["final_signal"].value_counts()
    print("Final decision applied. Shape:", df.shape)
    print("Signal counts:")
    print(counts)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print("Saved final signals to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
