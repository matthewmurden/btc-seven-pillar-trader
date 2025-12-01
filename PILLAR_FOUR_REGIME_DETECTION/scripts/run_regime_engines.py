# ======================================================
# run_regime_engines.py
# ======================================================

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from regime.deterministic_engines import run_all_engines


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
INPUT_PATH = DATA_DIR / "raw_regime_features.parquet"
OUTPUT_PATH = DATA_DIR / "deterministic_regime_engines.parquet"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input parquet: {INPUT_PATH}")

    logging.info("Loading raw regime features from %s", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)

    logging.info("Running deterministic engines on %s rows", len(df))
    result = run_all_engines(df)

    # Preserve timestamp for downstream fusion
    if "timestamp" in df.columns:
        result.insert(0, "timestamp", df["timestamp"].values)

    result.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Saved deterministic engine outputs → %s", OUTPUT_PATH)
    logging.info("Sample output:\n%s", result.head())


if __name__ == "__main__":
    main()
