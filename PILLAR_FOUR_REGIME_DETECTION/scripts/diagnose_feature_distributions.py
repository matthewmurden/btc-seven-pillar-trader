import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "processed" / "raw_regime_features.parquet"


def diagnose():
    df = pd.read_parquet(RAW)
    logging.info(f"Loaded raw dataset: {df.shape}")

    print("\n=== FEATURE DISTRIBUTIONS (SUMMARY) ===")
    print(df.describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T)

    # Save full stats to file
    out = PROJECT_ROOT / "data" / "processed" / "feature_distribution_stats.csv"
    df.describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T.to_csv(out)
    logging.info(f"Saved distribution stats to: {out}")

    # Check which features are always 0 / very low variance
    var = df.var()
    low_var = var[var < 1e-8]
    if len(low_var) > 0:
        print("\n=== LOW VARIANCE FEATURES ===")
        print(low_var)

    # Show 20 random rows for context
    print("\n=== SAMPLE ROWS (20) ===")
    print(df.sample(20))


if __name__ == "__main__":
    diagnose()