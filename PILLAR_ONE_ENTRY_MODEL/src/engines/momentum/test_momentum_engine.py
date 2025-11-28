# Momentum Engine Validation Script — FINAL VERSION

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.engines.momentum.momentum_engine import compute_momentum_engine_score
from src.engines.momentum.momentum_engine_config import DEFAULT_MOMENTUM_CONFIG


def main():
    df = pd.read_parquet("data/processed/features_5m.parquet")

    scored = compute_momentum_engine_score(df, DEFAULT_MOMENTUM_CONFIG)

    print("\nRows:", len(scored))
    print("NaN in score:", scored["momentum_score"].isna().sum())

    print("\nDistribution:")
    print(scored["momentum_score"].describe())

    print("\nRaw vs Smoothed correlation:")
    print(scored[["momentum_raw", "momentum_score"]].corr())

    print("\nSample:")
    print(scored[["timestamp", "momentum_raw", "momentum_score"]].head())
    print(scored[["timestamp", "momentum_raw", "momentum_score"]].tail())


if __name__ == "__main__":
    main()