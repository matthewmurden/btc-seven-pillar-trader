from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.engines.reversion.reversion_engine import compute_reversion_engine_score
from src.engines.reversion.reversion_engine_config import DEFAULT_REVERSION_CONFIG


def main():
    df = pd.read_parquet(Path("data/processed/features_5m.parquet"))
    scored = compute_reversion_engine_score(df, DEFAULT_REVERSION_CONFIG)
    rev = scored["reversion_score"]

    print("Rows:", len(rev))
    print("NaN in reversion_score:", rev.isnull().sum())
    print("\nReversion score distribution:")
    print(rev.describe(percentiles=[0.01, 0.1, 0.5, 0.9, 0.99]))

    corr = scored[["reversion_score", "rsi", "zscore", "volatility"]].corr()
    print("\nCorrelations vs drivers:")
    print(corr["reversion_score"])

    print("\nHead:")
    print(scored[["timestamp", "reversion_score"]].head())
    print("\nTail:")
    print(scored[["timestamp", "reversion_score"]].tail())


if __name__ == "__main__":
    main()
