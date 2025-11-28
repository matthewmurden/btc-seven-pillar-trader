from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.engines.volatility.volatility_engine import compute_volatility_engine_score
from src.engines.volatility.volatility_engine_config import DEFAULT_VOLATILITY_CONFIG


def main():
    data_path = Path("data/processed/features_5m.parquet")
    df = pd.read_parquet(data_path)

    scored = compute_volatility_engine_score(df, DEFAULT_VOLATILITY_CONFIG)
    vol_score = scored["volatility_score"]

    print("Rows:", len(vol_score))
    print("NaN in volatility_score:", vol_score.isnull().sum())

    print("\nVolatility score distribution:")
    print(vol_score.describe(percentiles=[0.01, 0.1, 0.5, 0.9, 0.99]))

    corr = scored[["volatility_score", "atr", "volatility", "vol_percentile"]].corr()
    print("\nCorrelations vs drivers:")
    print(corr["volatility_score"])

    print("\nHead:")
    print(scored[["timestamp", "volatility_score"]].head())
    print("\nTail:")
    print(scored[["timestamp", "volatility_score"]].tail())


if __name__ == "__main__":
    main()
