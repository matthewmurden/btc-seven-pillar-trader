from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.engines.trend.trend_engine import compute_trend_engine_score
from src.engines.trend.trend_engine_config import TrendConfig


def main():
    data_path = Path("data/processed/features_5m.parquet")
    df = pd.read_parquet(data_path)

    config = TrendConfig()
    scored = compute_trend_engine_score(df, config)
    trend = scored["trend_score"]

    print("Rows:", len(trend))
    print("NaN in trend_score:", trend.isnull().sum())
    print("\nTrend score distribution:")
    print(trend.describe(percentiles=[0.01, 0.1, 0.5, 0.9, 0.99]))

    corr = scored[["trend_score", "trend_slope", "trend_strength", "volatility"]].corr()
    print("\nCorrelations vs drivers:")
    print(corr["trend_score"])

    print("\nHead:")
    print(scored[["timestamp", "trend_score"]].head())
    print("\nTail:")
    print(scored[["timestamp", "trend_score"]].tail())


if __name__ == "__main__":
    main()
