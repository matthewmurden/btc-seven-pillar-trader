from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.engines.orderflow.orderflow_engine import compute_orderflow_engine_score
from src.engines.orderflow.orderflow_engine_config import DEFAULT_ORDERFLOW_CONFIG


def main():
    df = pd.read_parquet(Path("data/processed/features_5m.parquet"))
    scored = compute_orderflow_engine_score(df, DEFAULT_ORDERFLOW_CONFIG)
    ofs = scored["orderflow_score"]

    print("Rows:", len(ofs))
    print("NaN in orderflow_score:", ofs.isnull().sum())
    print("\nOrderflow score distribution:")
    print(ofs.describe(percentiles=[0.01, 0.1, 0.5, 0.9, 0.99]))

    corr = scored[["orderflow_score", "delta", "of_raw_imbalance", "of_imbalance"]].corr()
    print("\nCorrelations vs drivers:")
    print(corr["orderflow_score"])

    print("\nHead:")
    print(scored[["timestamp", "orderflow_score"]].head())
    print("\nTail:")
    print(scored[["timestamp", "orderflow_score"]].tail())


if __name__ == "__main__":
    main()
