from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.engines.pattern.pattern_engine import compute_pattern_engine_score
from src.engines.pattern.pattern_engine_config import DEFAULT_PATTERN_CONFIG


def main():
    df = pd.read_parquet(Path("data/processed/features_5m.parquet"))
    scored = compute_pattern_engine_score(df, DEFAULT_PATTERN_CONFIG)
    ps = scored["pattern_score"]

    print("Rows:", len(ps))
    print("NaN in pattern_score:", ps.isnull().sum())
    print("\nPattern score distribution:")
    print(ps.describe(percentiles=[0.01, 0.1, 0.5, 0.9, 0.99]))

    corr = scored[["pattern_score", "pin_bar", "momentum_bar", "engulfing"]].corr()
    print("\nCorrelations vs drivers:")
    print(corr["pattern_score"])

    print("\nHead:")
    print(scored[["timestamp", "pattern_score"]].head())
    print("\nTail:")
    print(scored[["timestamp", "pattern_score"]].tail())


if __name__ == "__main__":
    main()
