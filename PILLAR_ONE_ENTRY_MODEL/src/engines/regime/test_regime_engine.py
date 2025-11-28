from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.engines.regime.regime_engine import compute_regime_engine_state
from src.engines.regime.regime_engine_config import DEFAULT_REGIME_CONFIG


def main():
    df = pd.read_parquet(Path("data/processed/features_5m.parquet"))
    scored = compute_regime_engine_state(df, DEFAULT_REGIME_CONFIG)

    regime = scored["regime_state"]
    print("Rows:", len(regime))
    print("NaN in regime_state:", regime.isnull().sum())
    print("\nRegime counts:")
    print(regime.value_counts(dropna=False).sort_index())

    print("\nHead:")
    print(scored[["timestamp", "regime_state", "regime_confidence"]].head())
    print("\nTail:")
    print(scored[["timestamp", "regime_state", "regime_confidence"]].tail())

    corr = scored[["regime_confidence", "vol_percentile", "trend_strength", "bb_width"]].corr()
    print("\nCorrelations (confidence vs drivers):")
    print(corr["regime_confidence"])


if __name__ == "__main__":
    main()
