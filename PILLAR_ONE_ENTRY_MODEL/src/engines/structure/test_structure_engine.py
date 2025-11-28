from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.engines.structure.structure_engine import compute_structure_engine_score
from src.engines.structure.structure_engine_config import DEFAULT_STRUCTURE_CONFIG


def main():
    df = pd.read_parquet(Path("data/processed/features_5m.parquet"))
    scored = compute_structure_engine_score(df, DEFAULT_STRUCTURE_CONFIG)
    ss = scored["structure_score"]

    print("Rows:", len(ss))
    print("NaN in structure_score:", ss.isnull().sum())
    print("\nStructure score distribution:")
    print(ss.describe(percentiles=[0.01, 0.1, 0.5, 0.9, 0.99]))

    corr = scored[["structure_score", "bos_up", "bos_down", "structure_signal"]].corr()
    print("\nCorrelations vs drivers:")
    print(corr["structure_score"])

    print("\nHead:")
    print(scored[["timestamp", "structure_score"]].head())
    print("\nTail:")
    print(scored[["timestamp", "structure_score"]].tail())


if __name__ == "__main__":
    main()
