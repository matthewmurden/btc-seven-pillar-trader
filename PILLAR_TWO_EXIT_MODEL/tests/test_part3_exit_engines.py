"""Tests for Part 3 exit mini engines."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_engine.exit_features import ExitFeatureEngine  # noqa: E402
from mini_engines.exit_mini_engines_master import ExitMiniEnginesMaster  # noqa: E402

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "exit_raw.parquet"

REQUIRED = [
    "excursion_score",
    "mae_risk_score",
    "mfe_potential_score",
    "path_entropy_score",
    "exit_regime",
]


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing raw exit parquet at {RAW_PATH}")

    raw_df = pd.read_parquet(RAW_PATH)
    feats = ExitFeatureEngine().transform(raw_df)
    engines_df = ExitMiniEnginesMaster().run_all(feats, verbose=False)

    missing = [c for c in REQUIRED if c not in engines_df.columns]
    if missing:
        raise AssertionError(f"Missing engine outputs: {missing}")

    for col in REQUIRED[:-1]:
        vals = engines_df[col].dropna()
        if not ((vals >= 0).all() and (vals <= 1).all()):
            raise AssertionError(f"{col} outside [0,1]")

    print("Part 3 exit mini engines test passed")
    print(engines_df[REQUIRED].head())


if __name__ == "__main__":
    main()
