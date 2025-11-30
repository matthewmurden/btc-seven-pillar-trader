"""
Apply all mini-exit engines to exit_features.parquet and save augmented dataset.

Usage:
    python -m mini_engines.apply_all_mini_engines
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from mini_engines.adverse_move import compute_adverse_move
from mini_engines.favourable_move import compute_favourable_move
from mini_engines.path_shape import compute_path_shape
from mini_engines.regime_exit import compute_regime_exit
from mini_engines.volatility_excursion import compute_volatility_excursion


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "processed" / "exit_features.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "exit_features_with_mini_engines.parquet"


def apply_all_engines(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_adverse_move(df)
    df = compute_favourable_move(df)
    df = compute_path_shape(df)
    df = compute_regime_exit(df)
    df = compute_volatility_excursion(df)
    return df


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    logger.info("Loading exit features from %s", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)
    out = apply_all_engines(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Saved augmented features with mini engines to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
