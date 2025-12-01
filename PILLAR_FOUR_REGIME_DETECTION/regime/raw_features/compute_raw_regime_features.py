"""
Hybrid Raw Feature Generator for Pillar 4 Regime Detection.
Loads Pillar 1 enriched dataset and adds Pillar 4-specific regime features.
"""

from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd

from .feature_utils import (
    compute_trend_features,
    compute_volatility_features,
    compute_liquidity_features,
    compute_microstructure_features,
    compute_structure_features,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
P1_PATH = PROJECT_ROOT / "PILLAR_ONE_ENTRY_MODEL" / "data" / "processed" / "features_with_final_signal.parquet"
OUT_PATH = PROJECT_ROOT / "PILLAR_FOUR_REGIME_DETECTION" / "data" / "processed" / "raw_regime_features.parquet"


def compute_raw_regime_features() -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not P1_PATH.exists():
        raise FileNotFoundError(f"Pillar 1 enriched dataset not found: {P1_PATH}")

    logger.info("Loading base enriched dataset from Pillar 1…")
    df = pd.read_parquet(P1_PATH)

    logger.info("Computing hybrid Pillar 4 raw regime features…")

    df = compute_trend_features(df)
    df = compute_volatility_features(df)
    df = compute_liquidity_features(df)
    df = compute_microstructure_features(df)
    df = compute_structure_features(df)

    logger.info("Saving raw regime features to %s", OUT_PATH)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH)

    logger.info("Done. Shape: %s", df.shape)
    return df


if __name__ == "__main__":
    compute_raw_regime_features()