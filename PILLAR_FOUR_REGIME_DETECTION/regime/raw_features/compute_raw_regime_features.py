"""
Pillar 4 — RAW REGIME FEATURE BUILDER
=====================================

This module constructs ALL raw regime features used by:
- deterministic engines
- ML classifier
- regime fusion layer
- stability/smoothing pipeline

It is the SINGLE source of truth for raw features.
No leakage. No forward-looking windows. All features computable in real time.
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
    compute_range_features,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_OUTPUT = PROJECT_ROOT / "PILLAR_FOUR_REGIME_DETECTION" / "data" / "processed" / "raw_regime_features.parquet"


# ---------------------------------------------------------
# LOAD BASE OHLCV
# ---------------------------------------------------------

def _load_ohlcv() -> pd.DataFrame:
    """
    Replace this function if your OHLCV source changes.

    Requirements:
        - columns: ["timestamp", "open", "high", "low", "close", "volume"]
        - indexed in chronological order
    """

    possible_paths = [
        PROJECT_ROOT / "data" / "processed" / "ohlcv.parquet",
        PROJECT_ROOT / "PILLAR_ONE_ENTRY_MODEL" / "data" / "processed" / "ohlcv.parquet",
    ]

    for p in possible_paths:
        if p.exists():
            df = pd.read_parquet(p)
            logger.info(f"Loaded OHLCV from {p}")
            return df

    raise FileNotFoundError("Could not find OHLCV parquet in expected locations.")


# ---------------------------------------------------------
# MAIN FEATURE BUILDER
# ---------------------------------------------------------

def compute_raw_regime_features() -> pd.DataFrame:
    """
    Constructs the full raw regime feature matrix.

    Output columns include:
    - Trend features
    - Volatility structure features
    - Liquidity & orderflow proxies
    - Microstructure features
    - Range/contextual features
    """

    df = _load_ohlcv()

    # Ensure chronological order
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Validate structure
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV missing required columns: {missing}")

    logger.info("Computing trend features...")
    df = compute_trend_features(df)

    logger.info("Computing volatility features...")
    df = compute_volatility_features(df)

    logger.info("Computing liquidity features...")
    df = compute_liquidity_features(df)

    logger.info("Computing microstructure features...")
    df = compute_microstructure_features(df)

    logger.info("Computing range/context features...")
    df = compute_range_features(df)

    # Fill any NaNs caused by rolling windows
    df = df.fillna(0.0)

    # Ensure timestamp is the first column
    cols = ["timestamp"] + [c for c in df.columns if c != "timestamp"]
    df = df[cols]

    logger.info(f"Final raw regime feature shape: {df.shape}")
    df.to_parquet(RAW_OUTPUT, index=False)
    logger.info(f"Saved raw regime features → {RAW_OUTPUT}")

    return df


# ---------------------------------------------------------
# CLI ENTRYPOINT
# ---------------------------------------------------------

if __name__ == "__main__":
    compute_raw_regime_features()