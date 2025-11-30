"""
Compute raw regime features (Pillar 4, Part 1) with strict no-leakage.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from regime.raw_features.feature_utils import (
    build_liquidity_features,
    build_microstructure_features,
    build_structure_range_features,
    build_trend_features,
    build_volatility_features,
)


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENTRY_DATA_PATH = PROJECT_ROOT / "PILLAR_ONE_ENTRY_MODEL" / "data" / "processed" / "features_with_final_signal.parquet"
OUTPUT_PATH = PROJECT_ROOT / "PILLAR_TWO_EXIT_MODEL" / "data" / "processed" / "raw_regime_features.parquet"


def _load_entry_data() -> pd.DataFrame:
    if not ENTRY_DATA_PATH.exists():
        raise FileNotFoundError(f"Entry data not found at {ENTRY_DATA_PATH}")
    df = pd.read_parquet(ENTRY_DATA_PATH)
    if "timestamp" not in df.columns:
        raise ValueError("Input missing timestamp column.")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def compute_raw_regime_features() -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Loading entry data from %s", ENTRY_DATA_PATH)
    base = _load_entry_data()

    logger.info("Building trend features...")
    trend = build_trend_features(base)
    logger.info("Building volatility features...")
    vol = build_volatility_features(base)
    logger.info("Building liquidity features...")
    liq = build_liquidity_features(base)
    logger.info("Building microstructure features...")
    micro = build_microstructure_features(base)
    logger.info("Building structure & range features...")
    struct = build_structure_range_features(base)

    logger.info("Assembling raw regime feature set...")
    out = pd.concat(
        [
            base[["timestamp"]].reset_index(drop=True),
            trend.reset_index(drop=True),
            vol.reset_index(drop=True),
            liq.reset_index(drop=True),
            micro.reset_index(drop=True),
            struct.reset_index(drop=True),
        ],
        axis=1,
    )

    for idx, dtype in enumerate(out.dtypes):
        if dtype.kind in ("i", "u", "f"):
            arr = out.iloc[:, idx].to_numpy()
            arr = np.where(np.isinf(arr), np.nan, arr)
            arr = np.where(np.isnan(arr), 0.0, arr)
            out.iloc[:, idx] = arr

    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Saved raw regime features to %s", OUTPUT_PATH)
    return out


if __name__ == "__main__":
    compute_raw_regime_features()
