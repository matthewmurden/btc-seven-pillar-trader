"""
Run deterministic regime engines on raw regime features.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from regime.deterministic_engines.chop_engine import apply_chop_engine
from regime.deterministic_engines.liquidity_engine import apply_liquidity_engine
from regime.deterministic_engines.mean_reversion_engine import apply_mean_reversion_engine
from regime.deterministic_engines.trend_engine import apply_trend_engine
from regime.deterministic_engines.volatility_engine import apply_volatility_engine


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "data" / "processed" / "raw_regime_features.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "deterministic_regime_engines.parquet"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing raw regime features: {RAW_PATH}")

    logger.info("Loading raw regime features from %s", RAW_PATH)
    df = pd.read_parquet(RAW_PATH)

    logger.info("Applying trend engine")
    trend = apply_trend_engine(df)
    logger.info("Applying chop engine")
    chop = apply_chop_engine(df)
    logger.info("Applying volatility engine")
    vol = apply_volatility_engine(df)
    logger.info("Applying liquidity engine")
    liq = apply_liquidity_engine(df)
    logger.info("Applying mean reversion engine")
    mr = apply_mean_reversion_engine(df)

    result = pd.concat([df.reset_index(drop=True), trend, chop, vol, liq, mr], axis=1)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)

    logger.info("Saved deterministic regime engines to %s", OUTPUT_PATH)
    logger.info("Head:\n%s", result.head())
    logger.info("Summary:\n%s", result.describe())
    logger.info("NaN counts:\n%s", result.isna().sum())
    logger.info("Class counts trend_flag:\n%s", result.get("trend_flag", pd.Series()).value_counts(dropna=False))
    logger.info("Class counts chop_flag:\n%s", result.get("chop_flag", pd.Series()).value_counts(dropna=False))
    logger.info("Class counts volatility_regime:\n%s", result.get("volatility_regime", pd.Series()).value_counts(dropna=False))
    logger.info("Class counts liquidity_regime:\n%s", result.get("liquidity_regime", pd.Series()).value_counts(dropna=False))
    logger.info("Class counts mean_reversion_flag:\n%s", result.get("mean_reversion_flag", pd.Series()).value_counts(dropna=False))


if __name__ == "__main__":
    main()
