"""
CLI to compute raw regime features from Pillar One entry data.
"""

from __future__ import annotations

import logging

from regime.raw_features.compute_raw_regime_features import compute_raw_regime_features


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = compute_raw_regime_features()
    logging.info("Head:\n%s", df.head())
    logging.info("Summary:\n%s", df.describe())
    logging.info("NaN counts:\n%s", df.isna().sum())


if __name__ == "__main__":
    main()
