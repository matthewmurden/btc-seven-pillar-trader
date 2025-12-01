# ======================================================
# run_regime_raw_features.py
# ======================================================

from __future__ import annotations
import logging

from regime.raw_features.compute_raw_regime_features import compute_raw_regime_features


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    compute_raw_regime_features()


if __name__ == "__main__":
    main()