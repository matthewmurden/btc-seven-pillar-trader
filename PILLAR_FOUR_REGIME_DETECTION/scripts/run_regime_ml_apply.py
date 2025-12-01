# ======================================================
# run_regime_ml_apply.py
# ======================================================

from __future__ import annotations
import logging

from regime.ml_regime_classifier.apply_regime_classifier import apply_models


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    apply_models()


if __name__ == "__main__":
    main()