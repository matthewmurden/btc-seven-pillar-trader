"""
Applies ML Regime Classifier to full dataset (inference)
Uses saved models + feature list
Outputs: data/processed/regime_ml_predictions.parquet
"""

from __future__ import annotations
import logging
from regime.ml_regime_classifier.apply_regime_classifier import apply_regime_classifier

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    apply_regime_classifier()

if __name__ == "__main__":
    main()