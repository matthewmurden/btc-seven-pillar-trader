"""
Runs Part 3 — ML Regime Classifier Training
Uses: raw + deterministic + Pillar1 signals
Saves: LGBM/XGB models + feature list
"""

from __future__ import annotations
import logging
from regime.ml_regime_classifier.train_regime_classifier import train_and_save

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train_and_save()

if __name__ == "__main__":
    main()