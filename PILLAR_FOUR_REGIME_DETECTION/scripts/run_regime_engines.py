"""
Runs Part 2 — Deterministic Regime Engines
Requires: raw_regime_features.parquet
Outputs: deterministic_regime_engines.parquet
"""

from __future__ import annotations
import logging
from regime.deterministic_engines.engine_utils import run_all_engines

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_all_engines()

if __name__ == "__main__":
    main()