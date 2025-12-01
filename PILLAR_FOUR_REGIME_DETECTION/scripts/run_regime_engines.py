# ======================================================
# run_regime_engines.py
# ======================================================

from __future__ import annotations
import logging
import pandas as pd
import numpy as np

from regime.deterministic_engines import run_all_engines


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Demo input to validate wiring; replace with real pipeline data as needed.
    df = pd.DataFrame(
        {
            "trend_score": np.linspace(-0.2, 0.8, 10),
            "volatility_score": np.linspace(-0.5, 0.7, 10),
            "structure_score": np.linspace(-0.1, 0.5, 10),
            "noise_to_signal": np.linspace(0.1, 1.0, 10),
            "chop_index": np.linspace(0.0, 1.0, 10),
            "atr_expansion": np.linspace(0.0, 0.8, 10),
            "bb_width": np.linspace(0.05, 0.5, 10),
            "volume": np.linspace(1000, 2000, 10),
            "orderflow_score": np.linspace(-1, 1, 10),
            "delta_intensity": np.linspace(0.0, 1.0, 10),
            "reversion_score": np.linspace(0.0, 1.0, 10),
            "mean_reversion_distance": np.linspace(0.0, 2.0, 10),
            "sweep_signal": np.where(np.linspace(0, 1, 10) > 0.5, 1, 0),
        }
    )

    result = run_all_engines(df)
    logging.info("Deterministic engines output:\n%s", result.head())


if __name__ == "__main__":
    main()
