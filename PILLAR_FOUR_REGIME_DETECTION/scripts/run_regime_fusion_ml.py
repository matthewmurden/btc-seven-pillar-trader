"""
Run Part 4B — ML + deterministic fusion.
"""

from __future__ import annotations

import logging

from regime.fusion.fusion_layer import run_fusion


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = run_fusion()

    logging.info("Saved fused labels with %s rows", len(df))
    logging.info("Unique regime labels: %s", sorted(df["regime_label"].unique().tolist()))
    logging.info(
        "Confidence stats: mean=%.4f std=%.4f min=%.4f p25=%.4f p50=%.4f p75=%.4f max=%.4f",
        df["ml_confidence"].mean(),
        df["ml_confidence"].std(),
        df["ml_confidence"].min(),
        df["ml_confidence"].quantile(0.25),
        df["ml_confidence"].quantile(0.50),
        df["ml_confidence"].quantile(0.75),
        df["ml_confidence"].max(),
    )


if __name__ == "__main__":
    main()
