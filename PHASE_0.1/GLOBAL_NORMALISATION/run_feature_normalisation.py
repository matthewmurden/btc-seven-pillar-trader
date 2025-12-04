import logging
import platform
import time
from pathlib import Path

import pandas as pd

from normalisation_engine import NormalisationEngine, EngineConfig


ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = ROOT / "PILLAR_FOUR_REGIME_DETECTION" / "data" / "processed"
NORMALISATION_DIR = DATA_PROCESSED / "normalisation"
NORMALISATION_DIR.mkdir(parents=True, exist_ok=True)


def _load_phase0(logger):
    candidates = [
        DATA_PROCESSED / "phase0" / "canonical_raw_with_utilities.parquet",
        DATA_PROCESSED / "raw_layer" / "canonical_raw_with_utilities.parquet",
    ]
    for p in candidates:
        if p.exists():
            logger.info("Loading Phase 0 output from %s", p)
            df = pd.read_parquet(p)

            float_cols = df.select_dtypes(include=["float64"]).columns
            df[float_cols] = df[float_cols].astype("float32")
            return df

    raise FileNotFoundError("canonical_raw_with_utilities.parquet not found.")


def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("fast.norm")
    logger.info("Starting fast normalisation on %s (%s)", platform.system(), platform.machine())

    df = _load_phase0(logger)
    df = df.sort_values("open_time_ms")
    logger.info("Sorted Phase 0 data: shape=%s", df.shape)

    config = EngineConfig(
        zscore_window=300,
        mad_window=300,
        regime_window=250,
        minmax_window=200,
        progress_interval_seconds=60,
        zscore_clip=5.0,
        mad_clip=6.0,
    )

    engine = NormalisationEngine(config=config, logger=logger)
    engine.config.progress_interval_seconds = 60

    start = time.time()
    logger.info("Fitting + transforming...")
    norm = engine.fit_transform(df)
    logger.info("Done in %.2fs", time.time() - start)

    out_parquet = NORMALISATION_DIR / "normalised_features.parquet"
    out_params = NORMALISATION_DIR / "normalisation_params.json"

    norm.to_parquet(out_parquet, index=False)
    engine.save(out_params)

    logger.info("Wrote: %s", out_parquet)
    logger.info("Wrote: %s", out_params)
    logger.info("FAST normalisation complete.")


if __name__ == "__main__":
    run()
