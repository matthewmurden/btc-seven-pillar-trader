import logging
import time

import pandas as pd

from PHASE_0.PRE_PROCESS.global_engines.htf_rollup_engine import HTFRollupEngine
from PHASE_0.PRE_PROCESS.global_engines.session_engine import SessionEngine
from PHASE_0.PRE_PROCESS.global_engines.volatility_engine import VolatilityEngine
from PHASE_0.PRE_PROCESS.raw_data_layer import PROCESSED_DIR, load_raw_data


def _log_missing(df: pd.DataFrame, logger: logging.Logger) -> None:
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        logger.info("No missing values detected")
    else:
        logger.info("Missing values (top 15):\n%s", missing.head(15))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("phase0.runner")
    start = time.time()

    canonical = load_raw_data(logger)
    logger.info("Canonical raw shape %s", canonical.shape)

    df = canonical.copy()
    df = VolatilityEngine(logger).run(df)
    df = SessionEngine(logger).run(df)
    df = HTFRollupEngine(logger=logger).run(df)

    if len(df) != len(canonical):
        raise ValueError(f"Length mismatch after utilities: {len(df)} vs base {len(canonical)}")

    df = df.sort_values("open_time_ms").reset_index(drop=True)
    _log_missing(df, logger)

    out_path = PROCESSED_DIR / "canonical_raw_with_utilities.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow")

    elapsed = time.time() - start
    logger.info("Saved %s", out_path)
    logger.info("Phase 0 completed in %.2fs", elapsed)


if __name__ == "__main__":
    main()
