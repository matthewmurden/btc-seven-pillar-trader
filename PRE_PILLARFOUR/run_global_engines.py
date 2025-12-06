import argparse
import logging
from pathlib import Path

import polars as pl

from global_engines.volatility_engine import compute as vol_compute
from global_engines.session_engine import compute as session_compute
from global_engines.htf_rollup_engine import compute as htf_compute


ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "processed"
DEFAULT_CANON = PROCESSED_DIR / "canonical_raw_with_utilities.parquet"
DEFAULT_HFT_DIR = ROOT.parent / "PILLAR_FOUR_REGIME_DETECTION" / "data" / "raw" / "htf_candles"

VOL_OUT = PROCESSED_DIR / "volatility_features.parquet"
SESSION_OUT = PROCESSED_DIR / "session_features.parquet"
HTF_OUT = PROCESSED_DIR / "htf_features.parquet"


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def run(can_path: Path, htf_dir: Path) -> None:
    logging.info("Loading canonical from %s", can_path)
    df = pl.read_parquet(can_path)
    logging.info("Computing volatility features...")
    vol_df = vol_compute(df)
    VOL_OUT.parent.mkdir(parents=True, exist_ok=True)
    vol_df.write_parquet(VOL_OUT)
    logging.info("Saved volatility features -> %s", VOL_OUT)

    logging.info("Computing session features...")
    sess_df = session_compute(df)
    sess_df.write_parquet(SESSION_OUT)
    logging.info("Saved session features -> %s", SESSION_OUT)

    logging.info("Computing HTF rollup features from %s", htf_dir)
    htf_df = htf_compute(df, htf_dir)
    htf_df.write_parquet(HTF_OUT)
    logging.info("Saved HTF features -> %s", HTF_OUT)


def main():
    _setup_logging()
    parser = argparse.ArgumentParser(description="Run global utility engines to produce feature parquets.")
    parser.add_argument("--canon", type=Path, default=DEFAULT_CANON, help="Canonical 1m parquet path.")
    parser.add_argument("--htf-dir", type=Path, default=DEFAULT_HFT_DIR, help="Directory containing HTF parquets.")
    args = parser.parse_args()
    run(args.canon, args.htf_dir)


if __name__ == "__main__":
    main()
