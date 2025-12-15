from __future__ import annotations

"""
Live helper to build canonical_with_all_engines.parquet using Phase 0 outputs
from RUN_LAYER/data/processed/PRE_PILLARFOUR/canonical_raw_with_utilities.parquet
and HTF/parquet inputs under RUN_LAYER/data/raw.
"""

import logging
import sys
from pathlib import Path

import polars as pl

# Paths
THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parents[2]
RUN_RAW = BASE_DIR / "RUN_LAYER" / "data" / "raw"
OUT_DIR = BASE_DIR / "RUN_LAYER" / "data" / "processed" / "PRE_PILLARFOUR"
CANONICAL_INPUT = OUT_DIR / "canonical_raw_with_utilities.parquet"

PREFour_BASE = BASE_DIR / "PRE_PILLARFOUR"
if str(PREFour_BASE) not in sys.path:
    sys.path.append(str(PREFour_BASE))

from global_engines.volatility_engine import compute as vol_compute  # type: ignore
from global_engines.session_engine import compute as sess_compute  # type: ignore
from global_engines.htf_rollup_engine import compute as htf_compute  # type: ignore

HTF_DIR = RUN_RAW / "htf_candles"
OUTPUT_PATH = OUT_DIR / "canonical_with_all_engines.parquet"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not CANONICAL_INPUT.exists():
        raise FileNotFoundError(f"Canonical input missing: {CANONICAL_INPUT}")

    logging.info("Loading canonical base from %s", CANONICAL_INPUT)
    df = pl.read_parquet(CANONICAL_INPUT)

    # Ensure open_time_ms is present
    if "open_time_ms" not in df.columns:
        if "open_time" in df.columns:
            df = df.with_columns(
                (pl.col("open_time").str.to_datetime(time_zone="UTC").cast(pl.Int64) // 1_000_000).alias("open_time_ms")
            )
        else:
            raise ValueError("Missing open_time_ms/open_time in canonical input.")

    logging.info("Running volatility engine...")
    vol = vol_compute(df)
    logging.info("Running session engine...")
    sess = sess_compute(df)
    logging.info("Running HTF rollup engine from %s...", HTF_DIR)
    htf = htf_compute(df, HTF_DIR)

    df = df.join(vol, on="open_time_ms", how="left", suffix="_dup")
    df = df.join(sess, on="open_time_ms", how="left", suffix="_dup")
    df = df.join(htf, on="open_time_ms", how="left", suffix="_dup")

    dup_cols = [c for c in df.columns if c.endswith("_dup")]
    if dup_cols:
        df = df.drop(dup_cols)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUTPUT_PATH)
    logging.info("Saved canonical_with_all_engines -> %s (rows=%d cols=%d)", OUTPUT_PATH, df.height, df.width)


if __name__ == "__main__":
    main()
