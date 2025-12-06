import argparse
import logging
from pathlib import Path

import polars as pl

from global_engines.volatility_engine import compute as vol_compute
from global_engines.session_engine import compute as sess_compute
from global_engines.htf_rollup_engine import compute as htf_compute


DEFAULT_INPUT = Path("processed/canonical_raw_with_utilities.parquet")
DEFAULT_OUTPUT = Path("processed/canonical_with_all_engines.parquet")
DEFAULT_HTF_DIR = Path(__file__).resolve().parent.parent / "PILLAR_FOUR_REGIME_DETECTION" / "data" / "raw" / "htf_candles"


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def ensure_open_time_ms(df: pl.DataFrame) -> pl.DataFrame:
    if "open_time_ms" in df.columns:
        ts = df["open_time_ms"].max()
        if ts is not None and ts > 1_000_000_000_000_000:
            df = df.with_columns((pl.col("open_time_ms") // 1_000_000).alias("open_time_ms"))
        return df
    if "open_time" in df.columns:
        df = df.with_columns(
            (pl.col("open_time").str.to_datetime(time_zone="UTC").cast(pl.Int64) // 1_000_000).alias("open_time_ms")
        )
        return df
    raise ValueError("Missing open_time_ms/open_time in input.")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Build canonical with all global engines merged.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--htf-dir", type=Path, default=DEFAULT_HTF_DIR)
    args = parser.parse_args()

    logging.info("Loading canonical from %s", args.input)
    df = pl.read_parquet(args.input)
    df = ensure_open_time_ms(df)
    logging.info("Input shape: rows=%d cols=%d", df.height, df.width)

    logging.info("Running volatility engine...")
    vol = vol_compute(df)
    logging.info("Volatility shape: rows=%d cols=%d", vol.height, vol.width)

    logging.info("Running session engine...")
    sess = sess_compute(df)
    logging.info("Session shape: rows=%d cols=%d", sess.height, sess.width)

    logging.info("Running HTF rollup engine from %s...", args.htf_dir)
    if not args.htf_dir.exists():
        raise FileNotFoundError(f"HTF directory not found: {args.htf_dir}. Pass --htf-dir to override.")
    htf = htf_compute(df, args.htf_dir)
    logging.info("HTF shape: rows=%d cols=%d", htf.height, htf.width)

    merge_cols = ["open_time_ms"]
    df = df.join(vol, on="open_time_ms", how="left", suffix="_dup")
    df = df.join(sess, on="open_time_ms", how="left", suffix="_dup")
    df = df.join(htf, on="open_time_ms", how="left", suffix="_dup")

    dup_cols = [c for c in df.columns if c.endswith("_dup")]
    if dup_cols:
        df = df.drop(dup_cols)

    if "open_time_ms" not in df.columns:
        raise RuntimeError("open_time_ms missing after merge.")

    logging.info("Final merged shape: rows=%d cols=%d", df.height, df.width)
    df.write_parquet(args.output)
    logging.info("Wrote merged canonical with engines to %s", args.output)

    if df.height != vol.height or df.height != sess.height or df.height != htf.height:
        logging.warning("Row count mismatch after merge: base=%d vol=%d sess=%d htf=%d",
                        df.height, vol.height, sess.height, htf.height)


if __name__ == "__main__":
    main()
