import argparse
import logging
from pathlib import Path
from typing import Dict

import polars as pl


MINUTE_MS = 60_000

HTF_LABELS = ["5m", "15m", "1h", "4h", "1d"]
HTF_FILES = {
    "5m": "BTCUSDT_5m.parquet",
    "15m": "BTCUSDT_15m.parquet",
    "1h": "BTCUSDT_1h.parquet",
    "4h": "BTCUSDT_4h.parquet",
    "1d": "BTCUSDT_1d.parquet",
}


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------
def _ensure_open_time_ms(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure a column open_time_ms exists."""
    if "open_time_ms" in df.columns:
        logging.debug("open_time_ms present; rows=%d cols=%d", df.height, df.width)
        return df

    if "open_time" in df.columns:
        dtype = df.schema["open_time"]
        if dtype == pl.Utf8:
            logging.debug("Converting open_time Utf8 -> open_time_ms")
            return df.with_columns(
                (
                    pl.col("open_time")
                    .str.to_datetime(time_zone="UTC")
                    .cast(pl.Int64) // 1_000_000
                ).alias("open_time_ms")
            )
        if dtype.base_type() == pl.Datetime:
            logging.debug("Converting open_time Datetime -> open_time_ms")
            return df.with_columns((pl.col("open_time").cast(pl.Int64) // 1_000_000).alias("open_time_ms"))

    raise ValueError("Input must contain either open_time_ms or open_time.")


def _compute_atr(df: pl.DataFrame) -> pl.DataFrame:
    """Compute ATR(14) only (spec does not require ATR50/200 here)."""
    prev_close = pl.col("close").shift(1)
    tr = pl.max_horizontal(
        [
            pl.col("high") - pl.col("low"),
            (pl.col("high") - prev_close).abs(),
            (pl.col("low") - prev_close).abs(),
        ]
    ).alias("true_range")

    df = df.with_columns(tr)
    df = df.with_columns(
        pl.col("true_range").rolling_mean(window_size=14, min_periods=1).alias("atr_14")
    )
    return df


def _compute_htf_features(htf_df: pl.DataFrame, label: str) -> pl.DataFrame:
    """Compute HTF slopes and ATR slopes."""
    logging.debug("Computing HTF features for %s with rows=%d", label, htf_df.height)
    df = _ensure_open_time_ms(htf_df).sort("open_time_ms")
    df = _compute_atr(df)

    df = df.with_columns(
        [
            (pl.col("close") - pl.col("close").shift(3)).alias(f"htf_close_slope_{label}"),
            (pl.col("volume") - pl.col("volume").shift(3)).alias(f"htf_volume_slope_{label}"),
            ((pl.col("atr_14") - pl.col("atr_14").shift(3)) / 3.0).alias(f"htf_vol_slope_{label}"),
        ]
    )

    return df.select(
        [
            "open_time_ms",
            f"htf_close_slope_{label}",
            f"htf_volume_slope_{label}",
            f"htf_vol_slope_{label}",
        ]
    )


# ---------------------------------------------------------
# MAIN COMPUTE LOGIC
# ---------------------------------------------------------
def compute(df_1m: pl.DataFrame, htf_dir: Path, **kwargs) -> pl.DataFrame:
    """Merge all HTF features onto 1m timeline via backward ASOF joins."""

    # Ensure clean 1m timeline
    df_1m = (
        _ensure_open_time_ms(df_1m)
        .select(["open_time_ms"])
        .sort("open_time_ms")
    )

    result = df_1m

    # Process each HTF independently
    for label in HTF_LABELS:
        path = htf_dir / HTF_FILES[label]

        if not path.exists():
            logging.warning(f"[HTF {label}] Missing file {path}, filling zeros.")
            # Create zero features
            empty = df_1m.with_columns(
                [
                    pl.lit(0.0).alias(f"htf_close_slope_{label}"),
                    pl.lit(0.0).alias(f"htf_volume_slope_{label}"),
                    pl.lit(0.0).alias(f"htf_vol_slope_{label}"),
                ]
            )
            result = result.join_asof(
                empty, on="open_time_ms", strategy="backward"
            )
            continue

        # Safe load
        try:
            htf_df = pl.read_parquet(path)
            feats = _compute_htf_features(htf_df, label)
            logging.info(f"[HTF {label}] Loaded %d rows from %s", htf_df.height, path)
        except Exception as e:
            logging.error(f"[HTF {label}] Failed to load {path}: {e}. Filling zeros.")
            empty = df_1m.with_columns(
                [
                    pl.lit(0.0).alias(f"htf_close_slope_{label}"),
                    pl.lit(0.0).alias(f"htf_volume_slope_{label}"),
                    pl.lit(0.0).alias(f"htf_vol_slope_{label}"),
                ]
            )
            result = result.join_asof(
                empty, on="open_time_ms", strategy="backward"
            )
            continue

        # Merge using ASOF
        result = result.join_asof(
            feats.sort("open_time_ms"), on="open_time_ms", strategy="backward"
        )

    # Fill missing with zero to avoid NaNs in downstream models
    return result.fill_null(0)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def main():
    _setup_logging()
    parser = argparse.ArgumentParser(description="HTF rollup engine: compute HTF features and join onto 1m timeline.")
    parser.add_argument("input", type=Path, help="Input 1m parquet (only open_time_ms is used).")
    parser.add_argument("htf_dir", type=Path, help="Directory containing HTF parquet files.")
    parser.add_argument("output", type=Path, help="Output parquet for HTF features.")
    args = parser.parse_args()

    df_1m = pl.read_parquet(args.input)
    out = compute(df_1m, args.htf_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.output)

    logging.info(f"Wrote HTF feature parquet to: {args.output}")


if __name__ == "__main__":
    main()
