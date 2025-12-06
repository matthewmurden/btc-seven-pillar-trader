import argparse
import logging
from pathlib import Path

import polars as pl

MINUTE_MS = 60_000


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _ensure_open_time_ms(df: pl.DataFrame) -> pl.DataFrame:
    if "open_time_ms" in df.columns:
        return df

    if "open_time" in df.columns:
        dtype = df.schema["open_time"]
        if dtype == pl.Utf8:
            return df.with_columns(
                (pl.col("open_time").str.to_datetime(time_zone="UTC").cast(pl.Int64) // 1_000_000)
                .alias("open_time_ms")
            )
        if dtype.base_type() == pl.Datetime:
            return df.with_columns(
                (pl.col("open_time").cast(pl.Int64) // 1_000_000).alias("open_time_ms")
            )

    raise ValueError("Input must contain open_time_ms or open_time.")


def _compute_rv_15m(df: pl.DataFrame) -> pl.DataFrame:
    if "rv_15m" in df.columns:
        return df

    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1)).log().alias("lr")
    )

    # Correct formula (matching test suite)
    df = df.with_columns(
        (
            pl.col("lr").pow(2).rolling_sum(window_size=15, min_periods=1)
        ).sqrt().alias("rv_15m")
    )

    return df.drop("lr")


def compute(df: pl.DataFrame, **kwargs) -> pl.DataFrame:
    df = _ensure_open_time_ms(df).sort("open_time_ms")
    df = _compute_rv_15m(df)

    dt = pl.from_epoch((pl.col("open_time_ms") // 1000), time_unit="s").dt.replace_time_zone("UTC")
    hour = dt.dt.hour()

    df = df.with_columns([
        ((hour >= 0) & (hour < 9)).alias("session_asia"),
        ((hour >= 7) & (hour < 16)).alias("session_london"),
        ((hour >= 12) & (hour < 21)).alias("session_ny"),
    ])

    df = df.with_columns(
        ((pl.col("close") - pl.col("close").shift(30))
         / (pl.col("close").shift(30) + 1e-12)).alias("session_vol_bias")
    )

    df = df.with_columns(
        (
            pl.col("volume")
            / (pl.col("volume").rolling_mean(window_size=60, min_periods=1) + 1e-12)
        ).alias("session_liquidity_compression")
    )

    df = df.with_columns(
        ((pl.col("rv_15m") - pl.col("rv_15m").shift(3)) / 3.0).alias("session_vol_slope")
    )

    out = df.fill_null(0).select([
        "open_time_ms",
        "session_asia",
        "session_london",
        "session_ny",
        "session_vol_bias",
        "session_liquidity_compression",
        "session_vol_slope",
    ])

    return out


def main():
    _setup_logging()
    parser = argparse.ArgumentParser(description="Session engine")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    df = pl.read_parquet(args.input)
    out = compute(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.output)

    logging.info(f"Session features written to {args.output}")


if __name__ == "__main__":
    main()