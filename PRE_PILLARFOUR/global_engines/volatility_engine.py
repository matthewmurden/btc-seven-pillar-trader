import argparse
import logging
from pathlib import Path

import polars as pl
import numpy as np

MINUTE_MS = 60_000


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def _ensure_open_time_ms(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure open_time_ms (UTC integer ms) exists."""
    if "open_time_ms" in df.columns:
        return df

    if "open_time" in df.columns:
        dtype = df.schema["open_time"]
        if dtype == pl.Utf8:
            return df.with_columns(
                (
                    pl.col("open_time")
                    .str.to_datetime(time_zone="UTC")
                    .cast(pl.Int64) // 1_000_000
                ).alias("open_time_ms")
            )
        if dtype.base_type() == pl.Datetime:
            return df.with_columns(
                (pl.col("open_time").cast(pl.Int64) // 1_000_000).alias("open_time_ms")
            )

    raise ValueError("Input must contain open_time_ms or open_time.")


# ---------------------------------------------------------
# ATR (14/50/200)
# ---------------------------------------------------------
def _compute_atr(df: pl.DataFrame) -> pl.DataFrame:
    """Compute ATR(14), ATR(50), ATR(200)."""

    prev_close = pl.col("close").shift(1)

    true_range = pl.max_horizontal(
        [
            pl.col("high") - pl.col("low"),
            (pl.col("high") - prev_close).abs(),
            (pl.col("low") - prev_close).abs(),
        ]
    ).alias("true_range")

    df = df.with_columns(true_range)

    for w in (14, 50, 200):
        df = df.with_columns(
            pl.col("true_range")
            .rolling_mean(window_size=w, min_periods=1)
            .alias(f"atr_{w}")
        )

    return df


# ---------------------------------------------------------
# Realised volatility (5m, 15m, 1h)
# ---------------------------------------------------------
def _compute_realised_vol(df: pl.DataFrame) -> pl.DataFrame:
    """Correct realised volatility windows."""

    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1))
        .log()
        .alias("lr")
    )

    for w, name in [(5, "rv_5m"), (15, "rv_15m"), (60, "rv_1h")]:
        df = df.with_columns(
            (
                pl.col("lr")
                .pow(2)
                .rolling_sum(window_size=w, min_periods=1)
            ).sqrt().alias(name)
        )

    return df.drop("lr")


# ---------------------------------------------------------
# Intraday volatility cycles
# ---------------------------------------------------------
def _compute_cycles(df: pl.DataFrame) -> pl.DataFrame:
    """Daily volatility oscillation using sine/cosine cycles."""

    minute_of_day = ((pl.col("open_time_ms") // MINUTE_MS) % 1440).cast(pl.Float64)

    return df.with_columns(
        [
            (2 * np.pi * minute_of_day / 1440).sin().alias("vol_cycle_sin"),
            (2 * np.pi * minute_of_day / 1440).cos().alias("vol_cycle_cos"),
        ]
    )


# ---------------------------------------------------------
# Compression / Expansion
# ---------------------------------------------------------
def _compute_compression_expansion(df: pl.DataFrame) -> pl.DataFrame:
    """More stable compression/expansion definitions."""

    return df.with_columns([
        # ATR14 / ATR200 → contraction vs very-low vol environments
        (pl.col("atr_14") / (pl.col("atr_200") + 1e-9)).alias("vol_compression"),

        # Expansion = |ATR14 - ATR50| / max(ATR values)
        (
            (pl.col("atr_14") - pl.col("atr_50")).abs() /
            (pl.max_horizontal([pl.col("atr_14"), pl.col("atr_50")]) + 1e-9)
        ).alias("vol_expansion"),
    ])


# ---------------------------------------------------------
# Volatility Regime (ATR % version)
# ---------------------------------------------------------
def _compute_regime(df: pl.DataFrame) -> pl.DataFrame:
    """
    Vol regime based on ATR50 percentage (ATR50 / close).
    More stable than raw ATR and produces meaningful quantiles.
    """

    df = df.with_columns(
        (pl.col("atr_50") / (pl.col("close") + 1e-9)).alias("atr50_pct")
    )

    # Use linear interpolation to avoid quantile collapse
    q33 = df.select(
        pl.col("atr50_pct").quantile(0.33, interpolation="linear")
    ).item()

    q66 = df.select(
        pl.col("atr50_pct").quantile(0.66, interpolation="linear")
    ).item()

    df = df.with_columns(
        pl.when(pl.col("atr50_pct") < q33)
        .then(0)
        .when(pl.col("atr50_pct") < q66)
        .then(1)
        .otherwise(2)
        .alias("vol_regime")
    )

    return df


# ---------------------------------------------------------
# Main Compute
# ---------------------------------------------------------
def compute(df: pl.DataFrame, **kwargs) -> pl.DataFrame:
    df = _ensure_open_time_ms(df).sort("open_time_ms")

    df = _compute_atr(df)
    df = _compute_realised_vol(df)
    df = _compute_cycles(df)
    df = _compute_compression_expansion(df)
    df = _compute_regime(df)

    out_cols = [
        "open_time_ms",
        "atr_14",
        "atr_50",
        "atr_200",
        "rv_5m",
        "rv_15m",
        "rv_1h",
        "vol_cycle_sin",
        "vol_cycle_cos",
        "vol_compression",
        "vol_expansion",
        "vol_regime",
    ]

    return df.select(out_cols).fill_null(0)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def main():
    _setup_logging()
    parser = argparse.ArgumentParser(description="Volatility engine: compute volatility features from 1m OHLCV parquet.")
    parser.add_argument("input", type=Path, help="Input 1m OHLCV parquet.")
    parser.add_argument("output", type=Path, help="Output parquet with volatility features.")
    args = parser.parse_args()

    df = pl.read_parquet(args.input)
    out = compute(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.output)

    logging.info(f"Wrote volatility features to {args.output}")


if __name__ == "__main__":
    main()