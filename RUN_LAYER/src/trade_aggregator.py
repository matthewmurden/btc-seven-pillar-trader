from __future__ import annotations

"""
Aggregate raw trades CSVs under RUN_LAYER/data/raw/trades into a minute-level
parquet with columns:
  - open_time_ms
  - agg_qty
  - agg_buy_qty
  - agg_sell_qty
  - agg_delta
  - agg_trades
"""

import logging
from pathlib import Path
from typing import List

import pandas as pd
import polars as pl
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_TRADES_DIR = BASE_DIR / "data" / "raw" / "trades"
OUT_PATH = RAW_TRADES_DIR / "aggregated_trades.parquet"


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _process_csv(path: Path) -> pl.DataFrame:
    logging.info("Processing %s", path.name)
    pdf = pd.read_csv(path)
    # Expect columns: timestamp, price, qty, is_buyer_maker (plus others)
    if "timestamp" not in pdf.columns or "qty" not in pdf.columns or "is_buyer_maker" not in pdf.columns:
        logging.warning("Skipping %s (missing required columns)", path)
        return pl.DataFrame({"open_time_ms": [], "agg_qty": []})

    pdf["timestamp"] = pd.to_numeric(pdf["timestamp"], errors="coerce")
    pdf["qty"] = pd.to_numeric(pdf["qty"], errors="coerce")
    pdf = pdf.dropna(subset=["timestamp", "qty"])
    pdf["timestamp"] = pdf["timestamp"].astype("int64")
    pdf["qty"] = pdf["qty"].astype("float64")
    pdf["is_buyer_maker"] = (
        pdf["is_buyer_maker"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    pdf = pdf.dropna(subset=["is_buyer_maker"])

    pdf["open_time_ms"] = (pdf["timestamp"] // 60_000) * 60_000
    is_sell = pdf["is_buyer_maker"].astype(bool)

    pdf["agg_qty"] = pdf["qty"]
    pdf["agg_buy_qty"] = (~is_sell * pdf["qty"]).astype("float64")
    pdf["agg_sell_qty"] = (is_sell * pdf["qty"]).astype("float64")
    pdf["agg_delta"] = np.where(is_sell, -pdf["qty"], pdf["qty"])

    pl_df = pl.from_pandas(pdf[["open_time_ms", "agg_qty", "agg_buy_qty", "agg_sell_qty", "agg_delta"]])
    return (
        pl_df.group_by("open_time_ms")
        .agg(
            agg_qty=pl.col("agg_qty").sum(),
            agg_buy_qty=pl.col("agg_buy_qty").sum(),
            agg_sell_qty=pl.col("agg_sell_qty").sum(),
            agg_delta=pl.col("agg_delta").sum(),
            agg_trades=pl.len().cast(pl.Int32),
        )
        .sort("open_time_ms")
    )


def main() -> None:
    _setup_logging()
    if not RAW_TRADES_DIR.exists():
        raise FileNotFoundError(f"Trades directory not found: {RAW_TRADES_DIR}")

    frames: List[pl.DataFrame] = []
    for csv_file in sorted(RAW_TRADES_DIR.glob("*.csv")):
        frames.append(_process_csv(csv_file))

    if not frames:
        logging.warning("No trade CSVs found in %s", RAW_TRADES_DIR)
        return

    combined = (
        pl.concat(frames)
        .group_by("open_time_ms")
        .agg(
            agg_qty=pl.col("agg_qty").sum(),
            agg_buy_qty=pl.col("agg_buy_qty").sum(),
            agg_sell_qty=pl.col("agg_sell_qty").sum(),
            agg_delta=pl.col("agg_delta").sum(),
            agg_trades=pl.col("agg_trades").sum(),
        )
        .sort("open_time_ms")
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(OUT_PATH)
    logging.info("Saved aggregated trades -> %s (rows=%d)", OUT_PATH, combined.height)


if __name__ == "__main__":
    main()


def topup_to(end_ms: int, lookback_minutes: int) -> None:
    """
    Incrementally aggregate trades up to end_ms (exclusive) keeping a rolling window.
    Relies on trades CSVs already written by delta_and_micro.
    """
    _setup_logging()
    window_ms = lookback_minutes * 60_000
    target_start_ms = end_ms - window_ms

    # Load existing aggregated if present
    agg_df = pd.DataFrame()
    if OUT_PATH.exists():
        agg_df = pd.read_parquet(OUT_PATH)

    if not agg_df.empty and "open_time_ms" in agg_df.columns:
        existing_latest = int(agg_df["open_time_ms"].max())
    else:
        existing_latest = target_start_ms - 60_000

    desired_latest = end_ms - 60_000
    if existing_latest >= desired_latest:
        # Already current; just trim to window
        if not agg_df.empty:
            agg_df = agg_df[
                (agg_df["open_time_ms"] >= target_start_ms) & (agg_df["open_time_ms"] < end_ms)
            ]
            agg_df.to_parquet(OUT_PATH, index=False)
        return

    missing_start_ms = max(existing_latest + 60_000, target_start_ms)
    if missing_start_ms >= end_ms:
        return

    frames: List[pl.DataFrame] = []
    for csv_file in sorted(RAW_TRADES_DIR.glob("*.csv")):
        frames.append(_process_csv(csv_file))

    if not frames:
        logging.warning("No trade CSVs found in %s", RAW_TRADES_DIR)
        return

    combined = (
        pl.concat(frames)
        .group_by("open_time_ms")
        .agg(
            agg_qty=pl.col("agg_qty").sum(),
            agg_buy_qty=pl.col("agg_buy_qty").sum(),
            agg_sell_qty=pl.col("agg_sell_qty").sum(),
            agg_delta=pl.col("agg_delta").sum(),
            agg_trades=pl.col("agg_trades").sum(),
        )
        .sort("open_time_ms")
    )

    pdf = combined.to_pandas()
    pdf = pdf[(pdf["open_time_ms"] >= target_start_ms) & (pdf["open_time_ms"] < end_ms)]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pdf.to_parquet(OUT_PATH, index=False)
    logging.info("Top-up aggregated trades -> %s (rows=%d)", OUT_PATH, len(pdf))
