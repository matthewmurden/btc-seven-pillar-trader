import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import polars as pl


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT.parent / "PILLAR_FOUR_REGIME_DETECTION" / "data" / "raw"
PROCESSED_DIR = ROOT / "processed"

CANDLES_FILE = RAW_DIR / "candles" / "BTCUSDT_1m_2022-11-01_2025-10-31.csv"
DELTA_FILE = RAW_DIR / "delta" / "delta_stream.parquet"
MICRO_FILE = RAW_DIR / "microstructure" / "microstructure_metrics.parquet"
BULK_DIR = ROOT / "processed" / "bulk_parquet"

HTF_FILES = {
    "5m": RAW_DIR / "htf_candles" / "BTCUSDT_5m.parquet",
    "15m": RAW_DIR / "htf_candles" / "BTCUSDT_15m.parquet",
    "1h": RAW_DIR / "htf_candles" / "BTCUSDT_1h.parquet",
    "4h": RAW_DIR / "htf_candles" / "BTCUSDT_4h.parquet",
    "1d": RAW_DIR / "htf_candles" / "BTCUSDT_1d.parquet",
}


def _setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")


def safe_read_parquet(path: Path, retries: int = 10, delay: float = 0.2) -> pl.DataFrame:
    last_err = None
    for i in range(1, retries + 1):
        try:
            return pl.read_parquet(path)
        except PermissionError as exc:
            last_err = exc
            logging.warning(f"[{path.name}] PermissionError on attempt {i}/{retries}; retrying in {delay}s")
            time.sleep(delay)
        except Exception as exc:
            last_err = exc
            logging.warning(f"[{path.name}] Read attempt {i}/{retries} failed: {exc}; retrying in {delay}s")
            time.sleep(delay)
    raise PermissionError(f"Could not read {path} after {retries} retries") from last_err


def _to_float32(df: pd.DataFrame, cols: Iterable[str]):
    ex = [c for c in cols if c in df.columns]
    if ex:
        df[ex] = df[ex].astype("float32")
    return df


# --------------------------------------------------------------------
# LOAD CANDLES (PANDAS)
# --------------------------------------------------------------------
def load_candles(path: Path) -> pd.DataFrame:
    logging.info(f"Loading candles from {path}")
    df = pd.read_csv(path)

    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df["open_time_ms"] = df["open_time"].astype("int64") // 1_000_000

    df = df.drop_duplicates("open_time_ms").sort_values("open_time_ms")

    float_cols = ["open", "high", "low", "close", "volume",
                  "quote_volume", "taker_buy_base", "taker_buy_quote"]
    df = _to_float32(df, float_cols)

    df["trades"] = df["trades"].astype("int32")
    return df


# --------------------------------------------------------------------
# LOAD DELTA (PANDAS)
# --------------------------------------------------------------------
def load_delta(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["delta"] = df["delta"].astype("float32")
    df["open_time_ms"] = df["open_time_ms"].astype("int64")
    return df


# --------------------------------------------------------------------
# LOAD MICROSTRUCTURE (PANDAS)
# --------------------------------------------------------------------
def load_microstructure(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["open_time_ms"] = df["open_time_ms"].astype("int64")
    float_cols = [c for c in df.columns if c != "open_time_ms"]
    df = _to_float32(df, float_cols)
    df = df.drop_duplicates("open_time_ms")
    return df


# --------------------------------------------------------------------
# LOAD BULK TRADES (POLARS → PANDAS)
# --------------------------------------------------------------------
def _process_bulk_file(path: Path) -> pl.DataFrame:
    logging.info(f"Processing {path.name}")

    df = safe_read_parquet(path)

    # Rename whatever columns exist (since headers may have shifted)
    cols = df.columns
    mapping = {
        cols[0]: "agg_trade_id",
        cols[1]: "price",
        cols[2]: "qty",
        cols[3]: "first_trade_id",
        cols[4]: "last_trade_id",
        cols[5]: "timestamp_ms",
        cols[6]: "is_buyer_maker",
    }
    df = df.rename(mapping)

    df = df.with_columns([
        pl.col("timestamp_ms").cast(pl.Int64),
        pl.col("qty").cast(pl.Float64),
        pl.col("is_buyer_maker").cast(pl.Utf8).str.to_lowercase()
    ]).drop_nulls(subset=["timestamp_ms"])

    df = df.with_columns(
        (pl.col("timestamp_ms") // 60000 * 60000).alias("open_time_ms")
    )

    is_sell = pl.col("is_buyer_maker").is_in(["true", "1"])

    df = df.with_columns([
        pl.when(is_sell).then(-pl.col("qty")).otherwise(pl.col("qty")).alias("agg_delta"),
        pl.when(~is_sell).then(pl.col("qty")).otherwise(0.0).alias("agg_buy_qty"),
        pl.when(is_sell).then(pl.col("qty")).otherwise(0.0).alias("agg_sell_qty"),
    ])

    return (
        df.group_by("open_time_ms")
        .agg(
            agg_qty=pl.col("qty").sum(),
            agg_buy_qty=pl.col("agg_buy_qty").sum(),
            agg_sell_qty=pl.col("agg_sell_qty").sum(),
            agg_delta=pl.col("agg_delta").sum(),
            agg_trades=pl.len().cast(pl.Int32),
        )
        .sort("open_time_ms")
    )


def load_bulk_trades(folder: Path) -> pd.DataFrame:
    logging.info(f"Aggregating bulk from {folder}")
    frames = []

    for file in sorted(folder.glob("*.parquet")):
        try:
            frames.append(_process_bulk_file(file))
        except Exception as e:
            logging.error(f"Failed reading {file}: {e}")
            continue

    if not frames:
        return pd.DataFrame({"open_time_ms": [], "agg_qty": []})

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
    pdf = _to_float32(pdf, ["agg_qty", "agg_buy_qty",
                            "agg_sell_qty", "agg_delta"])
    return pdf


# --------------------------------------------------------------------
# LOAD HTF (PANDAS)
# --------------------------------------------------------------------
def load_htf(files: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
    out = {}
    for label, path in files.items():
        logging.info(f"Loading HTF {label}")

        df = pd.read_parquet(path)
        if "open_time_ms" not in df.columns:
            df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
            df["open_time_ms"] = df["open_time"].astype("int64") // 1_000_000

        df = df.sort_values("open_time_ms").reset_index(drop=True)

        rename = {c: f"{c}_{label}" for c in df.columns if c != "open_time_ms"}
        df = df.rename(columns=rename)
        out[label] = df

    return out


# --------------------------------------------------------------------
# MERGE EVERYTHING (ALL PANDAS)
# --------------------------------------------------------------------
def merge_all(candles: pd.DataFrame,
              delta: pd.DataFrame,
              micro: pd.DataFrame,
              bulk: pd.DataFrame,
              htf_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:

    df = candles.copy()

    # Basic merges
    for name, other in [("delta", delta), ("micro", micro), ("bulk", bulk)]:
        if not other.empty:
            df = df.merge(other, on="open_time_ms", how="left")

    df = df.sort_values("open_time_ms")

    # ASOF merges for HTF
    for label, htf_df in htf_map.items():
        df = pd.merge_asof(
            df,
            htf_df.sort_values("open_time_ms"),
            on="open_time_ms",
            direction="backward"
        )

    return df.reset_index(drop=True)


# --------------------------------------------------------------------
# SAVE OUTPUT
# --------------------------------------------------------------------
def save_outputs(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "canonical_raw_with_utilities.parquet"
    df.to_parquet(out_path, index=False)
    logging.info(f"Saved to {out_path}")


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
def main():
    args = parse_args()
    _setup_logging()

    t0 = time.time()
    logging.info("Starting Phase 0 canonical build")

    candles = load_candles(CANDLES_FILE)
    delta = load_delta(DELTA_FILE)
    micro = load_microstructure(MICRO_FILE)
    bulk = load_bulk_trades(BULK_DIR)
    htf_map = load_htf(HTF_FILES)

    canonical = merge_all(candles, delta, micro, bulk, htf_map)
    save_outputs(canonical, args.output_dir)

    logging.info(f"Completed Phase 0 in {time.time() - t0:.2f}s")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    return p.parse_args()


if __name__ == "__main__":
    main()
