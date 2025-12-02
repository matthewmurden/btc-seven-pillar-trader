import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
BULK_TRADES_DIR = DATA_DIR / "raw" / "bulk_trades"
OUTPUT_PATH = DATA_DIR / "raw" / "delta" / "delta_stream.parquet"

CHUNK_SIZE = 1_000_000   # trades per chunk
EPS = 1e-12


# ============================================================
# LOGGING
# ============================================================
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


# ============================================================
# STREAM TRADES (ZIPPED CSV)
# ============================================================
def stream_trades() -> Iterable[pd.DataFrame]:
    """
    Streams all aggTrades ZIP files one batch at a time.

    Expected columns in Binance aggTrades CSV:
    - price
    - quantity
    - transact_time
    - is_buyer_maker
    """
    files = sorted(BULK_TRADES_DIR.glob("*.zip"))
    if not files:
        raise FileNotFoundError(f"No ZIP files found in {BULK_TRADES_DIR}")

    logging.info("Found %d aggTrades ZIP files", len(files))

    for file in files:
        logging.info("Reading %s", file.name)

        reader = pd.read_csv(
            file,
            compression="zip",
            usecols=["price", "quantity", "transact_time", "is_buyer_maker"],
            dtype={
                "price": "float64",
                "quantity": "float64",
                "transact_time": "int64"
            },
            chunksize=CHUNK_SIZE
        )

        for i, chunk in enumerate(reader, start=1):
            # Standardise names
            chunk = chunk.rename(columns={
                "quantity": "qty",
                "transact_time": "timestamp"
            })

            # Convert is_buyer_maker to bool
            chunk["is_buyer_maker"] = (
                chunk["is_buyer_maker"]
                .astype(str)
                .str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
            )

            chunk = chunk.dropna(subset=["timestamp", "qty", "is_buyer_maker"])

            # dtype enforcement
            chunk["is_buyer_maker"] = chunk["is_buyer_maker"].astype(bool)

            logging.info(
                "%s | chunk %d: %s rows",
                file.name, i, f"{len(chunk):,}"
            )
            yield chunk


# ============================================================
# AGGREGATE DELTA PER MINUTE
# ============================================================
def aggregate_delta(chunks: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """
    Builds the 1-minute delta stream.

    Produces a continuous minute index with:
    - open_time_ms
    - delta
    """
    agg = None

    for chunk in chunks:
        # Compute signed delta per trade
        chunk["delta_signed"] = np.where(
            ~chunk["is_buyer_maker"], chunk["qty"], -chunk["qty"]
        )

        # Compute minute bucket
        chunk["minute"] = (chunk["timestamp"] // 60_000) * 60_000

        # Aggregate for this chunk
        chunk_agg = (
            chunk.groupby("minute")["delta_signed"]
            .sum()
            .astype("float64")
            .to_frame("delta")
        )

        if agg is None:
            agg = chunk_agg
        else:
            # Align indexes and sum
            common_index = agg.index.union(chunk_agg.index)
            agg = agg.reindex(common_index, fill_value=0)
            chunk_agg = chunk_agg.reindex(common_index, fill_value=0)
            agg["delta"] = agg["delta"] + chunk_agg["delta"]

    if agg is None:
        raise ValueError("No trade data found in ZIP files")

    # Ensure continuous timeline
    agg = agg.sort_index()
    full_idx = np.arange(
        agg.index.min(),
        agg.index.max() + 60_000,
        60_000,
        dtype="int64"
    )

    agg = agg.reindex(full_idx, fill_value=0)
    agg.index.name = "open_time_ms"

    logging.info("Final delta minute count: %s", f"{len(agg):,}")

    return agg.reset_index()


# ============================================================
# SAVE
# ============================================================
def save_delta(df: pd.DataFrame):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Final schema = ONLY TWO COLUMNS
    df = df[["open_time_ms", "delta"]]

    logging.info("Saving delta stream → %s", OUTPUT_PATH)
    df.to_parquet(OUTPUT_PATH, index=False)
    logging.info("DONE — saved %s rows", f"{len(df):,}")


# ============================================================
# MAIN
# ============================================================
def main():
    setup_logging()
    logging.info("=== DELTA GENERATOR START ===")

    try:
        chunks = stream_trades()
        agg = aggregate_delta(chunks)
        save_delta(agg)
        logging.info("=== DELTA GENERATOR FINISHED ===")

    except Exception as exc:
        logging.exception("Delta pipeline failure: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()