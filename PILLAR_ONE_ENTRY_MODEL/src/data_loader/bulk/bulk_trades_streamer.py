import os
import zipfile
import pandas as pd

CHUNK_SIZE = 100000

EXPECTED_COLUMNS = [
    "agg_id", "price", "qty", "first_id", "last_id", "timestamp_ms", "is_buyer_maker"
]


def _clean_chunk(chunk):
    """
    Removes header rows inside chunks and normalizes columns.
    """

    # Always make a real copy so pandas NEVER warns
    chunk = chunk.copy()

    # Remove header-like rows
    if str(chunk.iloc[0, 0]).lower() in ["agg_trade_id", "agg_id", "id"]:
        chunk = chunk.iloc[1:].reset_index(drop=True)

    # Assign proper columns
    chunk.columns = EXPECTED_COLUMNS

    # Integer timestamp (safe)
    chunk.loc[:, "timestamp"] = chunk["timestamp_ms"].astype("int64") // 1000

    # Safe numeric conversions
    chunk.loc[:, "price"] = pd.to_numeric(chunk["price"], errors="coerce")
    chunk.loc[:, "qty"] = pd.to_numeric(chunk["qty"], errors="coerce")

    # is_buyer flag (0 = buy, 1 = sell)
    chunk.loc[:, "is_buyer"] = chunk["is_buyer_maker"].astype(str).str.lower().eq("false").astype(int)

    # Drop any garbage rows
    chunk = chunk.dropna(subset=["price", "qty", "timestamp"])

    return chunk[["timestamp", "price", "qty", "is_buyer"]]


def stream_bulk_trade_chunks(folder="data/raw/bulk_trades"):
    """
    Streams all CSV/ZIP files in chunks.
    Handles files with or without headers.
    """

    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)

        if fname.endswith(".zip"):
            with zipfile.ZipFile(path) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:

                    for chunk in pd.read_csv(
                        f, header=None, chunksize=CHUNK_SIZE, dtype=str
                    ):
                        yield _clean_chunk(chunk)

        elif fname.endswith(".csv"):
            for chunk in pd.read_csv(
                path, header=None, chunksize=CHUNK_SIZE, dtype=str
            ):
                yield _clean_chunk(chunk)