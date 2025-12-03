import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

# Base paths (adjust if needed)
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "raw_layer"
CHUNK_SIZE = 250_000  # for aggTrades streaming (keep memory low)

OHLCV_PATH = RAW_DIR / "candles" / "BTCUSDT_1m_2022-11-01_2025-10-31.csv"
DELTA_PATH = RAW_DIR / "delta" / "delta_stream.parquet"
MICRO_PATH = RAW_DIR / "microstructure" / "microstructure_metrics.parquet"
AGG_TRADES_DIR = RAW_DIR / "bulk_trades"
HTF_DIR = RAW_DIR / "htf_candles"
HTF_FILES = {
    "5m": HTF_DIR / "BTCUSDT_5m.parquet",
    "15m": HTF_DIR / "BTCUSDT_15m.parquet",
    "1h": HTF_DIR / "BTCUSDT_1h.parquet",
    "4h": HTF_DIR / "BTCUSDT_4h.parquet",
    "1d": HTF_DIR / "BTCUSDT_1d.parquet",
}


def _ensure_open_time_ms(df: pd.DataFrame, time_col: str = "open_time") -> pd.DataFrame:
    if "open_time_ms" in df.columns:
        return df
    if time_col not in df.columns:
        raise ValueError("Missing open_time_ms and open_time columns.")
    df = df.copy()
    df["open_time_ms"] = pd.to_datetime(df[time_col]).astype("int64") // 1_000_000
    return df


def load_ohlcv(path: Path) -> pd.DataFrame:
    logger.info("Loading OHLCV from %s", path)
    df = pd.read_csv(path)
    df = _ensure_open_time_ms(df)
    df = df.sort_values("open_time_ms").drop_duplicates("open_time_ms").reset_index(drop=True)
    return df


def load_delta(path: Path, ohlcv: pd.DataFrame) -> pd.DataFrame:
    logger.info("Loading delta stream from %s", path)
    delta = pd.read_parquet(path)
    if set(delta.columns) >= {"open_time_ms", "delta"}:
        delta = delta[["open_time_ms", "delta"]]
    else:
        raise ValueError("delta_stream.parquet missing required columns ['open_time_ms','delta']")
    aligned = ohlcv[["open_time_ms"]].merge(delta, on="open_time_ms", how="left").fillna({"delta": 0.0})
    return aligned


def load_microstructure(path: Path, ohlcv: pd.DataFrame) -> pd.DataFrame:
    logger.info("Loading microstructure metrics from %s", path)
    micro = pd.read_parquet(path)
    micro = _ensure_open_time_ms(micro)
    aligned = ohlcv[["open_time_ms"]].merge(micro, on="open_time_ms", how="left").fillna(0.0)
    return aligned


def load_agg_trades(directory: Path, return_in_memory: bool = False) -> Tuple[Optional[pd.DataFrame], int, Path]:
    logger.info("Loading aggregated trades from %s", directory)
    processed = PROCESSED_DIR / "agg_trades.parquet"
    if processed.exists():
        logger.info("Cached agg_trades found at %s", processed)
        total_rows = 0
        try:
            import pyarrow.parquet as pq  # type: ignore
            total_rows = pq.ParquetFile(processed).metadata.num_rows
        except Exception:
            total_rows = 0
        if return_in_memory:
            logger.info("Reading cached agg_trades into memory (may be large)")
            return pd.read_parquet(processed), total_rows, processed
        # keep memory low: return None and metadata
        return None, total_rows, processed

    files = sorted(directory.glob("*.zip"))
    if not files:
        logger.warning("No aggTrades zip files found in %s", directory)
        return None, 0, processed

    # Stream and write to parquet to avoid keeping everything in memory
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise ImportError("pyarrow required to stream aggTrades") from exc

    writer = None
    total_rows = 0
    for f in files:
        logger.info("Streaming %s", f.name)
        iter_csv = pd.read_csv(
            f,
            compression="zip",
            usecols=["price", "quantity", "transact_time", "is_buyer_maker"],
            dtype={
                "price": "float32",
                "quantity": "float32",
                "transact_time": "int64",
                "is_buyer_maker": "string",
            },
            chunksize=CHUNK_SIZE,
        )
        for chunk in iter_csv:
            chunk = chunk.rename(columns={"quantity": "qty", "transact_time": "ts"})
            chunk["is_buyer_maker"] = (
                chunk["is_buyer_maker"].str.lower().map({"true": True, "false": False})
            )
            chunk = chunk.dropna(subset=["price", "qty", "ts", "is_buyer_maker"])
            chunk["minute"] = (chunk["ts"] // 60_000) * 60_000
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(processed, table.schema, compression="snappy")
            writer.write_table(table)
            total_rows += len(chunk)
    if writer:
        writer.close()
    logger.info("Agg trades written to %s (%s rows)", processed, f"{total_rows:,}")
    if return_in_memory:
        return pd.read_parquet(processed), total_rows, processed
    return None, total_rows, processed


def load_htf(htf_files: Dict[str, Path], ohlcv: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    htf = {}
    idx = ohlcv[["open_time_ms"]]
    for tf, path in htf_files.items():
        logger.info("Loading HTF %s from %s", tf, path)
        df = pd.read_parquet(path)
        df = _ensure_open_time_ms(df)
        df = df.sort_values("open_time_ms").drop_duplicates("open_time_ms")
        # align via asof forward fill
        merged = pd.merge_asof(
            idx.sort_values("open_time_ms"),
            df.sort_values("open_time_ms"),
            on="open_time_ms",
            direction="backward",
        )
        htf[tf] = merged.fillna(method="ffill").fillna(0.0)
    return htf


def validate_alignment(ohlcv: pd.DataFrame, aligned_frames: List[pd.DataFrame]) -> None:
    n = len(ohlcv)
    for frame in aligned_frames:
        if len(frame) != n:
            raise ValueError("Alignment error: frame length mismatch with OHLCV.")
        if not (frame["open_time_ms"].values == ohlcv["open_time_ms"].values).all():
            raise ValueError("Alignment error: open_time_ms mismatch.")

def load_raw_data(return_agg_trades: bool = False) -> Dict[str, object]:
    """
    Loads and standardises all raw inputs, returning canonical_raw_dataset.

    Parameters
    ----------
    return_agg_trades : bool
        If True, loads agg_trades.parquet into memory.
        If False: returns only metadata + path (memory safe).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    processed_files = {
        "ohlcv": PROCESSED_DIR / "ohlcv.parquet",
        "delta": PROCESSED_DIR / "delta.parquet",
        "microstructure": PROCESSED_DIR / "microstructure.parquet",
        "agg_trades": PROCESSED_DIR / "agg_trades.parquet",
    }
    processed_htf = {tf: PROCESSED_DIR / f"htf_{tf}.parquet" for tf in HTF_FILES}

    # ==============================================================
    # FAST PATH — if everything already processed
    # ==============================================================
    if all(p.exists() for p in processed_files.values()) and \
       all(p.exists() for p in processed_htf.values()):

        logger.info("Processed raw-layer data found. Loading from %s", PROCESSED_DIR)

        ohlcv = pd.read_parquet(processed_files["ohlcv"])
        delta = pd.read_parquet(processed_files["delta"])
        micro = pd.read_parquet(processed_files["microstructure"])

        agg_path = processed_files["agg_trades"]
        try:
            import pyarrow.parquet as pq
            agg_rows = pq.ParquetFile(agg_path).metadata.num_rows
        except Exception:
            agg_rows = 0

        agg_df = pd.read_parquet(agg_path) if return_agg_trades else None
        htf = {tf: pd.read_parquet(path) for tf, path in processed_htf.items()}

        return {
            "ohlcv": ohlcv,
            "delta": delta,
            "microstructure": micro,
            "agg_trades": agg_df,
            "agg_trades_rows": agg_rows,
            "agg_trades_path": agg_path,
            "htf": htf,
        }

    # ==============================================================
    # SLOW PATH — build processed raw layer
    # ==============================================================
    logger.info("Processed raw data missing. Building from raw sources...")

    ohlcv = load_ohlcv(OHLCV_PATH)
    delta = load_delta(DELTA_PATH, ohlcv)
    micro = load_microstructure(MICRO_PATH, ohlcv)

    agg_df, agg_rows, agg_path = load_agg_trades(AGG_TRADES_DIR, return_in_memory=False)
    htf = load_htf(HTF_FILES, ohlcv)

    validate_alignment(ohlcv, [delta, micro])

    # save processed outputs
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ohlcv.to_parquet(processed_files["ohlcv"], index=False)
    delta.to_parquet(processed_files["delta"], index=False)
    micro.to_parquet(processed_files["microstructure"], index=False)

    # agg trades parquet is written incrementally via ParquetWriter
    # so no need to write here unless it's in-memory
    if agg_df is not None:
        agg_df.to_parquet(processed_files["agg_trades"], index=False)

    for tf, df in htf.items():
        df.to_parquet(processed_htf[tf], index=False)

    return {
        "ohlcv": ohlcv,
        "delta": delta,
        "microstructure": micro,
        "agg_trades": None,
        "agg_trades_rows": agg_rows,
        "agg_trades_path": agg_path,
        "htf": htf,
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    data = load_raw_data()
    logger.info(
        "Raw data layer completed. ohlcv=%s rows, delta=%s rows, micro=%s rows, agg_trades=%s rows",
        len(data["ohlcv"]),
        len(data["delta"]),
        len(data["microstructure"]),
        data.get("agg_trades_rows", len(data["agg_trades"]) if data["agg_trades"] is not None else 0),
    )
