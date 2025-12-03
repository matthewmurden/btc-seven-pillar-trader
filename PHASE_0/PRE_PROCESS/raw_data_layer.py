import logging
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Paths are resolved relative to the repository root (two levels above this file).
_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = _ROOT / "PILLAR_FOUR_REGIME_DETECTION" / "data" / "raw"
PROCESSED_DIR = _ROOT / "PILLAR_FOUR_REGIME_DETECTION" / "data" / "processed" / "raw_layer"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

HTF_TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]


def _get_logger(logger: Optional[logging.Logger] = None) -> logging.Logger:
    if logger is not None:
        return logger
    logger = logging.getLogger("phase0.raw")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _ensure_monotonic(series: pd.Series, label: str) -> None:
    if not series.is_monotonic_increasing:
        raise ValueError(f"{label} must be strictly non-decreasing for merge alignment")


def _save_parquet(df: pd.DataFrame, path: Path, logger: logging.Logger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info("Saved %s (%d rows, %d cols)", path, len(df), df.shape[1])


def _read_cached(path: Path, loader_fn, logger: logging.Logger) -> pd.DataFrame:
    if path.exists():
        logger.info("Loading cached %s", path.name)
        return pd.read_parquet(path, engine="pyarrow")
    df = loader_fn()
    _save_parquet(df, path, logger)
    return df


def load_ohlcv(logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    logger = _get_logger(logger)
    raw_path = RAW_DIR / "candles" / "BTCUSDT_1m_2022-11-01_2025-10-31.csv"
    out_path = PROCESSED_DIR / "ohlcv.parquet"

    def _loader() -> pd.DataFrame:
        logger.info("Loading OHLCV from %s", raw_path)
        df = pd.read_csv(raw_path)
        df["open_time_ms"] = pd.to_datetime(df["open_time"], utc=True).astype("int64") // 10**6
        df = df.sort_values("open_time_ms")
        _ensure_monotonic(df["open_time_ms"], "OHLCV open_time_ms")
        expected_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "taker_buy_base",
            "taker_buy_quote",
        ]
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing OHLCV columns: {missing}")
        df = df[["open_time_ms"] + expected_cols]
        logger.info("OHLCV loaded: %d rows", len(df))
        return df

    return _read_cached(out_path, _loader, logger)


def load_delta(logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    logger = _get_logger(logger)
    raw_path = RAW_DIR / "delta" / "delta_stream.parquet"
    out_path = PROCESSED_DIR / "delta.parquet"

    def _loader() -> pd.DataFrame:
        logger.info("Loading delta stream from %s", raw_path)
        df = pd.read_parquet(raw_path, engine="pyarrow")
        if "open_time_ms" not in df.columns or "delta" not in df.columns:
            raise ValueError("delta_stream.parquet must contain open_time_ms and delta")
        df = df.sort_values("open_time_ms").reset_index(drop=True)
        _ensure_monotonic(df["open_time_ms"], "delta open_time_ms")
        logger.info("Delta loaded: %d rows", len(df))
        return df[["open_time_ms", "delta"]]

    return _read_cached(out_path, _loader, logger)


def load_microstructure(logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    logger = _get_logger(logger)
    raw_path = RAW_DIR / "microstructure" / "microstructure_metrics.parquet"
    out_path = PROCESSED_DIR / "microstructure.parquet"

    def _loader() -> pd.DataFrame:
        logger.info("Loading microstructure metrics from %s", raw_path)
        df = pd.read_parquet(raw_path, engine="pyarrow")
        if "open_time_ms" not in df.columns:
            raise ValueError("microstructure_metrics.parquet must contain open_time_ms")
        df = df.sort_values("open_time_ms").reset_index(drop=True)
        _ensure_monotonic(df["open_time_ms"], "microstructure open_time_ms")
        logger.info("Microstructure loaded: %d rows", len(df))
        return df

    return _read_cached(out_path, _loader, logger)


def _load_single_htf(tf: str, logger: logging.Logger) -> pd.DataFrame:
    raw_path = RAW_DIR / "htf_candles" / f"BTCUSDT_{tf}.parquet"
    out_path = PROCESSED_DIR / f"htf_{tf}.parquet"

    def _loader() -> pd.DataFrame:
        logger.info("Loading HTF %s from %s", tf, raw_path)
        df = pd.read_parquet(raw_path, engine="pyarrow")
        if "open_time" in df.columns and "open_time_ms" not in df.columns:
            df["open_time_ms"] = pd.to_datetime(df["open_time"], utc=True).astype("int64") // 10**6
        if "open_time_ms" not in df.columns:
            raise ValueError(f"{raw_path} must contain open_time_ms or open_time")
        df = df.sort_values("open_time_ms").reset_index(drop=True)
        _ensure_monotonic(df["open_time_ms"], f"HTF {tf} open_time_ms")
        rename_map = {
            col: f"{tf}_{col}"
            for col in df.columns
            if col not in {"open_time_ms", "open_time"}
        }
        df = df[["open_time_ms"] + [c for c in df.columns if c not in {"open_time", "open_time_ms"}]]
        df = df.rename(columns=rename_map)
        logger.info("HTF %s loaded: %d rows", tf, len(df))
        return df

    return _read_cached(out_path, _loader, logger)


def load_all_htf(logger: Optional[logging.Logger] = None) -> Dict[str, pd.DataFrame]:
    logger = _get_logger(logger)
    return {tf: _load_single_htf(tf, logger) for tf in HTF_TIMEFRAMES}


def _aggregate_trades_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.rename(columns={"transact_time": "open_time_ms"})
    chunk["open_time_ms"] = (chunk["open_time_ms"] // 60000) * 60000
    chunk["quote_value"] = chunk["price"] * chunk["quantity"]
    grouped = chunk.groupby("open_time_ms").agg(
        agg_trade_count=("agg_trade_id", "count"),
        agg_base_volume=("quantity", "sum"),
        agg_quote_volume=("quote_value", "sum"),
        agg_buy_maker_ratio=("is_buyer_maker", "mean"),
    )
    grouped["agg_vwap"] = grouped["agg_quote_volume"] / grouped["agg_base_volume"].replace(0, np.nan)
    return grouped.reset_index()


def load_agg_trades(
    canonical_index: Optional[pd.Series] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    logger = _get_logger(logger)
    out_path = PROCESSED_DIR / "agg_trades.parquet"

    def _loader() -> pd.DataFrame:
        trade_dir = RAW_DIR / "bulk_trades"
        zip_files = sorted(trade_dir.glob("BTCUSDT-aggTrades-*.zip"))
        if not zip_files:
            logger.warning("No aggTrades zip files found in %s", trade_dir)
            return pd.DataFrame(columns=["open_time_ms", "agg_trade_count", "agg_base_volume", "agg_quote_volume", "agg_buy_maker_ratio", "agg_vwap"])

        writer: Optional[pq.ParquetWriter] = None
        temp_path = out_path.with_suffix(".tmp.parquet")
        if temp_path.exists():
            temp_path.unlink()
        logger.info("Streaming %d aggTrades files into %s", len(zip_files), temp_path)
        for zip_file in zip_files:
            logger.info("Processing %s", zip_file.name)
            with zipfile.ZipFile(zip_file) as zf:
                names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not names:
                    continue
                with zf.open(names[0]) as fh:
                    for chunk in pd.read_csv(fh, chunksize=500_000):
                        aggregated = _aggregate_trades_chunk(chunk)
                        table = pa.Table.from_pandas(aggregated, preserve_index=False)
                        if writer is None:
                            writer = pq.ParquetWriter(temp_path, table.schema)
                        writer.write_table(table)
        if writer is not None:
            writer.close()
        else:
            logger.warning("No aggTrades data written; returning empty frame")
            return pd.DataFrame(columns=["open_time_ms", "agg_trade_count", "agg_base_volume", "agg_quote_volume", "agg_buy_maker_ratio", "agg_vwap"])

        logger.info("Reading back aggregated trades for consolidation")
        agg_df = pd.read_parquet(temp_path, engine="pyarrow")
        agg_df = agg_df.groupby("open_time_ms", as_index=False).agg(
            agg_trade_count=("agg_trade_count", "sum"),
            agg_base_volume=("agg_base_volume", "sum"),
            agg_quote_volume=("agg_quote_volume", "sum"),
            agg_buy_maker_ratio=("agg_buy_maker_ratio", "mean"),
            agg_vwap=("agg_vwap", "mean"),
        )
        agg_df = agg_df.sort_values("open_time_ms").reset_index(drop=True)
        temp_path.unlink(missing_ok=True)
        logger.info("Aggregated trades consolidated: %d rows", len(agg_df))
        return agg_df

    agg_trades = _read_cached(out_path, _loader, logger)
    if canonical_index is None or agg_trades.empty:
        return agg_trades

    aligned = pd.DataFrame({"open_time_ms": canonical_index})
    aligned = aligned.merge(agg_trades, on="open_time_ms", how="left")
    _ensure_monotonic(aligned["open_time_ms"], "agg_trades aligned")
    return aligned


def build_canonical_raw(logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    logger = _get_logger(logger)
    start = time.time()
    ohlcv = load_ohlcv(logger)
    delta = load_delta(logger)
    micro = load_microstructure(logger)
    agg_trades = load_agg_trades(ohlcv["open_time_ms"], logger)
    htf_map = load_all_htf(logger)

    df = ohlcv.copy()
    df = df.merge(delta, on="open_time_ms", how="left")
    df = df.merge(micro, on="open_time_ms", how="left")
    if not agg_trades.empty:
        df = df.merge(agg_trades, on="open_time_ms", how="left")

    df = df.sort_values("open_time_ms").reset_index(drop=True)
    _ensure_monotonic(df["open_time_ms"], "canonical open_time_ms before HTF merge")

    for tf, htf_df in htf_map.items():
        df = pd.merge_asof(
            df,
            htf_df.sort_values("open_time_ms"),
            on="open_time_ms",
            direction="backward",
        )
        logger.info("Merged HTF %s → shape now %s", tf, df.shape)

    if len(df) != len(ohlcv):
        raise ValueError(f"Canonical dataset length mismatch: {len(df)} vs OHLCV {len(ohlcv)}")

    elapsed = time.time() - start
    logger.info("Canonical raw built in %.2fs with shape %s", elapsed, df.shape)
    return df


def load_raw_data(logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    logger = _get_logger(logger)
    out_path = PROCESSED_DIR / "canonical_raw.parquet"
    if out_path.exists():
        logger.info("Loading cached canonical raw from %s", out_path)
        return pd.read_parquet(out_path, engine="pyarrow")
    df = build_canonical_raw(logger)
    _save_parquet(df, out_path, logger)
    return df


__all__ = [
    "load_raw_data",
    "build_canonical_raw",
    "load_ohlcv",
    "load_delta",
    "load_microstructure",
    "load_all_htf",
    "load_agg_trades",
    "HTF_TIMEFRAMES",
    "RAW_DIR",
    "PROCESSED_DIR",
]
