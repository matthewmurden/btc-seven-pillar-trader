"""
Offline data fetcher for optimisation runs (MAC-GUARDED).

Key problem:
- 6 months of BTCUSDT futures aggTrades is enormous.
- Even if download is chunked, downstream (delta_and_micro / trade_aggregator) often loads too much.
- On an 8GB Mac this can hard-OOM and crash.

Fix:
- Still fetch full 1m klines.
- For aggTrades: write a DOWNSAMPLED stream with a HARD ROW CAP per window.
  This keeps downstream memory bounded and prevents OS-kill.

Outputs written:
  RUN_LAYER/data/raw/BTC_CANDLES.csv
  RUN_LAYER/data/raw/trades/trades_BTCUSDT.csv

Notes:
- Uses https://data.binance.vision (futures/um daily zips).
- Cache stored under RUN_LAYER_OPTIMISING/data/cache
"""

from __future__ import annotations

import logging
import os
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Tuple, Optional, List

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision"
UM_PATH = "data/futures/um/daily"
SYMBOL = "BTCUSDT"
INTERVAL = "1m"

# Defaults (run_hparam_search.py patches RAW_DIR/TRADES_DIR/HTF_DIR at runtime)
BASE_ROOT = Path(__file__).resolve().parents[2]  # project root
RAW_DIR = BASE_ROOT / "RUN_LAYER" / "data" / "raw"
TRADES_DIR = RAW_DIR / "trades"
HTF_DIR = RAW_DIR / "htf_candles"  # not used here, but runner patches it

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"

# -----------------------------
# Guardrails (MAC SAFE DEFAULTS)
# -----------------------------
# Hard cap total trade rows written per window. This is the key OOM stopper.
MAX_TRADE_ROWS_PER_WINDOW = int(os.getenv("OFFLINE_FETCH_MAX_TRADE_ROWS", "1500000"))

# Chunk rows read from aggTrades zip member.
AGG_CHUNK_ROWS = int(os.getenv("OFFLINE_FETCH_AGG_CHUNK_ROWS", "250000"))

# Cache pruning (to avoid blowing disk)
CACHE_MAX_GB = float(os.getenv("OFFLINE_FETCH_CACHE_MAX_GB", "8.0"))

HTTP_TIMEOUT = int(os.getenv("OFFLINE_FETCH_HTTP_TIMEOUT", "60"))
HTTP_RETRIES = int(os.getenv("OFFLINE_FETCH_HTTP_RETRIES", "3"))


def _daterange(start: datetime, end: datetime) -> Iterable[datetime]:
    cur = start
    while cur < end:
        yield cur
        cur += timedelta(days=1)


def _cache_size_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for p in root.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except FileNotFoundError:
            pass
    return total


def _prune_cache_if_needed() -> None:
    if CACHE_MAX_GB <= 0:
        return
    max_bytes = int(CACHE_MAX_GB * (1024**3))
    cur_bytes = _cache_size_bytes(CACHE_DIR)
    if cur_bytes <= max_bytes:
        return

    # Delete oldest files first
    files: List[Path] = []
    for p in CACHE_DIR.rglob("*"):
        if p.is_file():
            files.append(p)

    files.sort(key=lambda p: p.stat().st_mtime if p.exists() else time.time())
    LOGGER.warning("Cache too big (%.2fGB > %.2fGB). Pruning...", cur_bytes / (1024**3), CACHE_MAX_GB)

    freed = 0
    for f in files:
        try:
            sz = f.stat().st_size
            f.unlink(missing_ok=True)  # type: ignore[arg-type]
            freed += sz
        except Exception:
            continue
        cur_bytes -= sz
        if cur_bytes <= max_bytes:
            break

    LOGGER.warning("Cache pruned. Freed %.2fGB. Current cache %.2fGB.", freed / (1024**3), cur_bytes / (1024**3))


def _stream_download_to_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    last_err: Optional[Exception] = None
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            LOGGER.info("Downloading (stream) %s", url)
            with requests.get(url, stream=True, timeout=HTTP_TIMEOUT) as resp:
                resp.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            _prune_cache_if_needed()
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            LOGGER.warning("Download failed (attempt %d/%d): %s", attempt, HTTP_RETRIES, e)

    raise RuntimeError(f"Failed to download after {HTTP_RETRIES} attempts: {url}") from last_err


def _get_zip_cached(url: str, cache_file: Path) -> Path:
    if cache_file.exists() and cache_file.stat().st_size > 0:
        return cache_file
    _stream_download_to_file(url, cache_file)
    return cache_file


def _zip_first_member(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if not names:
            raise RuntimeError(f"Zip has no members: {zip_path}")
        return names[0]


def _read_zip_csv_full(zip_path: Path, *, names: List[str]) -> pd.DataFrame:
    member = _zip_first_member(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as f:
            return pd.read_csv(f, header=None, names=names)


def _iter_zip_csv_chunks(zip_path: Path, *, names: List[str], chunksize: int):
    member = _zip_first_member(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as f:
            yield from pd.read_csv(
                f,
                header=None,
                names=names,
                chunksize=chunksize,
                low_memory=True,
            )


def _download_klines_zip_for_day(day: datetime) -> Path:
    fname = f"{SYMBOL}-{INTERVAL}-{day.strftime('%Y-%m-%d')}.zip"
    url = f"{BASE_URL}/{UM_PATH}/klines/{SYMBOL}/{INTERVAL}/{fname}"
    cache_file = CACHE_DIR / "klines" / SYMBOL / INTERVAL / fname
    return _get_zip_cached(url, cache_file)


def _download_aggtrades_zip_for_day(day: datetime) -> Path:
    fname = f"{SYMBOL}-aggTrades-{day.strftime('%Y-%m-%d')}.zip"
    url = f"{BASE_URL}/{UM_PATH}/aggTrades/{SYMBOL}/{fname}"
    cache_file = CACHE_DIR / "aggTrades" / SYMBOL / fname
    return _get_zip_cached(url, cache_file)


def _slice_ms(df: pd.DataFrame, col: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    return df[(df[col] >= start_ms) & (df[col] < end_ms)].copy()


def _is_header_row(df: pd.DataFrame, expected_first: str) -> bool:
    try:
        v = str(df.iloc[0, 0]).strip().lower()
        return v == expected_first.lower()
    except Exception:
        return False


def fetch_raw_window(start_ms: int, end_ms: int) -> None:
    """
    Download daily klines + aggTrades covering [start_ms, end_ms) and write:
      - RUN_LAYER/data/raw/BTC_CANDLES.csv
      - RUN_LAYER/data/raw/trades/trades_BTCUSDT.csv

    IMPORTANT:
      aggTrades output is hard-capped & downsampled to prevent OOM on 8GB Macs.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TRADES_DIR.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    end_dt = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

    candles_out = RAW_DIR / "BTC_CANDLES.csv"
    trades_out = TRADES_DIR / f"trades_{SYMBOL}.csv"

    candles_tmp = candles_out.with_suffix(".csv.tmp")
    trades_tmp = trades_out.with_suffix(".csv.tmp")
    if candles_tmp.exists():
        candles_tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
    if trades_tmp.exists():
        trades_tmp.unlink(missing_ok=True)  # type: ignore[arg-type]

    # ---- KLINES schema ----
    kline_cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
    ]

    wrote_candles_header = False
    wrote_any_candles = False

    # ---- AGGTRADES schema (vision daily is typically 7 cols) ----
    agg_cols = [
        "agg_trade_id", "price", "qty", "first_trade_id", "last_trade_id", "timestamp", "is_buyer_maker",
    ]

    wrote_trades_header = False
    wrote_any_trades = False
    total_written_trades = 0

    total_days = sum(1 for _ in _daterange(start_dt, end_dt))
    done_days = 0

    LOGGER.info(
        "Fetching %s..%s | trade cap=%d rows/window | cache max=%.1fGB",
        start_dt, end_dt, MAX_TRADE_ROWS_PER_WINDOW, CACHE_MAX_GB
    )

    for day in _daterange(start_dt, end_dt):
        # ---------------------
        # KLINES (small enough)
        # ---------------------
        try:
            zp = _download_klines_zip_for_day(day)
            dfk = _read_zip_csv_full(zp, names=kline_cols)

            if _is_header_row(dfk, "open_time"):
                dfk = dfk.iloc[1:].copy()

            dfk = dfk.iloc[:, :12].copy()
            dfk.columns = kline_cols
            dfk = dfk.drop(columns=["ignore"], errors="ignore")

            num_cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]
            int_cols = ["open_time", "close_time", "trades"]
            for c in num_cols:
                dfk[c] = pd.to_numeric(dfk[c], errors="coerce")
            for c in int_cols:
                dfk[c] = pd.to_numeric(dfk[c], errors="coerce")

            dfk = dfk.dropna(subset=["open_time", "close_time"]).copy()
            dfk["open_time"] = dfk["open_time"].astype("int64")
            dfk["close_time"] = dfk["close_time"].astype("int64")
            dfk["trades"] = dfk["trades"].fillna(0).astype("int64")

            dfk = _slice_ms(dfk, "open_time", start_ms, end_ms)
            if not dfk.empty:
                dfk["open_time"] = pd.to_datetime(dfk["open_time"], unit="ms", utc=True)
                dfk["close_time"] = pd.to_datetime(dfk["close_time"], unit="ms", utc=True)

                dfk.to_csv(candles_tmp, index=False, mode="a", header=not wrote_candles_header)
                wrote_candles_header = True
                wrote_any_candles = True

        except Exception as e:  # noqa: BLE001
            LOGGER.warning("Klines failed for %s: %s", day.date(), e)

        # --------------------------
        # AGGTRADES (cap + sample!)
        # --------------------------
        try:
            if total_written_trades >= MAX_TRADE_ROWS_PER_WINDOW:
                # We already hit the cap; skip remaining days entirely.
                pass
            else:
                zp = _download_aggtrades_zip_for_day(day)

                for chunk in _iter_zip_csv_chunks(zp, names=agg_cols, chunksize=AGG_CHUNK_ROWS):
                    if chunk is None or chunk.empty:
                        continue

                    # Drop header row if present (some files have headers)
                    if _is_header_row(chunk, "aggTradeId") or _is_header_row(chunk, "agg_trade_id"):
                        chunk = chunk.iloc[1:].copy()
                        if chunk.empty:
                            continue

                    # Coerce types
                    chunk["timestamp"] = pd.to_numeric(chunk["timestamp"], errors="coerce")
                    chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")
                    chunk["qty"] = pd.to_numeric(chunk["qty"], errors="coerce")

                    chunk = chunk.dropna(subset=["timestamp", "price", "qty", "is_buyer_maker"]).copy()
                    if chunk.empty:
                        continue

                    chunk["timestamp"] = chunk["timestamp"].astype("int64")
                    chunk["price"] = chunk["price"].astype("float64")
                    chunk["qty"] = chunk["qty"].astype("float64")
                    chunk["is_buyer_maker"] = chunk["is_buyer_maker"].astype(str).str.lower()
                    chunk = chunk[chunk["is_buyer_maker"].isin(["true", "false"])].copy()
                    if chunk.empty:
                        continue

                    chunk = _slice_ms(chunk, "timestamp", start_ms, end_ms)
                    if chunk.empty:
                        continue

                    remaining = MAX_TRADE_ROWS_PER_WINDOW - total_written_trades
                    if remaining <= 0:
                        break

                    # If chunk is too big for remaining budget, downsample deterministically.
                    if len(chunk) > remaining:
                        step = max(1, len(chunk) // max(1, remaining))
                        chunk = chunk.iloc[::step].copy()

                        # still may exceed by a tiny amount due to integer math
                        if len(chunk) > remaining:
                            chunk = chunk.iloc[:remaining].copy()

                    if chunk.empty:
                        continue

                    out = chunk[["timestamp", "price", "qty", "is_buyer_maker"]].copy()
                    out.to_csv(trades_tmp, index=False, mode="a", header=not wrote_trades_header)

                    wrote_trades_header = True
                    wrote_any_trades = True
                    total_written_trades += len(out)

                    if total_written_trades >= MAX_TRADE_ROWS_PER_WINDOW:
                        break

        except Exception as e:  # noqa: BLE001
            LOGGER.warning("AggTrades failed for %s: %s", day.date(), e)

        done_days += 1
        if done_days % 5 == 0 or done_days == total_days:
            LOGGER.info(
                "Progress %d/%d days | trades written=%d/%d",
                done_days, total_days, total_written_trades, MAX_TRADE_ROWS_PER_WINDOW
            )

    if not wrote_any_candles:
        raise RuntimeError("No candles written. Check dates/network/cache.")
    if not wrote_any_trades:
        raise RuntimeError("No trades written. Check dates/network/cache.")

    candles_tmp.replace(candles_out)
    trades_tmp.replace(trades_out)

    LOGGER.info("Wrote candles: %s", candles_out)
    LOGGER.info("Wrote trades : %s (rows=%d cap=%d)", trades_out, total_written_trades, MAX_TRADE_ROWS_PER_WINDOW)
    LOGGER.info("Raw fetch complete. Derived artifacts will be built by post-raw chain.")


def window_for_fold(test_start: datetime) -> Tuple[int, int]:
    test_start = test_start.replace(second=0, microsecond=0, tzinfo=timezone.utc)
    train_start = test_start - timedelta(days=30)
    test_end = test_start + timedelta(days=7)
    window_start = int(train_start.timestamp() * 1000)
    window_end = int(test_end.timestamp() * 1000)
    return window_start, window_end