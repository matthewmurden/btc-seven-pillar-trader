import logging
import sys
import time
import zipfile
from io import BytesIO
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import requests
from requests.exceptions import HTTPError
import warnings
import math

# ============================================================
# CONFIG / PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
DELTA_OUT = RAW_DIR / "delta.parquet"
MICRO_OUT = RAW_DIR / "microstructure.parquet"

TRADES_DIR = RAW_DIR / "trades"   # where we store per-run trades
TRADES_DIR.mkdir(parents=True, exist_ok=True)
TRADES_OUT = TRADES_DIR / "trades_BTCUSDT.csv"

CANDLES_PATH = RAW_DIR / "BTC_CANDLES.csv"

SYMBOL = "BTCUSDT"
BASE_URL = "https://fapi.binance.com"
ENDPOINT = "/fapi/v1/aggTrades"

# Optional: use Binance daily aggTrades dumps for the portion of the lookback
# that lies before today's UTC 00:00. This reduces API calls when the window
# crosses midnight.
USE_DAILY_DUMPS = True
BINANCE_VISION_BASE = (
    "https://data.binance.vision/data/futures/um/daily/aggTrades"
)
# Toggle to reduce verbose per-request logging and show a simple progress bar
SHOW_PROGRESS = True

# How much history you want in the final window.
# Example: LOOKBACK_DAYS = 30 for 30 days; right now ~30 minutes for testing
LOOKBACK_DAYS = 1 / 48  # 1/48 day ≈ 30 minutes

# Optional override for the exclusive end timestamp (ms). When provided,
# the pipeline will end exactly at that ms instead of "now".
FORCE_END_MS: int | None = None

# Binance / rate control
LIMIT = 1000           # max trades per aggTrades request
REQUEST_SLEEP = 0.25    # pause after each successful request (seconds)
WINDOW_HOURS = 1       # size of each time window (1h chunks)
WINDOW_SLEEP = 0     # pause after finishing each window (seconds)

# Retry safety so we don't get stuck forever on 429s / glitches
MAX_429_RETRIES_PER_WINDOW = 6
MAX_GENERIC_RETRIES_PER_WINDOW = 6

EPS = 1e-12  # safety epsilon
# Silence noisy FutureWarnings from groupby.apply deprecation
warnings.filterwarnings("ignore", category=FutureWarning)

# Microstructure columns that we persist to MICRO_OUT
MICRO_FEATURE_COLS = [
    "open_time_ms",
    "wick_body_ratio",
    "micro_volatility",
    "micro_return",
    "entropy",
    "order_imbalance",
    "micro_delta",
    "delta_persistence",
    "micro_clustering",
    "alternation_index",
    "no_trend_stat",
]


# ============================================================
# LOGGING
# ============================================================
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ============================================================
# API CALL
# ============================================================
def get_agg_trades(symbol: str, start_ms: int, end_ms: int, limit: int = LIMIT) -> List[dict]:
    """
    Fetch a batch of aggregated trades from Binance Futures.

    Response objects (aggTrades):
      {
        "a": aggTradeId,
        "p": price,
        "q": qty,
        "f": firstTradeId,
        "l": lastTradeId,
        "T": timestamp (ms),
        "m": isBuyerMaker,
        "M": isBestMatch
      }
    """
    url = f"{BASE_URL}{ENDPOINT}"
    params = {
        "symbol": symbol,
        "startTime": int(start_ms),
        "endTime": int(end_ms),
        "limit": limit,
    }

    if not SHOW_PROGRESS:
        logging.info(
            "Requesting aggTrades: %s | params=%s",
            url,
            params,
        )

    resp = requests.get(url, params=params, timeout=10)

    if not SHOW_PROGRESS:
        # Log basic response info before raising on error
        logging.info(
            "aggTrades response: status=%s, bytes=%s",
            resp.status_code,
            len(resp.content),
        )

    resp.raise_for_status()
    data = resp.json()

    if not isinstance(data, list):
        logging.error("Unexpected aggTrades response payload: %s", data)
        raise RuntimeError(f"Unexpected aggTrades response: {data}")

    # Also log how many trades we actually got in this raw payload
    if not SHOW_PROGRESS:
        logging.info(
            "aggTrades payload decoded: trades=%s (start_ms=%s, end_ms=%s)",
            len(data),
            start_ms,
            end_ms,
        )

    return data


# ============================================================
# STREAM TRADES OVER RANGE (USED PER HOUR)
# ============================================================
def stream_trades_range(start_ms: int, end_ms: int) -> Iterable[pd.DataFrame]:
    """
    Streams aggTrades between [start_ms, end_ms) in WINDOW_HOURS chunks.

    Yields DataFrames with columns:
      - price
      - qty
      - timestamp
      - is_buyer_maker
    """
    start_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)

    logging.info(
        "Fetching aggTrades for %s from %s to %s",
        SYMBOL,
        start_dt,
        end_dt,
    )

    window_ms = WINDOW_HOURS * 60 * 60 * 1000
    window_start = start_ms
    window_idx = 0
    total_trades = 0
    total_range_ms = end_ms - start_ms
    total_windows = max(1, math.ceil(total_range_ms / max(window_ms, 1)))

    while window_start < end_ms:
        window_idx += 1
        window_end = min(window_start + window_ms, end_ms)

        ws_dt = datetime.fromtimestamp(window_start / 1000, tz=timezone.utc)
        we_dt = datetime.fromtimestamp(window_end / 1000, tz=timezone.utc)
        logging.info("=== Window %d: %s -> %s ===", window_idx, ws_dt, we_dt)

        cursor = window_start
        backoff_429 = 10.0  # exponential backoff starting point
        n_429 = 0
        n_generic_err = 0

        while cursor < window_end:
            try:
                trades = get_agg_trades(SYMBOL, cursor, window_end, limit=LIMIT)

                if not trades:
                    # No trades in this subrange → skip ahead 1 minute
                    cursor = min(cursor + 60_000, window_end)
                    continue

                df_raw = pd.DataFrame(trades)

                # Map Binance fields → our schema
                df = df_raw[["p", "q", "T", "m"]].rename(
                    columns={
                        "p": "price",
                        "q": "qty",
                        "T": "timestamp",
                        "m": "is_buyer_maker",
                    }
                )

                # ---- SAFE TYPE COERCION ----
                df["price"] = pd.to_numeric(df["price"], errors="coerce")
                df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
                df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

                df = df.dropna(subset=["timestamp", "qty"])

                if df.empty:
                    cursor = min(cursor + 60_000, window_end)
                    continue

                df["timestamp"] = df["timestamp"].astype("int64")
                df["qty"] = df["qty"].astype("float64")
                df["price"] = df["price"].astype("float64")

                if df["is_buyer_maker"].dtype != bool:
                    df["is_buyer_maker"] = (
                        df["is_buyer_maker"]
                        .astype(str)
                        .str.lower()
                        .map({"true": True, "false": False, "1": True, "0": False})
                    )
                df = df.dropna(subset=["is_buyer_maker"])
                df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)

                if df.empty:
                    cursor = min(cursor + 60_000, window_end)
                    continue

                chunk_size = len(df)
                total_trades += chunk_size
                if SHOW_PROGRESS:
                    chunk_last_ts = int(df["timestamp"].iloc[-1])
                    progress = (chunk_last_ts - start_ms) / max(total_range_ms, 1)
                    sys.stdout.write(
                        f"\rFetching trades: {progress*100:5.1f}% | window {window_idx}/{total_windows} | total {total_trades:,}"
                    )
                    sys.stdout.flush()
                else:
                    logging.info(
                        "Window %d: fetched %s trades (chunk=%s), total=%s",
                        window_idx,
                        SYMBOL,
                        f"{chunk_size:,}",
                        f"{total_trades:,}",
                    )

                # Reset retry counters after a successful request
                backoff_429 = 10.0
                n_429 = 0
                n_generic_err = 0

                yield df

                last_ts = int(df["timestamp"].iloc[-1])
                cursor = max(last_ts + 1, cursor + 1)

                time.sleep(REQUEST_SLEEP)

            except HTTPError as http_err:
                status = getattr(http_err.response, "status_code", None)
                if status == 429:
                    n_429 += 1
                    logging.warning(
                        "429 Too Many Requests in window %d (attempt %d). "
                        "Backing off for %.1fs...",
                        window_idx,
                        n_429,
                        backoff_429,
                    )
                    time.sleep(backoff_429)
                    backoff_429 = min(backoff_429 * 2.0, 120.0)

                    if n_429 >= MAX_429_RETRIES_PER_WINDOW:
                        logging.error(
                            "Max 429 retries exceeded in window %d. "
                            "Skipping remaining range for this window.",
                            window_idx,
                        )
                        cursor = window_end
                    continue
                else:
                    n_generic_err += 1
                    logging.warning(
                        "HTTP error in window %d (attempt %d): %s. "
                        "Sleeping 10s then retrying...",
                        window_idx,
                        n_generic_err,
                        http_err,
                    )
                    time.sleep(10)

                    if n_generic_err >= MAX_GENERIC_RETRIES_PER_WINDOW:
                        logging.error(
                            "Max generic HTTP retries exceeded in window %d. "
                            "Skipping remaining range for this window.",
                            window_idx,
                        )
                        cursor = window_end
                    continue

            except Exception as exc:  # noqa: BLE001
                n_generic_err += 1
                logging.warning(
                    "Error fetching/processing aggTrades in window %d (attempt %d): %s. "
                    "Sleeping 10s then retrying...",
                    window_idx,
                    n_generic_err,
                    exc,
                )
                time.sleep(10)

                if n_generic_err >= MAX_GENERIC_RETRIES_PER_WINDOW:
                    logging.error(
                        "Max generic retries exceeded in window %d. "
                        "Skipping remaining range for this window.",
                        window_idx,
                    )
                    cursor = window_end
                continue

        logging.info(
            "Finished window %d. Sleeping %.1fs before next window.",
            window_idx,
            WINDOW_SLEEP,
        )
        time.sleep(WINDOW_SLEEP)

        window_start = window_end  # advance to next window

    if SHOW_PROGRESS:
        sys.stdout.write(f"\rFetching trades: 100.0% | window {total_windows}/{total_windows} | total {total_trades:,} | done\n")
        sys.stdout.flush()
    logging.info("All windows complete.")


def _dates_between(start_ms: int, end_ms: int) -> List[datetime.date]:
    start_date = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).date()
    end_date = datetime.fromtimestamp((end_ms - 1) / 1000, tz=timezone.utc).date()
    days = []
    cur = start_date
    while cur <= end_date:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def fetch_trades_from_dumps(start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Attempt to load aggTrades from Binance daily dumps for the date range covering [start_ms, end_ms).
    Returns a DataFrame with columns: price, qty, timestamp, is_buyer_maker.
    """
    dfs: List[pd.DataFrame] = []
    for d in _dates_between(start_ms, end_ms):
        url = f"{BINANCE_VISION_BASE}/{SYMBOL}/{SYMBOL}-aggTrades-{d}.zip"
        logging.info("Attempting daily dump: %s", url)
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                logging.warning("Daily dump not available (%s): HTTP %s", d, resp.status_code)
                continue
            with zipfile.ZipFile(BytesIO(resp.content)) as z:
                members = z.namelist()
                if not members:
                    logging.warning("Daily dump %s zip empty", d)
                    continue
                with z.open(members[0]) as f:
                    df = pd.read_csv(
                        f,
                        header=None,
                        names=[
                            "agg_trade_id",
                            "price",
                            "qty",
                            "first_trade_id",
                            "last_trade_id",
                            "timestamp",
                            "is_buyer_maker",
                        ],
                    )
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
            df = df.dropna(subset=["timestamp", "qty", "price"])
            df["timestamp"] = df["timestamp"].astype("int64")
            df["qty"] = df["qty"].astype("float64")
            df["price"] = df["price"].astype("float64")
            df["is_buyer_maker"] = (
                df["is_buyer_maker"]
                .astype(str)
                .str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
                .astype(bool)
            )
            df = df[(df["timestamp"] >= start_ms) & (df["timestamp"] < end_ms)]
            if not df.empty:
                dfs.append(df[["price", "qty", "timestamp", "is_buyer_maker"]])
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed to load daily dump %s: %s", d, exc)
            continue

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame(columns=["price", "qty", "timestamp", "is_buyer_maker"])


def compute_micro_from_trades_df(df_all: pd.DataFrame, start_ms: int, end_ms: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute microstructure from a preloaded trades DataFrame over [start_ms, end_ms).
    Returns (micro_df, trades_df).
    """
    if df_all.empty:
        return build_empty_micro_frame(start_ms, end_ms), df_all

    df_all = df_all.copy()
    df_all["delta"] = np.where(~df_all["is_buyer_maker"], df_all["qty"], -df_all["qty"])
    df_all["minute"] = (df_all["timestamp"] // 60_000) * 60_000

    micro = compute_minute_microstructure(df_all)
    full_idx = np.arange(start_ms, end_ms, 60_000, dtype="int64")
    micro = micro.reindex(full_idx, fill_value=0.0)
    micro.index.name = "open_time_ms"

    return micro.reset_index(), df_all


# ============================================================
# MICROSTRUCTURE HELPERS
# ============================================================
def longest_run(delta_signs: np.ndarray) -> int:
    """Longest consecutive run of the same sign in an array of {-1,0,1}."""
    longest = 0
    cur = 0
    prev = 0
    for s in delta_signs:
        if s == prev:
            cur += 1
        else:
            cur = 1
            prev = s
        if cur > longest:
            longest = cur
    return longest


def alternation_index(prices: np.ndarray) -> float:
    """
    Fraction of sign flips in price differences within the minute.
    """
    if len(prices) < 2:
        return 0.0
    diffs = np.diff(prices)
    signs = np.sign(diffs)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0.0
    return float(np.mean(signs[1:] != signs[:-1]))


def compute_minute_microstructure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a per-trade DataFrame with columns:
      - price
      - qty
      - timestamp
      - is_buyer_maker
      - delta        (net taker volume per trade)
      - minute       (bucketed ms)
    compute per-minute microstructure stats.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "delta",
                "trade_count",
                "buy_count",
                "sell_count",
                "entropy",
                "delta_persistence",
                "micro_clustering",
                "alternation_index",
                "micro_delta",
            ]
        )

    g = df.groupby("minute")

    out = pd.DataFrame(index=g.size().index)

    # Net volume per minute
    out["delta"] = g["delta"].sum()

    # Counts
    out["trade_count"] = g.size()
    out["buy_count"] = g.apply(lambda x: (x["delta"] > 0).sum())
    out["sell_count"] = g.apply(lambda x: (x["delta"] < 0).sum())

    # Entropy of buy/sell mix
    pb = out["buy_count"] / out["trade_count"]
    ps = out["sell_count"] / out["trade_count"]
    entropy = -(pb * np.log2(pb + EPS) + ps * np.log2(ps + EPS))
    entropy[out["trade_count"] < 5] = 0.0
    out["entropy"] = entropy

    # Delta persistence: longest same-sign streak / trade_count
    out["delta_persistence"] = g.apply(
        lambda x: (
            longest_run(np.sign(x["delta"].values)) / len(x)
            if len(x) >= 3 else 0.0
        )
    )

    # Micro clustering: std of price changes inside the minute
    out["micro_clustering"] = g.apply(
        lambda x: float(np.std(np.diff(x["price"].values))) if len(x) > 1 else 0.0
    )

    # Alternation index: how often price diff sign flips
    out["alternation_index"] = g["price"].apply(
        lambda p: float(alternation_index(p.values))
    )

    out["micro_delta"] = out["delta"]

    return out


def build_empty_micro_frame(start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Build an all-zero microstructure frame for [start_ms, end_ms) at 1m resolution.
    """
    idx = np.arange(start_ms, end_ms, 60_000, dtype="int64")
    return pd.DataFrame(
        {
            "open_time_ms": idx,
            "delta": np.zeros(len(idx), dtype="float64"),
            "micro_delta": np.zeros(len(idx), dtype="float64"),
            "trade_count": np.zeros(len(idx), dtype="float64"),
            "buy_count": np.zeros(len(idx), dtype="float64"),
            "sell_count": np.zeros(len(idx), dtype="float64"),
            "entropy": np.zeros(len(idx), dtype="float64"),
            "delta_persistence": np.zeros(len(idx), dtype="float64"),
            "micro_clustering": np.zeros(len(idx), dtype="float64"),
            "alternation_index": np.zeros(len(idx), dtype="float64"),
        }
    )


# ============================================================
# SAVE TRADES (APPEND TO ONE CSV)
# ============================================================
def append_trades_csv(df_all: pd.DataFrame, header: bool) -> None:
    """
    Append the raw trades to a single CSV under raw/trades.
    """
    df_to_save = df_all.copy()
    df_to_save["timestamp_utc"] = pd.to_datetime(
        df_to_save["timestamp"], unit="ms", utc=True
    )

    cols_order = [
        "timestamp",
        "timestamp_utc",
        "price",
        "qty",
        "is_buyer_maker",
        "delta",
        "minute",
    ]
    existing_cols = [c for c in cols_order if c in df_to_save.columns]
    other_cols = [c for c in df_to_save.columns if c not in existing_cols]
    df_to_save = df_to_save[existing_cols + other_cols]

    df_to_save.to_csv(TRADES_OUT, mode="a", index=False, header=header)
    logging.info(
        "Appended trades → %s (%s rows, header=%s)",
        TRADES_OUT,
        len(df_to_save),
        header,
    )


# ============================================================
# MICROSTRUCTURE FOR A SINGLE [start_ms, end_ms) BUCKET
# ============================================================
def compute_micro_for_range(start_ms: int, end_ms: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch trades in [start_ms, end_ms), compute per-trade delta and
    per-minute microstructure, and return:

      micro_hour: DataFrame with columns
        - open_time_ms
        - delta, micro_delta, trade_count, buy_count, sell_count,
          entropy, delta_persistence, micro_clustering, alternation_index

      trades_hour: DataFrame of raw trades used

    This is called per hour, so memory stays bounded.
    """
    trades_chunks: List[pd.DataFrame] = []

    for chunk in stream_trades_range(start_ms, end_ms):
        trades_chunks.append(chunk)

    if not trades_chunks:
        logging.warning(
            "No trades found for range %s → %s. Using zero micro frame.",
            start_ms,
            end_ms,
        )
        micro = build_empty_micro_frame(start_ms, end_ms)
        empty_trades = pd.DataFrame(
            columns=["timestamp", "price", "qty", "is_buyer_maker", "delta", "minute"]
        )
        return micro, empty_trades

    df_all = pd.concat(trades_chunks, ignore_index=True)

    # Binance semantics:
    # is_buyer_maker == True  → buyer is maker → SELL TAKER → delta = -qty
    # is_buyer_maker == False → buyer is taker → BUY TAKER  → delta = +qty
    df_all["delta"] = np.where(~df_all["is_buyer_maker"], df_all["qty"], -df_all["qty"])

    # Minute bucket
    df_all["minute"] = (df_all["timestamp"] // 60_000) * 60_000

    micro = compute_minute_microstructure(df_all)

    # Make continuous timeline [start_ms, end_ms) at 1m
    full_idx = np.arange(start_ms, end_ms, 60_000, dtype="int64")
    micro = micro.reindex(full_idx, fill_value=0.0)
    micro.index.name = "open_time_ms"

    logging.info("Microstructure minute count (bucket): %s", f"{len(micro):,}")

    return micro.reset_index(), df_all


# ============================================================
# MERGE WITH CANDLES
# ============================================================
def load_candles() -> pd.DataFrame:
    if not CANDLES_PATH.exists():
        raise FileNotFoundError(f"Candles file missing: {CANDLES_PATH}")

    candles = pd.read_csv(CANDLES_PATH)
    candles["open_time"] = pd.to_datetime(candles["open_time"], utc=True)
    candles["open_time_ms"] = (
        candles["open_time"].astype("int64") // 1_000_000
    ).astype("int64")
    candles = candles.drop_duplicates("open_time_ms").sort_values("open_time_ms")
    return candles


def _merge_with_candles(candles: pd.DataFrame, micro: pd.DataFrame) -> pd.DataFrame:
    """
    Merge per-minute micro stats with candle features to derive:
      wick_body_ratio, micro_volatility, micro_return, order_imbalance, no_trend_stat
    """
    merged = candles.merge(
        micro,
        on="open_time_ms",
        how="left",
    ).fillna(0)

    body = (merged.close - merged.open).abs()
    wick = (merged.high - merged.low) - body

    merged["wick_body_ratio"] = wick / (body + EPS)
    merged["micro_volatility"] = body / merged.close.shift(1).replace(0, np.nan).fillna(0)
    merged["micro_return"] = (merged.close - merged.open) / merged.open.replace(0, np.nan).fillna(1)
    merged["order_imbalance"] = (
        merged.taker_buy_base - (merged.volume - merged.taker_buy_base)
    ) / merged.volume.replace(0, np.nan).fillna(1)
    merged["no_trend_stat"] = (
        (merged.high - merged.low) /
        merged.close.replace(0, np.nan).fillna(1) < 0.0003
    ).astype(int)

    merged.fillna(0.0, inplace=True)
    return merged


def save_outputs(micro_full: pd.DataFrame, candles: pd.DataFrame) -> None:
    """
    Merge the raw microstructure stats with candles, derive higher-level
    micro features, and persist delta.parquet and microstructure.parquet.

    This is the **canonical** writer: MICRO_OUT will always contain only
    MICRO_FEATURE_COLS, no datetime/string columns, so that
    build_canonical_raw.load_microstructure can safely float32-cast
    its numeric columns.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    merged = _merge_with_candles(candles, micro_full)

    # Normalise any datetime columns to int ms to avoid downstream float32 cast errors
    for col in ["open_time", "close_time"]:
        if col in merged.columns:
            merged[col] = pd.to_datetime(merged[col], utc=True, errors="coerce")
            merged[col] = (merged[col].astype("int64") // 1_000_000).astype("int64")

    delta_only = merged[["open_time_ms", "micro_delta"]].rename(
        columns={"micro_delta": "delta"}
    )

    micro_only = merged[MICRO_FEATURE_COLS].copy()

    delta_only.to_parquet(DELTA_OUT, index=False)
    micro_only.to_parquet(MICRO_OUT, index=False)

    logging.info("Saved delta → %s (rows=%s)", DELTA_OUT, len(delta_only))
    logging.info("Saved micro features → %s (rows=%s)", MICRO_OUT, len(micro_only))


# ============================================================
# MAIN (FULL WINDOW)
# ============================================================
def main() -> None:
    setup_logging()
    logging.info("=== DELTA + MICROSTRUCTURE GENERATOR (BUCKETED) START ===")

    # ---- 1) Fixed target window based on LOOKBACK_DAYS ----
    if FORCE_END_MS is not None:
        wall_finish = datetime.fromtimestamp(
            FORCE_END_MS / 1000, tz=timezone.utc
        ).replace(second=0, microsecond=0)
    else:
        wall_finish = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    window_ms = int(round(LOOKBACK_DAYS * 24 * 60 * 60 * 1000))

    end_ms = int(wall_finish.timestamp() * 1000)  # end EXCLUSIVE
    start_ms = end_ms - window_ms

    start_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)

    logging.info(
        "Target window: %s -> %s (~%.3f days, end exclusive)",
        start_dt,
        end_dt,
        LOOKBACK_DAYS,
    )

    # If trades CSV already exists, start fresh for this run
    if TRADES_OUT.exists():
        TRADES_OUT.unlink()
        logging.info("Removed existing trades CSV: %s", TRADES_OUT)

    try:
        # ---- 2) Load candles once (fits in memory, unlike trades) ----
        candles = load_candles()

        # Restrict candles just to the window we care about for speed
        candles = candles[
            (candles["open_time_ms"] >= start_ms)
            & (candles["open_time_ms"] < end_ms)
        ].copy()

        # ---- 3) Process trades, optionally using daily dumps for the pre-today portion ----
        bucket_ms = 60 * 60 * 1000
        micro_parts: List[pd.DataFrame] = []
        first_trades = True

        today_start_ms = int(
            datetime.now(timezone.utc)
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .timestamp()
            * 1000
        )

        # Historical portion from daily dumps (if enabled and window crosses midnight)
        current_start = start_ms
        if USE_DAILY_DUMPS and start_ms < today_start_ms:
            hist_end_ms = min(end_ms, today_start_ms)
            logging.info(
                "Fetching historical trades from daily dumps for %s -> %s",
                datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc),
                datetime.fromtimestamp(hist_end_ms / 1000, tz=timezone.utc),
            )
            trades_hist = fetch_trades_from_dumps(start_ms, hist_end_ms)
            micro_hist, trades_hist = compute_micro_from_trades_df(trades_hist, start_ms, hist_end_ms)
            micro_parts.append(micro_hist)
            if not trades_hist.empty:
                append_trades_csv(trades_hist, header=first_trades)
                first_trades = False
            current_start = hist_end_ms

        # Live/API portion for the remaining window
        while current_start < end_ms:
            current_end = min(current_start + bucket_ms, end_ms)

            logging.info(
                "--- Bucket %s -> %s ---",
                datetime.fromtimestamp(current_start / 1000, tz=timezone.utc),
                datetime.fromtimestamp(current_end / 1000, tz=timezone.utc),
            )

            micro_hour, trades_hour = compute_micro_for_range(current_start, current_end)

            # Append per-minute micro for this bucket to list (tiny)
            micro_parts.append(micro_hour)

            # Append trades for this bucket to one CSV (streaming)
            if not trades_hour.empty:
                append_trades_csv(trades_hour, header=first_trades)
                first_trades = False

            current_start = current_end

        # ---- 4) Concatenate all per-minute microstructure (small) ----
        if micro_parts:
            micro_full = pd.concat(micro_parts, ignore_index=True)
        else:
            micro_full = build_empty_micro_frame(start_ms, end_ms)

        logging.info("Full microstructure frame has %s rows", len(micro_full))

        # ---- 5) Save final outputs (delta.parquet & microstructure.parquet) ----
        save_outputs(micro_full, candles)

        logging.info("=== DELTA + MICROSTRUCTURE GENERATOR (BUCKETED) FINISHED ===")

    except Exception as exc:  # noqa: BLE001
        logging.exception("Delta + microstructure pipeline failure: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()


# ============================================================
# INCREMENTAL TOP-UP
# ============================================================
def topup_to(end_ms: int, lookback_minutes: int) -> None:
    """
    Incrementally fetch only missing trades/microstructure up to end_ms (exclusive)
    and keep a rolling window of size lookback_minutes on disk.

    Key invariants:
      - MICRO_OUT always contains only MICRO_FEATURE_COLS (no datetime strings).
      - DELTA_OUT always contains [open_time_ms, delta].
      - We never refetch historical minutes we already have; we start from the
        last known open_time_ms + 60_000.
    """
    global LOOKBACK_DAYS, FORCE_END_MS  # single global declaration, at top of function

    setup_logging()

    window_ms = lookback_minutes * 60_000
    target_start_ms = end_ms - window_ms

    # ---- Load candles over the target window ----
    try:
        candles = load_candles()
    except FileNotFoundError:
        # Bootstrap full window via main()
        LOOKBACK_DAYS = lookback_minutes / 1440.0
        FORCE_END_MS = end_ms
        logging.info("Candles missing; bootstrapping full window via main()")
        main()
        return

    candles = candles[
        (candles["open_time_ms"] >= target_start_ms)
        & (candles["open_time_ms"] < end_ms)
    ].copy()

    # ---- Load existing micro + delta (if any) ----
    if MICRO_OUT.exists():
        existing_micro = pd.read_parquet(MICRO_OUT)
    else:
        existing_micro = pd.DataFrame(columns=MICRO_FEATURE_COLS)

    # ensure correct column order/types
    if not existing_micro.empty:
        # keep only expected columns if file ever drifted
        existing_micro = existing_micro[
            [c for c in MICRO_FEATURE_COLS if c in existing_micro.columns]
        ]
    else:
        existing_micro = pd.DataFrame(columns=MICRO_FEATURE_COLS)

    if existing_micro.empty or "open_time_ms" not in existing_micro.columns:
        existing_latest = target_start_ms - 60_000
    else:
        existing_latest = int(existing_micro["open_time_ms"].max())

    desired_latest = end_ms - 60_000

    logging.info(
        "Incremental delta+micro topup: existing_latest=%s, desired_latest=%s",
        existing_latest,
        desired_latest,
    )

    # Already current: just trim to rolling window and rebuild delta
    if existing_latest >= desired_latest:
        logging.info("Microstructure already up to date; trimming to rolling window.")
        trimmed = existing_micro[
            (existing_micro["open_time_ms"] >= target_start_ms)
            & (existing_micro["open_time_ms"] < end_ms)
        ].copy()
        trimmed = trimmed.sort_values("open_time_ms")
        trimmed.to_parquet(MICRO_OUT, index=False)

        delta_only = trimmed[["open_time_ms", "micro_delta"]].rename(
            columns={"micro_delta": "delta"}
        )
        delta_only.to_parquet(DELTA_OUT, index=False)
        return

    # Need new minutes
    missing_start_ms = max(existing_latest + 60_000, target_start_ms)
    if missing_start_ms >= end_ms:
        logging.info("No positive missing range after adjustment; nothing to do.")
        return

    logging.info(
        "Fetching missing microstructure for [%s, %s)",
        datetime.fromtimestamp(missing_start_ms / 1000, tz=timezone.utc),
        datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc),
    )

    try:
        micro_new_full, _ = compute_micro_for_range(missing_start_ms, end_ms)
    except Exception as exc:  # noqa: BLE001
        # On failure, bootstrap full window
        logging.exception(
            "Incremental micro fetch failed, falling back to full bootstrap: %s",
            exc,
        )
        LOOKBACK_DAYS = lookback_minutes / 1440.0
        FORCE_END_MS = end_ms
        main()
        return

    # Merge new micro with candles slice to derive feature block for new minutes
    candles_slice = candles[
        (candles["open_time_ms"] >= missing_start_ms)
        & (candles["open_time_ms"] < end_ms)
    ].copy()

    merged_new = _merge_with_candles(candles_slice, micro_new_full)
    micro_new_only = merged_new[MICRO_FEATURE_COLS].copy()

    # Combine existing + new microfeature rows
    combined = pd.concat([existing_micro, micro_new_only], ignore_index=True)

    combined = (
        combined
        .drop_duplicates(subset=["open_time_ms"], keep="last")
        .sort_values("open_time_ms")
    )

    # Trim to rolling window
    combined = combined[
        (combined["open_time_ms"] >= target_start_ms)
        & (combined["open_time_ms"] < end_ms)
    ].copy()

    combined.to_parquet(MICRO_OUT, index=False)
    logging.info(
        "Incremental microstructure written → %s (rows=%s)",
        MICRO_OUT,
        len(combined),
    )

    # Rebuild delta.parquet from combined micro features
    delta_only = combined[["open_time_ms", "micro_delta"]].rename(
        columns={"micro_delta": "delta"}
    )
    delta_only.to_parquet(DELTA_OUT, index=False)
    logging.info(
        "Incremental delta written → %s (rows=%s)",
        DELTA_OUT,
        len(delta_only),
    )
