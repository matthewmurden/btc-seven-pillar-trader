import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import requests
from requests.exceptions import HTTPError

# ============================================================
# CONFIG / PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
OUTPUT_PATH = RAW_DIR / "delta.csv"
TRADES_DIR = RAW_DIR / "trades"   # where we store per-run trades

SYMBOL = "BTCUSDT"
BASE_URL = "https://fapi.binance.com"
ENDPOINT = "/fapi/v1/aggTrades"

# How much history you want in the final window.
# For testing 1 hour:
LOOKBACK_DAYS = 1 / 24  # 1 hour

# Binance / rate control
LIMIT = 1000            # max trades per aggTrades request
REQUEST_SLEEP = 1.0     # pause after each successful request (seconds)
WINDOW_HOURS = 1        # size of each time window (1h chunks)
WINDOW_SLEEP = 1.0      # pause after finishing each window (seconds)

# Retry safety so we don't get stuck forever on 429s / glitches
MAX_429_RETRIES_PER_WINDOW = 6
MAX_GENERIC_RETRIES_PER_WINDOW = 6

EPS = 1e-12  # safety epsilon

# Global accumulator for per-trade data across the whole run
TRADES_CHUNKS: List[pd.DataFrame] = []


# ============================================================
# LOGGING
# ============================================================
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,   # no debug spam
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

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected aggTrades response: {data}")
    return data


# ============================================================
# STREAM TRADES OVER ARBITRARY RANGE
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

            except Exception as exc:
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

    logging.info("All windows complete.")


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
# SAVE TRADES
# ============================================================
def save_trades_csv(df_all: pd.DataFrame, start_ms: int, end_ms: int):
    """
    Save the raw trades used for this run as a single CSV under raw/trades.
    File name includes symbol + UTC start/end for readability.
    """
    TRADES_DIR.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)

    start_str = start_dt.strftime("%Y%m%d_%H%M")
    end_str = end_dt.strftime("%Y%m%d_%H%M")

    fname = f"trades_{SYMBOL}_{start_str}_{end_str}.csv"
    out_path = TRADES_DIR / fname

    df_to_save = df_all.copy()
    # Handy readable timestamp column
    df_to_save["timestamp_utc"] = pd.to_datetime(
        df_to_save["timestamp"], unit="ms", utc=True
    )

    # Put the useful columns first
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

    df_to_save.to_csv(out_path, index=False)
    logging.info(
        "Saved raw trades used for this run → %s (%s rows)",
        out_path,
        len(df_to_save),
    )


# ============================================================
# MICROSTRUCTURE FOR RANGE
# ============================================================
def compute_micro_for_range(start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Fetch trades in [start_ms, end_ms), compute per-trade delta and
    per-minute microstructure, and return a continuous 1-minute frame.

    Columns:
      - open_time_ms
      - delta
      - trade_count
      - buy_count
      - sell_count
      - entropy
      - delta_persistence
      - micro_clustering
      - alternation_index

    Also appends the per-trade data used into the global TRADES_CHUNKS
    list, so we can save a single CSV at the end of the run.
    """
    chunks = list(stream_trades_range(start_ms, end_ms))

    if not chunks:
        logging.warning(
            "No trades found for range %s → %s. Returning zero micro frame.",
            start_ms,
            end_ms,
        )
        return build_empty_micro_frame(start_ms, end_ms)

    df_all = pd.concat(chunks, ignore_index=True)

    # Binance semantics:
    # is_buyer_maker == True  → buyer is maker → SELL TAKER → delta = -qty
    # is_buyer_maker == False → buyer is taker → BUY TAKER  → delta = +qty
    df_all["delta"] = np.where(~df_all["is_buyer_maker"], df_all["qty"], -df_all["qty"])

    # Minute bucket
    df_all["minute"] = (df_all["timestamp"] // 60_000) * 60_000

    # Accumulate trades for this run (we'll save them once in main())
    TRADES_CHUNKS.append(df_all)

    micro = compute_minute_microstructure(df_all)

    # Make continuous timeline [start_ms, end_ms) at 1m
    full_idx = np.arange(start_ms, end_ms, 60_000, dtype="int64")
    micro = micro.reindex(full_idx, fill_value=0.0)
    micro.index.name = "open_time_ms"

    logging.info("Microstructure minute count: %s", f"{len(micro):,}")

    return micro.reset_index()


# ============================================================
# SAVE DELTA + MICRO
# ============================================================
def save_delta_csv(df: pd.DataFrame):
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    cols = [
        "open_time_ms",
        "delta",
        "trade_count",
        "buy_count",
        "sell_count",
        "entropy",
        "delta_persistence",
        "micro_clustering",
        "alternation_index",
    ]
    df = df[cols]

    logging.info("Saving delta + microstructure → %s", OUTPUT_PATH)
    df.to_csv(OUTPUT_PATH, index=False)
    logging.info("DONE — saved %s rows", f"{len(df):,}")


# ============================================================
# MAIN
# ============================================================
def main():
    setup_logging()
    logging.info("=== DELTA + MICROSTRUCTURE GENERATOR START ===")

    # ---- 1) Initial window based on LOOKBACK_DAYS at script START ----
    wall_start = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    initial_end_dt = wall_start                      # end EXCLUSIVE
    initial_start_dt = initial_end_dt - timedelta(days=LOOKBACK_DAYS)

    initial_start_ms = int(initial_start_dt.timestamp() * 1000)
    initial_end_ms = int(initial_end_dt.timestamp() * 1000)

    logging.info(
        "Initial target window: %s -> %s (~%.3f days, end exclusive)",
        initial_start_dt,
        initial_end_dt,
        LOOKBACK_DAYS,
    )

    try:
        # ---- 2) Fetch & compute microstructure for main window ----
        agg_main = compute_micro_for_range(initial_start_ms, initial_end_ms)

        # ---- 3) Compute gap between script start and now, and top-up ----
        wall_finish = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        final_end_ms = int(wall_finish.timestamp() * 1000)   # end EXCLUSIVE

        logging.info("Initial computation finished at %s", wall_finish)

        if final_end_ms > initial_end_ms:
            # There is a gap between the initial end and "now"
            gap_minutes = int((final_end_ms - initial_end_ms) // 60_000)
            logging.info(
                "Top-up gap detected: %d minutes between %s and %s",
                gap_minutes,
                datetime.fromtimestamp(initial_end_ms / 1000, tz=timezone.utc),
                wall_finish,
            )

            gap_start_ms = initial_end_ms           # start from old end (no overlap)
            gap_end_ms = final_end_ms               # end EXCLUSIVE

            agg_gap = compute_micro_for_range(gap_start_ms, gap_end_ms)

            # Merge main + gap, favouring gap data on overlaps
            main_idxed = agg_main.set_index("open_time_ms")
            gap_idxed = agg_gap.set_index("open_time_ms")
            combined = pd.concat([main_idxed, gap_idxed])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()

            # Enforce EXACT LOOKBACK_DAYS ending at wall_finish (end EXCLUSIVE)
            window_ms = int(LOOKBACK_DAYS * 24 * 60 * 60 * 1000)
            target_start_ms = final_end_ms - window_ms

            combined = combined[combined.index >= target_start_ms]
            combined = combined[combined.index < final_end_ms]

            agg_final = combined.reset_index()
        else:
            logging.info("No significant gap; using initial computation only.")
            agg_final = agg_main

        logging.info(
            "Final window: %s rows from %s to %s (end exclusive)",
            len(agg_final),
            datetime.fromtimestamp(
                int(agg_final["open_time_ms"].iloc[0]) / 1000, tz=timezone.utc
            ),
            datetime.fromtimestamp(final_end_ms / 1000, tz=timezone.utc),
        )

        # ---- 4) Save all raw trades used in this run as a single CSV ----
        if TRADES_CHUNKS:
            trades_all = pd.concat(TRADES_CHUNKS, ignore_index=True)
            trades_all.sort_values("timestamp", inplace=True)

            start_ms_final = int(agg_final["open_time_ms"].min())
            end_ms_final = int(agg_final["open_time_ms"].max()) + 60_000  # end exclusive

            save_trades_csv(trades_all, start_ms_final, end_ms_final)
        else:
            logging.warning("No trades collected; skipping trades CSV save.")

        # ---- 5) Save final, exact-length minute window ----
        save_delta_csv(agg_final)
        logging.info("=== DELTA + MICROSTRUCTURE GENERATOR FINISHED ===")

    except Exception as exc:
        logging.exception("Delta + microstructure pipeline failure: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()