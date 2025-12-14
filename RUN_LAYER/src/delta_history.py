# RUN_LAYER/src/delta_history.py

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

# Retry safety so we don't get stuck forever on 429s
MAX_429_RETRIES_PER_WINDOW = 6      # after this, skip the rest of the window
MAX_GENERIC_RETRIES_PER_WINDOW = 6  # same for other errors

EPS = 1e-12  # kept for compatibility if you extend later


# ============================================================
# LOGGING
# ============================================================
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,   # keep at INFO, no debug spam
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ============================================================
# API CALL
# ============================================================
def get_agg_trades(symbol: str, start_ms: int, end_ms: int, limit: int = LIMIT) -> List[list]:
    """
    Fetch a batch of aggregated trades from Binance Futures.

    Response (aggTrades):
    [
      aggTradeId (a),
      price      (p),
      qty        (q),
      firstTradeId (f),
      lastTradeId  (l),
      timestamp    (T, ms),
      isBuyerMaker (m),
      isBestMatch  (not used here)
    ]
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
    Streams aggTrades between [start_ms, end_ms] in WINDOW_HOURS chunks.

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

                # trades is a list of dicts with keys: a, p, q, f, l, T, m, (bestMatch)
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
# AGGREGATE DELTA PER MINUTE
# ============================================================
def aggregate_delta(
    chunks: Iterable[pd.DataFrame],
    expected_start_ms: Optional[int] = None,
    expected_end_ms: Optional[int] = None,
) -> pd.DataFrame:
    """
    Builds the 1-minute delta stream from streamed trades.

    Output columns:
      - open_time_ms
      - delta

    expected_start_ms / expected_end_ms are treated as:
      [expected_start_ms, expected_end_ms)  (end EXCLUSIVE)

    If no trades are found at all and expected_start_ms/expected_end_ms are given,
    returns a zero-delta frame over that range instead of raising.
    """
    agg = None

    for chunk in chunks:
        # Signed delta: +qty if buyer is taker, -qty if seller is taker
        chunk["delta_signed"] = np.where(
            ~chunk["is_buyer_maker"], chunk["qty"], -chunk["qty"]
        )

        # Minute bucket (ms)
        chunk["minute"] = (chunk["timestamp"] // 60_000) * 60_000

        # Aggregate within this chunk
        chunk_agg = (
            chunk.groupby("minute")["delta_signed"]
            .sum()
            .astype("float64")
            .to_frame("delta")
        )

        if agg is None:
            agg = chunk_agg
        else:
            common_index = agg.index.union(chunk_agg.index)
            agg = agg.reindex(common_index, fill_value=0.0)
            chunk_agg = chunk_agg.reindex(common_index, fill_value=0.0)
            agg["delta"] = agg["delta"] + chunk_agg["delta"]

    # If no trades at all, fall back to zero-delta frame for expected window
    if agg is None:
        if expected_start_ms is not None and expected_end_ms is not None:
            logging.warning(
                "No trade data fetched; building zero-delta frame "
                "from %s to %s (end exclusive).",
                expected_start_ms,
                expected_end_ms,
            )
            full_idx = np.arange(
                expected_start_ms,
                expected_end_ms,          # end EXCLUSIVE
                60_000,
                dtype="int64",
            )
            df_zero = pd.DataFrame(
                {
                    "open_time_ms": full_idx,
                    "delta": np.zeros_like(full_idx, dtype="float64"),
                }
            )
            return df_zero
        else:
            raise ValueError("No trade data fetched for delta computation.")

    # Ensure continuous minute timeline
    agg = agg.sort_index()

    if expected_start_ms is None:
        start_idx = agg.index.min()
    else:
        start_idx = expected_start_ms

    if expected_end_ms is None:
        # Make end exclusive by adding one minute past the last index
        end_idx = agg.index.max() + 60_000
    else:
        end_idx = expected_end_ms

    full_idx = np.arange(
        start_idx,
        end_idx,            # end EXCLUSIVE
        60_000,
        dtype="int64",
    )
    agg = agg.reindex(full_idx, fill_value=0.0)
    agg.index.name = "open_time_ms"

    logging.info("Delta minute count: %s", f"{len(agg):,}")
    return agg.reset_index()


# ============================================================
# SAVE
# ============================================================
def save_delta_csv(df: pd.DataFrame):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = df[["open_time_ms", "delta"]]

    logging.info("Saving delta stream → %s", OUTPUT_PATH)
    df.to_csv(OUTPUT_PATH, index=False)
    logging.info("DONE — saved %s rows", f"{len(df):,}")


# ============================================================
# MAIN
# ============================================================
def main():
    setup_logging()
    logging.info("=== DELTA GENERATOR START ===")

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
        # ---- 2) Fetch & aggregate main window ----
        chunks_main = stream_trades_range(initial_start_ms, initial_end_ms)
        agg_main = aggregate_delta(
            chunks_main,
            expected_start_ms=initial_start_ms,
            expected_end_ms=initial_end_ms,
        )

        # ---- 3) Compute gap between script start and now, and top-up ----
        wall_finish = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        final_end_ms = int(wall_finish.timestamp() * 1000)   # end EXCLUSIVE

        logging.info("Initial aggregation finished at %s", wall_finish)

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

            chunks_gap = stream_trades_range(gap_start_ms, gap_end_ms)
            agg_gap = aggregate_delta(
                chunks_gap,
                expected_start_ms=gap_start_ms,
                expected_end_ms=gap_end_ms,
            )

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
            logging.info("No significant gap; using initial aggregation only.")
            agg_final = agg_main

        logging.info(
            "Final delta window: %s rows from %s to %s (end exclusive)",
            len(agg_final),
            datetime.fromtimestamp(
                int(agg_final["open_time_ms"].iloc[0]) / 1000, tz=timezone.utc
            ),
            datetime.fromtimestamp(
                initial_end_ms / 1000, tz=timezone.utc
            )
            if final_end_ms == initial_end_ms
            else datetime.fromtimestamp(
                final_end_ms / 1000, tz=timezone.utc
            ),
        )

        # ---- 4) Save final, exact-length window ----
        save_delta_csv(agg_final)
        logging.info("=== DELTA GENERATOR FINISHED ===")

    except Exception as exc:
        logging.exception("Delta pipeline failure: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()