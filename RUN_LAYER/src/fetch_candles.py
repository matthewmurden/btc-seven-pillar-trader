import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import requests

# =========================
# CONFIG
# =========================
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
BASE_URL = "https://fapi.binance.com"
ENDPOINT = "/fapi/v1/klines"

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = RAW_DIR

LIMIT = 1500  # Binance max rows per request

# Can be fractional (e.g. 30 minutes = 30/1440)
LOOKBACK_DAYS: float = 1.0
# Optional exclusive end timestamp override (ms)
FORCE_END_MS: int | None = None


def ensure_output_dir(path: Path) -> None:
    """Make sure the target directory exists and is writable before downloads."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_check"
        probe.touch()
        probe.unlink()
        print(f"Output directory ready: {path}")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Output directory not writable: {path}") from exc


# =========================
# API CALL FUNCTION
# =========================
def get_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = LIMIT,
) -> List[List[Any]]:
    """Fetch klines safely from Binance Futures."""
    url = f"{BASE_URL}{ENDPOINT}"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(start_ms),
        "endTime": int(end_ms),
        "limit": limit,
    }

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


# =========================
# MAIN DOWNLOAD FUNCTION
# =========================
def download_last_14_days() -> None:
    """
    Download a continuous window of 1m candles ending at FORCE_END_MS (if set)
    or 'now', with total length LOOKBACK_DAYS days, and write BTC_CANDLES.csv.
    """
    ensure_output_dir(OUTPUT_DIR)

    # Now (UTC) and start based on lookback; end is exclusive
    if FORCE_END_MS is not None:
        end_dt = datetime.fromtimestamp(
            FORCE_END_MS / 1000, tz=timezone.utc
        ).replace(second=0, microsecond=0)
    else:
        end_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    window_minutes = int(LOOKBACK_DAYS * 24 * 60)
    start_dt = end_dt - timedelta(minutes=window_minutes)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    cursor = start_ms

    rows: List[Dict[str, Any]] = []

    print(f"\nDownloading {SYMBOL} {INTERVAL} klines")
    print(f"From {start_dt} to {end_dt} (~{window_minutes} minutes)")

    while cursor < end_ms:
        try:
            klines = get_klines(SYMBOL, INTERVAL, cursor, end_ms)

            if not klines:
                # No data → skip forward 1 minute
                cursor += 60_000
                continue

            for k in klines:
                rows.append(
                    {
                        "open_time": k[0],
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                        "close_time": k[6],
                        "quote_volume": float(k[7]),
                        "trades": int(k[8]),
                        "taker_buy_base": float(k[9]),
                        "taker_buy_quote": float(k[10]),
                    }
                )

            # Advance cursor by EXACTLY one candle from the last batch
            last_open = klines[-1][0]  # open_time in ms
            cursor = last_open + 60_000  # next candle

            # Rate limit protection
            time.sleep(0.2)

        except Exception as e:  # noqa: BLE001
            print(f"Error: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            continue

        # Progress indicator every ~100k rows
        if rows and len(rows) % 100_000 < LIMIT:
            now = datetime.fromtimestamp(cursor / 1000, tz=timezone.utc)
            print(f"Downloaded {len(rows):,} candles up to {now}")

    # =========================
    # SAVE TO DATAFRAME
    # =========================
    df = pd.DataFrame(rows)
    if not df.empty:
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        # Final duplicate prevention
        df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time")

        # Enforce exact window size (end exclusive), keep last N minutes
        df = df.tail(window_minutes)

    # Fixed filename
    outfile = OUTPUT_DIR / "BTC_CANDLES.csv"
    df.to_csv(outfile, index=False)

    print("\n==============================")
    print("DOWNLOAD COMPLETE")
    print(f"Total 1-minute candles: {len(df):,}")
    print(f"Saved to: {outfile}")
    print("==============================\n")


# Entry point for orchestrator
def main() -> None:
    download_last_14_days()


# =========================
# RUN (manual)
# =========================
if __name__ == "__main__":
    download_last_14_days()


# =========================
# INCREMENTAL TOP-UP
# =========================
def topup_to(end_ms: int, lookback_minutes: int) -> None:
    """
    Incrementally fetch only missing 1m candles up to end_ms (exclusive),
    and keep a rolling window of size lookback_minutes on disk.

    Behaviour:
      - If BTC_CANDLES.csv is missing or broken → bootstrap full window via
        download_last_14_days() using LOOKBACK_DAYS and FORCE_END_MS.
      - Otherwise, append only the missing minutes, then trim to
        [end_ms - lookback_minutes, end_ms).
    """
    global LOOKBACK_DAYS, FORCE_END_MS  # must be first statement after docstring

    ensure_output_dir(OUTPUT_DIR)

    window_ms = lookback_minutes * 60_000
    target_start_ms = end_ms - window_ms

    outfile = OUTPUT_DIR / "BTC_CANDLES.csv"

    # 1) If no file yet, or clearly broken → bootstrap full window
    if not outfile.exists():
        LOOKBACK_DAYS = lookback_minutes / 1440.0
        FORCE_END_MS = end_ms
        download_last_14_days()
        return

    df = pd.read_csv(outfile)
    if df.empty or "open_time" not in df.columns:
        LOOKBACK_DAYS = lookback_minutes / 1440.0
        FORCE_END_MS = end_ms
        download_last_14_days()
        return

    # 2) Compute existing open_time_ms and sort
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df["open_time_ms"] = (
        df["open_time"].astype("int64") // 1_000_000
    ).astype("int64")
    df = df.sort_values("open_time_ms")

    existing_latest = int(df["open_time_ms"].max())
    desired_latest = end_ms - 60_000  # last full candle before end_ms

    # 3) If already up to date, just trim to rolling window and overwrite
    if existing_latest >= desired_latest:
        df = df[
            (df["open_time_ms"] >= target_start_ms)
            & (df["open_time_ms"] < end_ms)
        ].copy()
        df = df.drop(columns=["open_time_ms"])
        df.to_csv(outfile, index=False)
        return

    # 4) Otherwise, fetch only the missing tail [missing_start_ms, end_ms)
    missing_start_ms = max(existing_latest + 60_000, target_start_ms)
    rows: List[Dict[str, Any]] = []
    cursor = missing_start_ms

    while cursor < end_ms:
        try:
            klines = get_klines(SYMBOL, INTERVAL, cursor, end_ms)
        except Exception as e:  # noqa: BLE001
            print(f"Topup error: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            continue

        if not klines:
            cursor += 60_000
            continue

        for k in klines:
            rows.append(
                {
                    "open_time": k[0],
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "close_time": k[6],
                    "quote_volume": float(k[7]),
                    "trades": int(k[8]),
                    "taker_buy_base": float(k[9]),
                    "taker_buy_quote": float(k[10]),
                }
            )

        last_open = klines[-1][0]
        cursor = last_open + 60_000
        time.sleep(0.2)

    # 5) Append new rows, re-deduplicate, enforce rolling window
    if rows:
        new_df = pd.DataFrame(rows)
        new_df["open_time"] = pd.to_datetime(new_df["open_time"], unit="ms", utc=True)
        new_df["open_time_ms"] = (
            new_df["open_time"].astype("int64") // 1_000_000
        ).astype("int64")

        df = pd.concat([df, new_df], ignore_index=True)

    df = df.drop_duplicates(subset=["open_time_ms"]).sort_values("open_time_ms")
    df = df[
        (df["open_time_ms"] >= target_start_ms)
        & (df["open_time_ms"] < end_ms)
    ].copy()

    # Drop helper column before saving
    df = df.drop(columns=["open_time_ms"])
    df.to_csv(outfile, index=False)