import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import time
from pathlib import Path

# =========================
# CONFIG
# =========================
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
BASE_URL = "https://fapi.binance.com"
ENDPOINT = "/fapi/v1/klines"
DATA_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = DATA_DIR / "raw" / "candles"

LIMIT = 1500  # Binance max rows per request

LOOKBACK_DAYS = 1  # <-- 14-day rolling window


def ensure_output_dir(path: Path):
    """Make sure the target directory exists and is writable before downloads."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_check"
        probe.touch()
        probe.unlink()
        print(f"Output directory ready: {path}")
    except Exception as exc:
        raise RuntimeError(f"Output directory not writable: {path}") from exc


# =========================
# API CALL FUNCTION
# =========================
def get_klines(symbol, interval, start_ms, end_ms, limit=LIMIT):
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
def download_last_14_days():
    ensure_output_dir(OUTPUT_DIR)

    # Now (UTC) and 14 days ago
    end_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=LOOKBACK_DAYS)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    cursor = start_ms

    rows = []

    print(f"\nDownloading {SYMBOL} {INTERVAL} klines")
    print(f"From {start_dt} to {end_dt} (~{LOOKBACK_DAYS} days)")

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

        except Exception as e:
            print(f"Error: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            continue

        # Progress indicator every ~100k rows
        if len(rows) % 100_000 < LIMIT:
            now = datetime.fromtimestamp(cursor / 1000, tz=timezone.utc)
            print(f"Downloaded {len(rows):,} candles up to {now}")

    # =========================
    # SAVE TO DATAFRAME
    # =========================
    df = pd.DataFrame(rows)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # Final duplicate prevention
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time")

    # Auto name with actual start/end date
    outfile = OUTPUT_DIR / f"{SYMBOL}_{INTERVAL}_{start_dt.date()}_{end_dt.date()}.csv"
    df.to_csv(outfile, index=False)

    print("\n==============================")
    print("DOWNLOAD COMPLETE")
    print(f"Total 1-minute candles: {len(df):,}")
    print(f"Saved to: {outfile}")
    print("==============================\n")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    download_last_14_days()