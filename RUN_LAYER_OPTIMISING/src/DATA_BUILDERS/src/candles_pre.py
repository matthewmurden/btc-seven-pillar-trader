import json
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Iterator

import requests
import pandas as pd


# ---------------- CONFIG ---------------- #

def load_config() -> dict:
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, "r") as f:
        return json.load(f)


def log(msg: str):
    print(msg, flush=True)


# ---------------- REST KLINES ---------------- #

def get_klines(
    symbol: str,
    interval: str,
    start_time: int,
    end_time: int,
    limit: int = 1000,
    min_sleep: float = 0.1,
) -> Iterator[pd.DataFrame]:
    base_url = "https://fapi.binance.com"
    endpoint = "/fapi/v1/klines"

    while start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit,
        }

        try:
            r = requests.get(base_url + endpoint, params=params, timeout=10)
            r.raise_for_status()
        except requests.RequestException as e:
            log(f"REST error: {e}, retrying...")
            time.sleep(5)
            continue

        data = r.json()
        if not data:
            break

        df = pd.DataFrame(
            data,
            columns=[
                "open_time_ms",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "closeTime",
                "quoteAssetVolume",
                "numberOfTrades",
                "takerBuyBase",
                "takerBuyQuoteVolume",
                "_ignore",
            ],
        )

        df.drop(columns="_ignore", inplace=True)
        yield df

        start_time = int(df["closeTime"].iloc[-1]) + 1
        time.sleep(min_sleep)


# ---------------- MAIN ---------------- #

def main(current_time: Optional[int] = None) -> Optional[Path]:
    config = load_config()
    lookback_days = float(config.get("LOOKBACK_TIME", 1))

    symbol = "BTCUSDT"
    interval = "1m"

    # ---- Accept time from CLI (ISO) ----
    if current_time is None and len(sys.argv) > 1:
        current_time = int(
            datetime.fromisoformat(sys.argv[1])
            .replace(tzinfo=timezone.utc)
            .timestamp() * 1000
        )

    if current_time is None:
        current_time = int(datetime.now(timezone.utc).timestamp() * 1000)

    # ---- Align to latest CLOSED minute ----
    current_dt = datetime.fromtimestamp(current_time / 1000, tz=timezone.utc)
    current_dt = current_dt.replace(second=0, microsecond=0)
    end_time = int(current_dt.timestamp() * 1000)

    lookback_dt = current_dt - timedelta(days=lookback_days)
    start_time = int(lookback_dt.timestamp() * 1000)

    log(f"Fetching 1m candles for {symbol}")
    log(f"From {lookback_dt} → {current_dt}")

    base_dir = Path(__file__).parent.parent.parent / "data_builders" / "data"
    base_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    total = 0

    for df in get_klines(symbol, interval, start_time, end_time):
        parts.append(df)
        total += len(df)

    if not parts:
        log("No candles fetched")
        return None

    final_df = pd.concat(parts, ignore_index=True)

    out_file = base_dir / f"candles_1m_{current_dt.strftime('%Y%m%d_%H%M%S')}.parquet"
    final_df.to_parquet(out_file, engine="pyarrow", index=False)

    log(f"Saved {total} candles → {out_file}")
    return out_file


if __name__ == "__main__":
    main()