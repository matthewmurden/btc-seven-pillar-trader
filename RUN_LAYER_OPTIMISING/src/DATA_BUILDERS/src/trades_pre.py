import json
import time
import zipfile
import io
import shutil
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Iterator

import requests
import pandas as pd
import duckdb


# ---------------- CONFIG ---------------- #

def load_config() -> dict:
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, "r") as f:
        return json.load(f)


def log(msg: str):
    print(msg, flush=True)


# ---------------- REST AGG TRADES ---------------- #

REST_TO_VISION = {
    "a": "aggTradeId",
    "p": "price",
    "q": "quantity",
    "f": "firstTradeId",
    "l": "lastTradeId",
    "T": "timestamp",
    "m": "isBuyerMaker",
}


def get_binance_agg_trades(
    symbol: str,
    start_time: int,
    end_time: int,
    chunk_size: int = 1000,
    min_sleep: float = 0.1,   # conservative to avoid 429s
) -> Iterator[pd.DataFrame]:
    base_url = "https://fapi.binance.com"
    endpoint = "/fapi/v1/aggTrades"

    from_id = None
    backoff = 1.0

    while True:
        params = {
            "symbol": symbol,
            "startTime": start_time,
            "endTime": end_time,
            "limit": chunk_size,
        }

        if from_id is not None:
            params["fromId"] = from_id

        try:
            r = requests.get(base_url + endpoint, params=params, timeout=10)

            if r.status_code == 429:
                log(f"REST 429 — backing off {backoff:.1f}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue

            r.raise_for_status()
            backoff = 1.0

        except requests.RequestException as e:
            log(f"REST error: {e}, retrying...")
            time.sleep(5)
            continue

        data = r.json()
        if not data:
            break

        df = pd.DataFrame(data)

        # ---- NORMALISE TO VISION SCHEMA (ONCE, HERE) ----
        df.rename(columns=REST_TO_VISION, inplace=True)

        yield df

        if int(df["timestamp"].iloc[-1]) >= end_time:
            break

        from_id = int(data[-1]["a"]) + 1
        time.sleep(min_sleep)


# ---------------- VISION AGG TRADES ---------------- #

def download_historical_data(
    symbol: str,
    date: datetime,
    lookback_timestamp_ms: int,
    chunk_size: int = 500_000,
) -> Iterator[pd.DataFrame]:
    date_str = date.strftime("%Y-%m-%d")
    url = (
        "https://data.binance.vision/data/futures/um/daily/aggTrades/"
        f"{symbol}/{symbol}-aggTrades-{date_str}.zip"
    )

    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        log(f"Vision download failed ({date_str}): {e}")
        return

    buf = io.BytesIO(r.content)

    with zipfile.ZipFile(buf) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            reader = pd.read_csv(
                f,
                names=[
                    "aggTradeId",
                    "price",
                    "quantity",
                    "firstTradeId",
                    "lastTradeId",
                    "timestamp",
                    "isBuyerMaker",
                ],
                chunksize=chunk_size,
                low_memory=False,
            )

            for chunk in reader:
                ts = pd.to_numeric(chunk["timestamp"], errors="coerce")

                # µs vs ms
                if ts.max() > 10**14:
                    chunk["timestamp"] = ts // 1000
                else:
                    chunk["timestamp"] = ts

                mask = chunk["timestamp"] >= lookback_timestamp_ms
                if mask.any():
                    yield chunk.loc[mask].copy()


# ---------------- MAIN PIPELINE ---------------- #

def main(current_time: Optional[int] = None) -> Optional[Path]:
    config = load_config()
    lookback_days = float(config.get("LOOKBACK_TIME", 1))
    symbol = "BTCUSDT"

    if current_time is None:
        current_time = int(datetime.now(timezone.utc).timestamp() * 1000)

    current_dt = datetime.fromtimestamp(current_time / 1000, tz=timezone.utc)
    current_dt = current_dt.replace(second=0, microsecond=0)
    current_time = int(current_dt.timestamp() * 1000)

    lookback_dt = current_dt - timedelta(days=lookback_days)
    lookback_ts = int(lookback_dt.timestamp() * 1000)

    log(f"Current minute: {current_dt}")
    log(f"Lookback time:  {lookback_dt}")

    base_dir = Path(__file__).parent.parent.parent / "data_builders" / "data"
    tmp_dir = base_dir / "_tmp_aggTrades"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    part = 0
    total = 0

    # -------- Historical (Vision) -------- #
    if lookback_dt.date() < current_dt.date():
        log(f"Downloading from Binance Vision ({lookback_dt.date()} → {current_dt.date()})")

        for df in download_historical_data(symbol, lookback_dt, lookback_ts):
            out = tmp_dir / f"part-{part:05d}.parquet"
            df.to_parquet(out, engine="pyarrow", index=False)
            total += len(df)
            part += 1

        log("Binance Vision download complete")

    # -------- Today (REST) -------- #
    start_today = max(
        lookback_ts,
        int(datetime.combine(
            current_dt.date(),
            datetime.min.time(),
            tzinfo=timezone.utc
        ).timestamp() * 1000),
    )

    if start_today < current_time:
        log(
            f"Downloading from Binance REST API "
            f"({datetime.fromtimestamp(start_today/1000, tz=timezone.utc)} → {current_dt})"
        )

        for df in get_binance_agg_trades(symbol, start_today, current_time):
            out = tmp_dir / f"part-{part:05d}.parquet"
            df.to_parquet(out, engine="pyarrow", index=False)
            total += len(df)
            part += 1

        log("REST API download complete")

    parts = list(tmp_dir.glob("*.parquet"))
    if not parts:
        log("No aggTrades found for lookback window — nothing to save")
        shutil.rmtree(tmp_dir)
        return None

    # -------- FINAL SINGLE PARQUET -------- #
    final_file = base_dir / f"aggTrades_{current_dt.strftime('%Y%m%d_%H%M%S')}.parquet"

    duckdb.sql(f"""
        COPY (
            SELECT * FROM read_parquet('{tmp_dir}/*.parquet')
        )
        TO '{final_file}'
        (FORMAT 'parquet');
    """)

    shutil.rmtree(tmp_dir)

    log("All downloads complete")
    log(f"Saved {total} trades → {final_file}")

    return final_file


if __name__ == "__main__":
    main()