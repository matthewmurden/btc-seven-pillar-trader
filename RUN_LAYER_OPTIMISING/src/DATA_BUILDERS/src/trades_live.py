import json
import websocket
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

import pandas as pd


# ---------------- CONFIG ---------------- #

SYMBOL = "btcusdt"
WS_URL = f"wss://fstream.binance.com/ws/{SYMBOL}@aggTrade"

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "live_aggtrades"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- HELPERS ---------------- #

def log(msg: str):
    print(msg, flush=True)


def minute_bucket(ts_ms: int) -> int:
    """Floor timestamp (ms) to minute bucket."""
    return (ts_ms // 60_000) * 60_000


# ---------------- STATE ---------------- #

buffer = defaultdict(list)
current_minute = None


# ---------------- WS CALLBACKS ---------------- #

def on_message(ws, message):
    global current_minute

    data = json.loads(message)

    if data.get("e") != "aggTrade":
        return

    ts = data["T"]
    minute = minute_bucket(ts)

    # First message
    if current_minute is None:
        current_minute = minute

    # Minute rolled → flush closed minute
    if minute > current_minute:
        flush_minute(current_minute)
        del buffer[current_minute]
        current_minute = minute

    # ---- NORMALISE TO FULL WORD SCHEMA ----
    buffer[minute].append({
        "aggTradeId": data["a"],
        "price": data["p"],
        "quantity": data["q"],
        "firstTradeId": data["f"],
        "lastTradeId": data["l"],
        "timestamp": data["T"],
        "isBuyerMaker": data["m"],
    })


def flush_minute(minute_ts: int):
    trades = buffer.get(minute_ts, [])

    if not trades:
        return

    dt = datetime.fromtimestamp(minute_ts / 1000, tz=timezone.utc)
    fname = OUTPUT_DIR / f"aggTrades_{dt.strftime('%Y%m%d_%H%M')}.parquet"

    df = pd.DataFrame(trades)

    df.to_parquet(
        fname,
        engine="pyarrow",
        index=False,
    )

    log(f"Closed minute {dt} — {len(df)} aggTrades → {fname.name}")


def on_error(ws, error):
    log(f"WebSocket error: {error}")


def on_close(ws):
    log("WebSocket closed")


def on_open(ws):
    log("Connected to Binance aggTrade websocket")


# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    log("Starting aggTrade websocket (buffering by minute)")

    ws = websocket.WebSocketApp(
        WS_URL,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open,
    )

    ws.run_forever()