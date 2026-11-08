import json
import websocket
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ---------------- CONFIG ---------------- #

SYMBOL = "btcusdt"
INTERVAL = "1m"
WS_URL = f"wss://fstream.binance.com/ws/{SYMBOL}@kline_{INTERVAL}"

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "live_candles"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- HELPERS ---------------- #

def log(msg: str):
    print(msg, flush=True)


# ---------------- WS CALLBACKS ---------------- #

def on_message(ws, message):
    data = json.loads(message)

    if data.get("e") != "kline":
        return

    k = data["k"]

    # We only care about CLOSED candles
    if not k["x"]:
        return

    candle = {
        "open_time_ms": k["t"],
        "open": k["o"],
        "high": k["h"],
        "low": k["l"],
        "close": k["c"],
        "volume": k["v"],
        "closeTime": k["T"],
        "quoteAssetVolume": k["q"],
        "numberOfTrades": k["n"],
        "takerBuyBase": k["V"],
        "takerBuyQuoteVolume": k["Q"],
    }

    close_dt = datetime.fromtimestamp(k["T"] / 1000, tz=timezone.utc)
    fname = OUTPUT_DIR / f"candles_1m_{close_dt.strftime('%Y%m%d_%H%M')}.parquet"

    df = pd.DataFrame([candle])
    df.to_parquet(fname, engine="pyarrow", index=False)

    log(f"Closed candle {close_dt} → {fname.name}")


def on_error(ws, error):
    log(f"WebSocket error: {error}")


def on_close(ws):
    log("WebSocket closed")


def on_open(ws):
    log("Connected to Binance 1m candle websocket")


# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    log("Starting live 1m candle stream")

    ws = websocket.WebSocketApp(
        WS_URL,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open,
    )

    ws.run_forever()