import requests
import pandas as pd

BASE = "https://fapi.binance.com"

def load_candles(symbol="BTCUSDT", interval="5m", limit=1500):
    """
    Loads OHLCV candles from Binance Futures API.
    Used for LIVE trading.
    """
    url = f"{BASE}/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    raw = requests.get(url).json()

    df = pd.DataFrame(raw, columns=[
        "open_time","open","high","low","close","volume","close_time",
        "quote_volume","taker_buy_vol","taker_buy_quote","trades","ignore"
    ])

    df = df[["open_time","open","high","low","close","volume","taker_buy_vol"]]
    df.columns = ["timestamp","open","high","low","close","volume","taker_buy_vol"]

    df["timestamp"] = df["timestamp"] // 1000  # ms → seconds
    df = df.astype({
        "open": float, "high": float, "low": float,
        "close": float, "volume": float, "taker_buy_vol": float
    })

    return df