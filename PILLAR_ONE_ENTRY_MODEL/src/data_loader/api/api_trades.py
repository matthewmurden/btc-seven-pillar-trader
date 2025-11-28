import requests
import pandas as pd

BASE = "https://fapi.binance.com"

def load_trades(symbol="BTCUSDT", limit=1000):
    """
    Loads LIVE aggTrades for real-time delta / orderflow.
    """
    url = f"{BASE}/fapi/v1/aggTrades?symbol={symbol}&limit={limit}"
    raw = requests.get(url).json()

    df = pd.DataFrame(raw)
    df["timestamp"] = df["T"] // 1000  # trade time
    df["price"] = df["p"].astype(float)
    df["qty"] = df["q"].astype(float)
    df["is_buyer"] = df["m"].apply(lambda x: 0 if x else 1)

    return df[["timestamp","price","qty","is_buyer"]]