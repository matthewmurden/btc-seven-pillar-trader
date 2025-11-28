from .api_candles import load_candles

def load_htf(symbol="BTCUSDT"):
    """
    Loads 15m and 1h candles for HTF trend alignment.
    """
    df_15m = load_candles(symbol=symbol, interval="15m")
    df_1h  = load_candles(symbol=symbol, interval="1h")

    return {"15m": df_15m, "1h": df_1h}