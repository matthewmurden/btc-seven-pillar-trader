import pandas as pd

def aggregate_trade_chunk(chunk):
    """
    Aggregate a single chunk of trades to 5m bars.
    """
    df = chunk.copy()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.set_index("datetime")

    agg = df.resample("5T").agg({
        "qty": "sum",
        "is_buyer": ["sum", "count"]
    })

    agg.columns = ["total_qty", "taker_buy_qty", "trade_count"]
    agg["taker_sell_qty"] = agg["total_qty"] - agg["taker_buy_qty"]
    agg["delta"] = agg["taker_buy_qty"] - agg["taker_sell_qty"]
    agg["timestamp"] = agg.index.astype(int) // 10**9

    return agg[["timestamp","taker_buy_qty","taker_sell_qty","delta"]]