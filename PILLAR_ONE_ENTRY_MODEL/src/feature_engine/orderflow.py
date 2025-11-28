import numpy as np

def compute_orderflow_signals(df):
    # -----------------------------------------------------
    # 1. True delta from aggregated trades
    # -----------------------------------------------------
    # Your aggregated trades already compute delta,
    # so only recompute if missing
    if "delta" not in df.columns:
        df["delta"] = df["taker_buy_qty"] - df["taker_sell_qty"]

    # -----------------------------------------------------
    # 2. Order flow imbalance (pro-level)
    # delta / total volume, smoothed
    # -----------------------------------------------------
    total_vol = df["taker_buy_qty"] + df["taker_sell_qty"]
    df["of_raw_imbalance"] = df["delta"] / total_vol.replace(0, np.nan)

    # Smooth the signal
    df["of_imbalance"] = df["of_raw_imbalance"].rolling(5).mean()

    # -----------------------------------------------------
    # 3. Liquidity sweep detection (institutional version)
    # -----------------------------------------------------
    df["upper_wick"] = df["high"] - df[["open","close"]].max(axis=1)
    df["lower_wick"] = df[["open","close"]].min(axis=1) - df["low"]

    wick_avg_high = df["upper_wick"].rolling(30).mean()
    wick_avg_low  = df["lower_wick"].rolling(30).mean()

    delta_std = df["delta"].rolling(30).std()

    # Sweep UP = large lower wick + large negative delta spike previously + reclaim
    df["sweep_up"] = (
        (df["lower_wick"] > wick_avg_low * 2) &
        (df["delta"] > delta_std) &
        (df["close"] > df["open"])  # candle reclaimed
    ).astype(int)

    # Sweep DOWN = large upper wick + large positive delta spike previously + reject
    df["sweep_down"] = (
        (df["upper_wick"] > wick_avg_high * 2) &
        (df["delta"] < -delta_std) &
        (df["close"] < df["open"])  # candle rejected
    ).astype(int)

    # Combined sweep score for the model
    df["sweep_signal"] = df["sweep_up"] - df["sweep_down"]

    return df