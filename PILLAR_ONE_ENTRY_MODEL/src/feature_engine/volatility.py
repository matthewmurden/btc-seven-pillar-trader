import pandas as pd
import numpy as np

def compute_volatility_signals(df):

    # -------------------------------------------
    # 8 — ATR Expansion / Compression (True ATR)
    # -------------------------------------------
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # Change in ATR = expansion/compression
    df["atr_expansion"] = df["atr"].pct_change()

    # -------------------------------------------
    # 9 — Volatility Percentile (30-day)
    # -------------------------------------------
    window = 288 * 30  # 30 days of 5m candles

    # Raw volatility (intraday return std)
    df["volatility"] = close.pct_change().rolling(288).std()

    # Efficient rolling percentile using numpy
    def fast_percentile(arr):
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return np.nan
        # percentile of last element compared to window
        return (arr <= arr[-1]).mean()

    df["vol_percentile"] = (
        df["volatility"]
        .rolling(window, min_periods=window)
        .apply(fast_percentile, raw=True)
    )

    # -------------------------------------------
    # 10 — Bollinger Band Width (proper)
    # -------------------------------------------
    ma = close.rolling(20).mean()
    std = close.rolling(20).std()

    df["bb_width"] = (4 * std) / ma  # proper BB width

    return df