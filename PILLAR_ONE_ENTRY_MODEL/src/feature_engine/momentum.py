import pandas as pd
import numpy as np

def compute_momentum_signals(df):
    close = df["close"]
    volume = df["volume"]

    # ---------------------------------------------------------
    # 1 — Short-term ROC (5–20 blended)
    # ---------------------------------------------------------
    df["roc_5"] = close.pct_change(5)
    df["roc_10"] = close.pct_change(10)
    df["roc_20"] = close.pct_change(20)

    # A blended short-term momentum signal
    df["mom_short"] = (df["roc_5"] + df["roc_10"] + df["roc_20"]) / 3

    # ---------------------------------------------------------
    # 2 — Medium-term momentum (EMA slope)
    # ---------------------------------------------------------
    ema = close.ewm(span=20).mean()
    df["mom_medium"] = ema.diff()

    # ---------------------------------------------------------
    # 3 — Volume-adjusted momentum
    #     OBV + VWAP deviation
    # ---------------------------------------------------------

    # Correct OBV
    obv = [0]
    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])

    df["obv"] = obv
    df["obv_slope"] = df["obv"].diff()

    # VWAP -> volume-weighted average price
    df["vwap"] = (close * volume).rolling(20).sum() / volume.rolling(20).sum()

    df["vwap_dev"] = close - df["vwap"]

    # Combined volume-adjusted momentum
    df["mom_volume_adj"] = (
        df["obv_slope"].fillna(0) +
        df["vwap_dev"].fillna(0)
    )

    return df