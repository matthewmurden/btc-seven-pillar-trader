import pandas as pd
import numpy as np

def compute_mean_reversion_signals(df):
    close = df["close"]

    # -------------------------------------------------------
    # 11 — Z-Score stretch (20-period)
    # -------------------------------------------------------
    mean20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()

    df["zscore"] = (close - mean20) / std20

    # -------------------------------------------------------
    # 12 — RSI (safe version)
    # -------------------------------------------------------
    delta = close.diff()

    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    # Prevent division by zero
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)

    # -------------------------------------------------------
    # 13 — Overshoot score (RSI deviation)
    # Used only in low-vol regimes
    # -------------------------------------------------------
    df["rsi_overshoot"] = df["rsi"] - df["rsi"].rolling(50).mean()

    # Volatility gating (assumes df["vol_percentile"] exists)
    if "vol_percentile" in df.columns:
        df["reversion_signal"] = np.where(
            df["vol_percentile"] < 30,       # only low volatility
            df["zscore"].abs() + df["rsi_overshoot"].abs(),
            0                                # disabled in trend/high-vol
        )
    else:
        df["reversion_signal"] = df["zscore"].abs() + df["rsi_overshoot"].abs()

    return df