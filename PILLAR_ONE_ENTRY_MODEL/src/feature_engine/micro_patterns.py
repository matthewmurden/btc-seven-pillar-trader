import numpy as np
import pandas as pd

def compute_micro_pattern_signals(df):

    # --------------------------------------------------------
    # Basic candle body/range
    # --------------------------------------------------------
    df["candle_body"] = df["close"] - df["open"]
    df["candle_range"] = df["high"] - df["low"]
    df["candle_strength"] = df["candle_body"].abs() / df["candle_range"].replace(0, np.nan)

    # --------------------------------------------------------
    # Wick sizes
    # --------------------------------------------------------
    df["upper_wick"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["lower_wick"] = df[["close", "open"]].min(axis=1) - df["low"]

    # --------------------------------------------------------
    # 1. Pin Bar (large wick, small body)
    # --------------------------------------------------------
    df["pin_bar"] = (
        (df["upper_wick"] > df["candle_body"].abs() * 2) |
        (df["lower_wick"] > df["candle_body"].abs() * 2)
    ).astype(int)

    # --------------------------------------------------------
    # 2. Momentum Bar (large body, tiny wicks)
    # --------------------------------------------------------
    df["momentum_bar"] = (
        (df["candle_body"].abs() > df["candle_range"] * 0.7) &
        (df["upper_wick"] < df["candle_range"] * 0.1) &
        (df["lower_wick"] < df["candle_range"] * 0.1)
    ).astype(int)

    # --------------------------------------------------------
    # 3. Engulfing (current body > previous body AND expands past highs/lows)
    # --------------------------------------------------------
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)

    prev_body = (prev_close - prev_open).abs()
    curr_body = df["candle_body"].abs()

    df["engulfing"] = (
        (curr_body > prev_body) &
        (df["high"] > prev_high) &
        (df["low"] < prev_low)
    ).astype(int)

    # --------------------------------------------------------
    # 4. Imbalance bar (large body relative to last N bars)
    # --------------------------------------------------------
    rolling_body = df["candle_body"].abs().rolling(20).mean()

    df["imbalance_bar"] = (
        df["candle_body"].abs() > rolling_body * 1.5
    ).astype(int)

    # --------------------------------------------------------
    # Final micro-pattern score (0–4)
    # --------------------------------------------------------
    df["pattern_score"] = (
        df["pin_bar"] +
        df["momentum_bar"] +
        df["engulfing"] +
        df["imbalance_bar"]
    )

    return df