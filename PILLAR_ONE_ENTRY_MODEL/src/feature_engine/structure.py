import pandas as pd
import numpy as np

def compute_structure_signals(df):

    # ----------------------------------------------------
    # 1. Detect swing highs/lows (pivot structure)
    # ----------------------------------------------------
    df["swing_high"] = (
        (df["high"] > df["high"].shift(1)) &
        (df["high"] > df["high"].shift(-1))
    ).astype(int)

    df["swing_low"] = (
        (df["low"] < df["low"].shift(1)) &
        (df["low"] < df["low"].shift(-1))
    ).astype(int)

    # Store last swing points
    df["last_swing_high"] = df["high"].where(df["swing_high"] == 1).ffill()
    df["last_swing_low"] = df["low"].where(df["swing_low"] == 1).ffill()

    # ----------------------------------------------------
    # 2. Break Of Structure (BOS)
    # ----------------------------------------------------
    df["bos_up"] = (
        (df["close"] > df["last_swing_high"].shift(1)) &
        (df["trend_slope"] > 0)
    ).astype(int)

    df["bos_down"] = (
        (df["close"] < df["last_swing_low"].shift(1)) &
        (df["trend_slope"] < 0)
    ).astype(int)

    # ----------------------------------------------------
    # 3. CHOCH — Change of Character
    # bullish choch = break previous swing high while in downtrend
    # bearish choch = break previous swing low while in uptrend
    # ----------------------------------------------------
    df["choch_up"] = (
        (df["close"] > df["last_swing_high"].shift(1)) &
        (df["trend_slope"] < 0)        # opposite trend
    ).astype(int)

    df["choch_down"] = (
        (df["close"] < df["last_swing_low"].shift(1)) &
        (df["trend_slope"] > 0)
    ).astype(int)

    # ----------------------------------------------------
    # 4. Combine signals for model input
    # ----------------------------------------------------
    df["structure_signal"] = (
        df["bos_up"] -
        df["bos_down"] +
        df["choch_up"] -
        df["choch_down"]
    )

    return df