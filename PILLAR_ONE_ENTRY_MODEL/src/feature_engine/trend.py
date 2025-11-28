import numpy as np
import pandas as pd

def compute_trend_signals(df, df_htf):
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # ----------------------------------------------------
    # 4 — Trend slope (EMA)
    # ----------------------------------------------------
    ema = close.ewm(span=50).mean()
    df["trend_slope"] = ema.diff()

    # ----------------------------------------------------
    # 5 — Trend strength (proper ADX)
    # ----------------------------------------------------
    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    plus_dm = (high.diff() > low.diff().abs()) * (high.diff().clip(lower=0))
    minus_dm = (low.diff().abs() > high.diff()) * (low.diff().abs().clip(lower=0))

    atr = tr.ewm(alpha=1/14).mean()
    pdi = (plus_dm.ewm(alpha=1/14).mean() / atr) * 100
    mdi = (minus_dm.ewm(alpha=1/14).mean() / atr) * 100

    dx = (abs(pdi - mdi) / (pdi + mdi)).fillna(0) * 100
    adx = dx.ewm(alpha=1/14).mean()

    df["trend_strength"] = adx

    # ----------------------------------------------------
    # 6 — HTF Trend Alignment
    # ----------------------------------------------------
    # Ensure timestamps exist first
    df_5m_index = pd.to_datetime(df["timestamp"], unit="s")

    df_15m = df_htf["15m"].copy()
    df_1h = df_htf["1h"].copy()

    df_15m.index = pd.to_datetime(df_15m["timestamp"], unit="s")
    df_1h.index = pd.to_datetime(df_1h["timestamp"], unit="s")

    # Resample to 5m so timestamps align perfectly
    df_15m_res = df_15m["close"].resample("5T").last().ffill()
    df_1h_res = df_1h["close"].resample("5T").last().ffill()

    # Reindex to exact 5m timestamps
    df["htf_trend_15m"] = df_15m_res.reindex(df_5m_index).ewm(span=50).mean().values
    df["htf_trend_1h"] = df_1h_res.reindex(df_5m_index).ewm(span=50).mean().values

    df["htf_alignment"] = (
        (df["htf_trend_15m"].diff() > 0).astype(int) +
        (df["htf_trend_1h"].diff() > 0).astype(int)
    ) / 2

    # ----------------------------------------------------
    # 7 — Trend Efficiency (R² of linear regression)
    # ----------------------------------------------------
    window = 50
    r2 = np.full(len(close), np.nan)

    for i in range(window, len(close)):
        y = close[i-window:i].values
        x = np.arange(window)

        slope, intercept = np.polyfit(x, y, 1)
        y_pred = intercept + slope * x

        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2[i] = 1 - ss_res / ss_tot

    df["trend_efficiency"] = r2

    return df