import numpy as np

def compute_regime_features(df):

    # ----------------------------------------------------
    # 1. Volatility percentile (needed for all logic)
    # ----------------------------------------------------
    if "vol_percentile" not in df.columns:
        returns = df["close"].pct_change().abs()
        df["vol_percentile"] = returns.rolling(200).apply(
            lambda x: (x < x.iloc[-1]).mean(), raw=False
        )

    # ----------------------------------------------------
    # 2. Trend regime
    # Strong trend = positive trend slope + smooth trend + medium volatility
    # ----------------------------------------------------
    df["regime_trend"] = (
        (df["trend_slope"] > 0) &
        (df["trend_efficiency"] > 0.4) &
        (df["vol_percentile"].between(20, 70))
    ).astype(int)

    # ----------------------------------------------------
    # 3. Range regime
    # Low volatility + wide oscillation + low trend efficiency
    # ----------------------------------------------------
    df["regime_range"] = (
        (df["vol_percentile"] < 25) &
        (df["trend_efficiency"] < 0.3)
    ).astype(int)

    # ----------------------------------------------------
    # 4. Breakout regime
    # Volatility expansion + BB squeeze released
    # ----------------------------------------------------
    if "bb_width" not in df.columns:
        df["bb_width"] = (df["high"] - df["low"]).rolling(20).std()

    if "atr_expansion" not in df.columns:
        df["atr_expansion"] = df["atr"].diff()

    df["regime_breakout"] = (
        (df["atr_expansion"] > df["atr_expansion"].rolling(200).mean()) &
        (df["bb_width"] > df["bb_width"].rolling(200).mean())
    ).astype(int)

    # ----------------------------------------------------
    # 5. Select ONE regime only
    # Priority: breakout > trend > range
    # ----------------------------------------------------
    df["regime"] = 0

    df.loc[df["regime_range"] == 1, "regime"] = 2
    df.loc[df["regime_trend"] == 1, "regime"] = 1
    df.loc[df["regime_breakout"] == 1, "regime"] = 3

    # ----------------------------------------------------
    # 6. Smooth regime changes (persistence filter)
    # ----------------------------------------------------
    df["regime_smoothed"] = (
        df["regime"]
        .rolling(5)
        .apply(lambda x: np.bincount(x.astype(int)).argmax(), raw=False)
    ).fillna(method="bfill")

    df["regime"] = df["regime_smoothed"].astype(int)

    return df