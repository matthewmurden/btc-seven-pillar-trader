import pandas as pd

from .momentum import compute_momentum_signals
from .trend import compute_trend_signals
from .volatility import compute_volatility_signals
from .mean_reversion import compute_mean_reversion_signals
from .orderflow import compute_orderflow_signals
from .structure import compute_structure_signals
from .micro_patterns import compute_micro_pattern_signals
from .regime_features import compute_regime_features


def compute_all_features(df_candles, df_trades_5m, df_htf=None):
    """
    MASTER FEATURE ENGINE
    df_candles: 5m candles with OHLCV + taker buy vol
    df_trades_5m: aggregated trades for that 5-minute candle (delta, buy/sell vol)
    df_htf: dict = {"15m": df_15, "1h": df_1h}
    """
    
    df = df_candles.copy()

    # Merge order flow
    df = df.merge(df_trades_5m, how="left", on="timestamp")

    # Compute 16 signals
    df = compute_momentum_signals(df)
    df = compute_trend_signals(df, df_htf)
    df = compute_volatility_signals(df)
    df = compute_mean_reversion_signals(df)
    df = compute_orderflow_signals(df)
    df = compute_structure_signals(df)
    df = compute_micro_pattern_signals(df)
    df = compute_regime_features(df)

    # Fill small NaNs (rolling windows)
    df = df.fillna(method="ffill").fillna(method="bfill")

    return df