"""
High-level feature builders for raw regime detection.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from regime.rolling_metrics import (
    rolling_chop_index,
    rolling_entropy,
    rolling_hurst_exponent,
    rolling_lr_gradient,
    rolling_range_compression,
    rolling_slope,
    rolling_variance,
    rolling_volatility,
)

logger = logging.getLogger(__name__)


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window, min_periods=1).mean()


def compute_impulse_correction(df: pd.DataFrame) -> pd.Series:
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    impulse = (df["close"] - df["open"]).abs() / candle_range
    return impulse.rolling(20, min_periods=1).mean().fillna(0)


def compute_bos_choch_frequency(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    # Keep these as boolean for logical operations
    higher_high = df["high"] > df["high"].shift(1)
    higher_low = df["low"] > df["low"].shift(1)

    # BOS occurs when both HH and HL are true
    bos_signal = higher_high & higher_low

    # Rolling BOS counts
    bos_20 = bos_signal.rolling(20, min_periods=1).sum().astype(float)
    bos_50 = bos_signal.rolling(50, min_periods=1).sum().astype(float)

    return bos_20, bos_50

def compute_vol_compression(df: pd.DataFrame) -> pd.Series:
    rng = df["high"] - df["low"]
    return rng / (rng.rolling(20, min_periods=1).mean() + 1e-12)


def compute_vol_expansion(ret: pd.Series) -> pd.Series:
    std20 = rolling_volatility(ret, 20)
    return (std20 - std20.rolling(50, min_periods=1).mean()) / (
        std20.rolling(50, min_periods=1).std(ddof=0) + 1e-12
    )


def compute_liquidity_drought(volume: pd.Series) -> pd.Series:
    return (volume < volume.rolling(50, min_periods=1).quantile(0.2)).astype(np.float32)


def compute_noise_to_signal(df: pd.DataFrame) -> pd.Series:
    return ((df["close"] - df["open"]).abs() / (df["high"] - df["low"]).replace(0, np.nan)).fillna(0)


def compute_htf_alignment(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"].astype("datetime64[ns]")
    tmp = df.set_index(ts)
    out = pd.DataFrame(index=df.index)
    for freq, name in [("5T", "5m"), ("15T", "15m"), ("1H", "1h")]:
        res = tmp["close"].resample(freq).last().ffill()
        # slope window must be >= 2
        out[f"htf_trend_{name}"] = rolling_slope(res, 10).reindex(tmp.index, method="ffill")
    return out


def compute_sweep_flags(df: pd.DataFrame) -> pd.Series:
    high = df["high"]
    low = df["low"]
    roll_high = high.rolling(20, min_periods=1).max()
    roll_low = low.rolling(20, min_periods=1).min()
    sweep_high = (high >= roll_high.shift(1)) & (df["close"] < df["open"])
    sweep_low = (low <= roll_low.shift(1)) & (df["close"] > df["open"])
    return (sweep_high | sweep_low).astype(np.float32)


def build_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    ret = close.pct_change().fillna(0)
    atr14 = _atr(df, 14)

    # FIX: slope_1 removed — slope needs >=2 window
    out["slope_2"] = rolling_slope(close, 2).fillna(0)
    out["slope_5"] = rolling_slope(close, 5)
    out["slope_15"] = rolling_slope(close, 15)

    out["lr_grad_fast"] = rolling_lr_gradient(close, 10)
    out["trend_strength"] = out["slope_5"] / (rolling_volatility(ret, 20) + 1e-12)
    out["atr_adjusted_trend"] = out["slope_5"] / (atr14 + 1e-12)

    htf = compute_htf_alignment(df)
    out = pd.concat([out, htf], axis=1)

    out["impulse_correction_ratio"] = compute_impulse_correction(df)
    out["higher_high_score"] = (df["high"] > df["high"].shift(1)).rolling(20, min_periods=1).mean()
    out["higher_low_score"] = (df["low"] > df["low"].shift(1)).rolling(20, min_periods=1).mean()

    bos20, bos50 = compute_bos_choch_frequency(df)
    out["bos_count_20"] = bos20
    out["bos_count_50"] = bos50
    return out


def build_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    ret = df["close"].pct_change().fillna(0)

    out["realized_vol_1"] = rolling_volatility(ret, 1)
    out["realized_vol_5"] = rolling_volatility(ret, 5)
    out["realized_vol_15"] = rolling_volatility(ret, 15)
    out["realized_vol_30"] = rolling_volatility(ret, 30)

    out["var_ratio_5_20"] = (rolling_variance(ret, 5) + 1e-12) / (rolling_variance(ret, 20) + 1e-12)
    out["var_ratio_20_50"] = (rolling_variance(ret, 20) + 1e-12) / (rolling_variance(ret, 50) + 1e-12)

    out["vol_compression_score"] = compute_vol_compression(df)
    out["vol_expansion_score"] = compute_vol_expansion(ret)

    atr = _atr(df, 14)
    out["atr_percentile"] = rolling_entropy(atr, 200)
    out["vol_clustering"] = rolling_chop_index(ret.pow(2), 20)
    out["vol_breakout_flag"] = (out["atr_percentile"] > 0.9).astype(np.float32)

    return out


def build_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    volume = df["volume"].replace(0, np.nan)
    delta = df.get("delta", pd.Series(0, index=df.index))

    out["delta_intensity"] = (delta.abs() / volume).fillna(0)
    out["imbalance_persistence"] = np.sign(delta).rolling(20, min_periods=1).sum()
    cvd = delta.cumsum()

    out["cvd_slope"] = rolling_slope(cvd, 20)
    out["volume_expansion_5"] = volume / (volume.rolling(5, min_periods=1).mean() + 1e-12)
    out["volume_expansion_20"] = volume / (volume.rolling(20, min_periods=1).mean() + 1e-12)
    out["liquidity_drought_score"] = compute_liquidity_drought(volume)

    return out


def build_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    ret = df["close"] - df["close"].shift(1)

    out["micro_trend_entropy"] = rolling_entropy(ret.fillna(0), 20)

    direction = np.sign(ret.fillna(0))
    out["bar_clustering_score"] = direction.rolling(20, min_periods=1).mean().abs()
    out["chop_index"] = rolling_chop_index(direction, 20)

    body = (df["close"] - df["open"]).replace(0, np.nan)
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

    out["upper_wick_body"] = (upper_wick / body.abs()).fillna(0)
    out["lower_wick_body"] = (lower_wick / body.abs()).fillna(0)
    out["noise_to_signal"] = compute_noise_to_signal(df)
    out["hurst_exponent"] = rolling_hurst_exponent(df["close"], 50)

    return out


def build_structure_range_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    high = df["high"]
    low = df["low"]
    close = df["close"]

    atr = _atr(df, 14)

    out["distance_to_last_swing_high"] = (close - high.rolling(20, min_periods=1).max()) / (atr + 1e-12)
    out["distance_to_last_swing_low"] = (close - low.rolling(20, min_periods=1).min()) / (atr + 1e-12)

    out["range_compression"] = rolling_range_compression(high, low, 20)

    ts = df["timestamp"].astype("datetime64[ns]")
    tmp = df.set_index(ts)

    htf = tmp.resample("1H").agg({"high": "max", "low": "min", "close": "last"}).ffill()
    htf_pos = (htf["close"] - htf["low"]) / ((htf["high"] - htf["low"]).replace(0, np.nan))

    out["htf_range_position"] = htf_pos.reindex(tmp.index, method="ffill").fillna(0)
    out["sweep_flags"] = compute_sweep_flags(df)

    out["mean_reversion_distance"] = close - close.rolling(20, min_periods=1).mean()
    out["range_tightness_score"] = ((high - low).rolling(20, min_periods=1).mean()) / (atr + 1e-12)

    return out