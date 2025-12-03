import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# Pull in pre-process raw data layer
ROOT = Path(__file__).resolve().parent.parent
PREPROCESS_DIR = ROOT / "PRE-PROCESS"
if str(PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_DIR))
from raw_data_layer import load_raw_data  # type: ignore

OUT = ROOT / "data" / "processed" / "part_one" / "raw_regime_features.parquet"

EPS = 1e-12


# ============================================================
# Helpers
# ============================================================
def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Fast linear regression slope."""
    idx = np.arange(window)
    denom = ((idx - idx.mean()) ** 2).sum()

    def _apply(x):
        y = x.values
        if len(y) < window:
            return 0.0
        return ((idx - idx.mean()) * (y - y.mean())).sum() / denom

    return series.rolling(window).apply(_apply, raw=False).fillna(0)


# ============================================================
# Feature blocks
# ============================================================
def trend_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]

    out["rolling_slope"] = rolling_slope(close, 20)

    ret = close.pct_change().fillna(0)
    vol = ret.rolling(20).std().replace(0, np.nan)
    out["trend_strength"] = (out["rolling_slope"].abs() / vol).fillna(0)

    out["higher_high_flag"] = (out["high"] > out["high"].shift(1)).astype(int)
    out["higher_low_flag"] = (out["low"] > out["low"].shift(1)).astype(int)

    out["regression_gradient"] = rolling_slope(close, 50)

    fast = close.ewm(span=20, adjust=False).mean()
    slow = close.ewm(span=50, adjust=False).mean()
    out["st_lt_divergence"] = (fast - slow) / (close.abs() + EPS)

    return out


def volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ret = out["close"].pct_change().fillna(0)

    out["realised_vol_5m"] = ret.rolling(5).std().fillna(0)
    out["realised_vol_15m"] = ret.rolling(15).std().fillna(0)
    out["realised_vol_1h"] = ret.rolling(60).std().fillna(0)

    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - out["close"].shift(1)).abs(),
            (out["low"] - out["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr14 = tr.rolling(14).mean()
    atr200 = tr.rolling(200).mean()
    out["atr_percentile"] = (atr14 / (atr200 + EPS)).fillna(0)

    vol20 = ret.abs().rolling(200).quantile(0.2)
    out["compression_flag"] = (ret.abs() < vol20).astype(int).fillna(0)

    out["vol_clustering"] = ret.rolling(50).std().fillna(0)

    fast_var = ret.rolling(20).var()
    slow_var = ret.rolling(60).var()
    out["variance_ratio"] = (fast_var / (slow_var + EPS)).fillna(0)

    return out


def liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["delta_persistence_feat"] = out["delta_persistence"]

    ob = out["order_imbalance"].fillna(0)
    sign = np.sign(ob)
    out["imbalance_cycles"] = (sign != sign.shift(1)).rolling(50).sum().fillna(0)

    drought_q = out["volume"].rolling(240).quantile(0.05)
    out["volume_drought"] = (out["volume"] < drought_q).astype(int).fillna(0)

    out["wick_body_irregularity"] = out["wick_body_ratio"]

    out["order_book_proxy"] = out["order_imbalance"]

    cum_delta = out["delta"].cumsum()
    out["cvd_slope"] = rolling_slope(cum_delta, 20)

    return out


def range_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    prev_high = out["high"].shift(1).rolling(20).max()
    prev_low = out["low"].shift(1).rolling(20).min()
    out["sweep_flag"] = (
        (out["high"] > prev_high) | (out["low"] < prev_low)
    ).astype(int).fillna(0)

    width = (out["high"] - out["low"]).rolling(50).mean()
    out["range_compression"] = (width / (out["close"].abs() + EPS)).fillna(0)

    hi = out["high"].rolling(50).max()
    lo = out["low"].rolling(50).min()
    mid = (hi + lo) / 2
    out["mean_reversion_distance"] = (out["close"] - mid).fillna(0)

    swing_hi = out["high"].shift(1).rolling(100).max()
    swing_lo = out["low"].shift(1).rolling(100).min()
    out["dist_prev_swing_high"] = (out["close"] - swing_hi).fillna(0)
    out["dist_prev_swing_low"] = (out["close"] - swing_lo).fillna(0)

    out["fair_value_divergence"] = (out["close"] - mid.shift(1)).fillna(0)

    return out


def microbehaviour_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ret = out["close"].pct_change().fillna(0)

    flips = (np.sign(ret) != np.sign(ret.shift(1))).rolling(30).mean()
    out["chop_index"] = flips.fillna(0)

    out["entropy_feat"] = out["entropy"]

    cd = np.sign(out["close"] - out["open"])
    out["alternating_candles"] = (
        (cd != cd.shift(1)).rolling(20).mean().fillna(0)
    )

    out["no_trend_stat_feat"] = out["no_trend_stat"]

    return out


def session_context_features(df: pd.DataFrame, htf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = df.copy()

    ts = pd.to_datetime(out["open_time_ms"], unit="ms", utc=True)
    hours = ts.dt.hour

    out["session_asia"] = hours.between(0, 7).astype(int)
    out["session_london"] = hours.between(7, 13).astype(int)
    out["session_ny"] = hours.between(13, 20).astype(int)

    ret = out["close"].pct_change().fillna(0)
    out["session_vol_bias"] = ret.rolling(60).std().fillna(0)

    vol24 = ret.rolling(1440).std().fillna(0)
    out["session_liq_compression"] = (
        out["session_vol_bias"] / (vol24 + EPS)
    ).fillna(0)

    # --- HTF volatility slope ---
    merged = out[["open_time_ms"]].copy()
    for key, fr in htf.items():
        fr["ret"] = fr["close"].pct_change()
        fr["vol"] = fr["ret"].rolling(5).std().fillna(0)
        merged = merged.merge(
            fr[["open_time_ms", "vol"]].rename(columns={"vol": f"vol_{key}"}),
            on="open_time_ms",
            how="left",
        )

    merged = merged.fillna(method="ffill").fillna(0)

    def slope_row(r):
        x = np.array([5, 15, 60, 240], float)
        y = np.array(
            [r["vol_5m"], r["vol_15m"], r["vol_1h"], r["vol_4h"]], float
        )
        if np.all(y == 0):
            return 0.0
        return np.polyfit(x, y, 1)[0]

    out["htf_vol_slope"] = merged.apply(slope_row, axis=1)

    # --- Daily trend orientation ---
    d = htf["1d"].copy()
    d["ema_fast"] = d["close"].ewm(span=20).mean()
    d["ema_slow"] = d["close"].ewm(span=50).mean()
    d["daily_trend_orientation"] = np.sign(d["ema_fast"] - d["ema_slow"])

    out = out.merge(
        d[["open_time_ms", "daily_trend_orientation"]],
        on="open_time_ms",
        how="left",
    ).fillna({"daily_trend_orientation": 0})

    # --- Volatility cycle phase ---
    tr = pd.concat(
        [
            d["high"] - d["low"],
            (d["high"] - d["close"].shift(1)).abs(),
            (d["low"] - d["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(14).mean()
    mean = atr.rolling(50).mean()
    std = atr.rolling(50).std().replace(0, np.nan)
    d["vol_cycle_phase"] = ((atr - mean) / (std + EPS)).fillna(0)

    out = out.merge(
        d[["open_time_ms", "vol_cycle_phase"]],
        on="open_time_ms",
        how="left",
    ).fillna({"vol_cycle_phase": 0})

    return out


# ============================================================
# Main
# ============================================================
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logging.info("Loading canonical raw dataset via pre-process layer")
    data = load_raw_data()
    candles = data["ohlcv"]
    delta = data["delta"]
    micro = data["microstructure"]
    htf = data["htf"]

    df = (
        candles.merge(delta, on="open_time_ms", how="left")
               .merge(micro, on="open_time_ms", how="left")
               .fillna(0)
    )

    logging.info("Building feature blocks...")
    df = trend_features(df)
    df = volatility_features(df)
    df = liquidity_features(df)
    df = range_features(df)
    df = microbehaviour_features(df)
    df = session_context_features(df, htf)

    feature_cols = [
        "open_time_ms",
        # Trend
        "rolling_slope", "trend_strength",
        "higher_high_flag", "higher_low_flag",
        "regression_gradient", "st_lt_divergence",
        # Volatility
        "realised_vol_5m","realised_vol_15m","realised_vol_1h",
        "atr_percentile","compression_flag","vol_clustering","variance_ratio",
        # Liquidity
        "delta_persistence_feat","imbalance_cycles","volume_drought",
        "wick_body_irregularity","order_book_proxy","cvd_slope",
        # Range
        "sweep_flag","range_compression","mean_reversion_distance",
        "dist_prev_swing_high","dist_prev_swing_low","fair_value_divergence",
        # Market Micro-Behaviour
        "chop_index","entropy_feat","alternating_candles","no_trend_stat_feat",
        # Session Context
        "session_asia","session_london","session_ny",
        "session_vol_bias","session_liq_compression",
        "htf_vol_slope","daily_trend_orientation","vol_cycle_phase",
    ]

    out = df[feature_cols].fillna(0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    logging.info(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
