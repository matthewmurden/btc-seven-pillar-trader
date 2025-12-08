import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed" / "part_one" / "raw_regime_features.parquet"
# Normalised input produced upstream by Global Scaling Engine
NORM_INPUT = ROOT.parent / "PRE_PILLARFOUR" / "processed" / "normalised_features.parquet"

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
    """
    Trend structure signals:
      - rolling_slope
      - trend_strength
      - higher_high_flag / higher_low_flag
      - regression_gradient
      - st_lt_divergence
    These are genuinely new regime features, so we build them here.
    """
    out = df.copy()
    close = out["close"]

    # Local slope of 1m close
    out["rolling_slope"] = rolling_slope(close, 20)

    # Trend strength as slope / local volatility (still raw, regime-specific)
    ret = close.pct_change().fillna(0)
    vol = ret.rolling(20).std().replace(0, np.nan)
    out["trend_strength"] = (out["rolling_slope"].abs() / vol).fillna(0)

    # Higher-high and higher-low flags
    out["higher_high_flag"] = (out["high"] > out["high"].shift(1)).astype(int)
    out["higher_low_flag"] = (out["low"] > out["low"].shift(1)).astype(int)

    # Longer regression gradient
    out["regression_gradient"] = rolling_slope(close, 50)

    # Short-term vs long-term trend divergence
    fast = close.ewm(span=20, adjust=False).mean()
    slow = close.ewm(span=50, adjust=False).mean()
    out["st_lt_divergence"] = (fast - slow) / (close.abs() + EPS)

    return out


def volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volatility structure signals:
      - realised_vol_5m / 15m / 1h
      - atr_percentile
      - compression_flag
      - vol_clustering
      - variance_ratio

    Uses existing Phase 0 engine outputs if present; otherwise falls back
    to local computations.
    """
    out = df.copy()

    # Realised volatility horizons: prefer Vol Engine outputs if available
    if {"rv_5m", "rv_15m", "rv_1h"}.issubset(out.columns):
        out["realised_vol_5m"] = out["rv_5m"]
        out["realised_vol_15m"] = out["rv_15m"]
        out["realised_vol_1h"] = out["rv_1h"]
    else:
        close = out["close"]
        ret = close.pct_change().fillna(0)
        out["realised_vol_5m"] = ret.rolling(5).std().fillna(0)
        out["realised_vol_15m"] = ret.rolling(15).std().fillna(0)
        out["realised_vol_1h"] = ret.rolling(60).std().fillna(0)

    # ATR percentile: prefer global ATRs if present
    if {"atr_14", "atr_200"}.issubset(out.columns):
        atr14 = out["atr_14"]
        atr200 = out["atr_200"]
        out["atr_percentile"] = (atr14 / (atr200 + EPS)).fillna(0)
    else:
        # Fallback ATR computation from raw OHLC
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

    # Compression flag: reuse Vol Engine if available
    if "vol_compression" in out.columns:
        out["compression_flag"] = out["vol_compression"]
    else:
        close = out["close"]
        ret = close.pct_change().fillna(0)
        vol20 = ret.abs().rolling(200).quantile(0.2)
        out["compression_flag"] = (ret.abs() < vol20).astype(int).fillna(0)

    # Vol clustering and variance ratio are genuinely new regime features
    close = out["close"]
    ret = close.pct_change().fillna(0)

    out["vol_clustering"] = ret.rolling(50).std().fillna(0)

    fast_var = ret.rolling(20).var()
    slow_var = ret.rolling(60).var()
    out["variance_ratio"] = (fast_var / (slow_var + EPS)).fillna(0)

    return out


def liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Liquidity / microstructure signals:
      - delta_persistence_feat (alias)
      - imbalance_cycles
      - volume_drought
      - wick_body_irregularity (alias)
      - order_book_proxy (alias)
      - cvd_slope
    """
    out = df.copy()

    # Direct aliases of Phase 0 features
    out["delta_persistence_feat"] = out["delta_persistence"]
    out["wick_body_irregularity"] = out["wick_body_ratio"]
    out["order_book_proxy"] = out["order_imbalance"]

    # Imbalance flip cycles
    ob = out["order_imbalance"].fillna(0)
    sign = np.sign(ob)
    out["imbalance_cycles"] = (sign != sign.shift(1)).rolling(50).sum().fillna(0)

    # Volume drought relative to rolling low volume quantile
    drought_q = out["volume"].rolling(240).quantile(0.05)
    out["volume_drought"] = (out["volume"] < drought_q).astype(int).fillna(0)

    # Cumulative delta slope
    cum_delta = out["delta"].cumsum()
    out["cvd_slope"] = rolling_slope(cum_delta, 20)

    return out


def range_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Range / structure context:
      - sweep_flag
      - range_compression
      - mean_reversion_distance
      - dist_prev_swing_high / low
      - fair_value_divergence
    """
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
    """
    Market micro-behaviour:
      - chop_index
      - entropy_feat (alias)
      - alternating_candles
      - no_trend_stat_feat (alias)
    """
    out = df.copy()
    close = out["close"]
    ret = close.pct_change().fillna(0)

    flips = (np.sign(ret) != np.sign(ret.shift(1))).rolling(30).mean()
    out["chop_index"] = flips.fillna(0)

    out["entropy_feat"] = out["entropy"]
    cd = np.sign(out["close"] - out["open"])
    out["alternating_candles"] = (cd != cd.shift(1)).rolling(20).mean().fillna(0)

    out["no_trend_stat_feat"] = out["no_trend_stat"]

    return out


def session_context_features(df: pd.DataFrame, htf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Session context:
      - session_asia / london / ny (reuse Phase 0 if present)
      - session_vol_bias (reuse if present)
      - session_liq_compression (alias of session_liquidity_compression if present)
      - htf_vol_slope (new cross-TF vol slope)
      - daily_trend_orientation (new)
      - vol_cycle_phase (new)
    """
    out = df.copy()

    # Session flags: reuse existing if available, otherwise derive from hour of day
    need_session_flags = not (
        {"session_asia", "session_london", "session_ny"}.issubset(out.columns)
    )
    if need_session_flags:
        ts = pd.to_datetime(out["open_time_ms"], unit="ms", utc=True)
        hours = ts.dt.hour
        out["session_asia"] = hours.between(0, 7).astype(int)
        out["session_london"] = hours.between(7, 13).astype(int)
        out["session_ny"] = hours.between(13, 20).astype(int)

    # Vol bias and liquidity compression: prefer Phase 0 engine outputs
    ret = None

    if "session_vol_bias" not in out.columns:
        close = out["close"]
        ret = close.pct_change().fillna(0)
        out["session_vol_bias"] = ret.rolling(60).std().fillna(0)

    # Alias session_liquidity_compression -> session_liq_compression if present
    if "session_liq_compression" not in out.columns:
        if "session_liquidity_compression" in out.columns:
            out["session_liq_compression"] = out["session_liquidity_compression"]
        else:
            # Need to compute manually
            if ret is None:
                close = out["close"]
                ret = close.pct_change().fillna(0)
            vol24 = ret.rolling(1440).std().fillna(0)
            out["session_liq_compression"] = (
                out["session_vol_bias"] / (vol24 + EPS)
            ).fillna(0)

    # --- HTF volatility slope (new cross-TF vol slope) ---
    merged = out[["open_time_ms"]].copy()
    for key, fr in htf.items():
        fr = fr.copy()
        fr["ret"] = fr["close"].pct_change().fillna(0)
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
            [r.get("vol_5m", 0.0), r.get("vol_15m", 0.0),
             r.get("vol_1h", 0.0), r.get("vol_4h", 0.0)],
            float,
        )
        if np.all(y == 0):
            return 0.0
        return np.polyfit(x, y, 1)[0]

    out["htf_vol_slope"] = merged.apply(slope_row, axis=1)

    # --- Daily trend orientation (new) ---
    if "1d" in htf:
        d = htf["1d"].copy()
        d["ema_fast"] = d["close"].ewm(span=20).mean()
        d["ema_slow"] = d["close"].ewm(span=50).mean()
        d["daily_trend_orientation"] = np.sign(d["ema_fast"] - d["ema_slow"])

        out = out.merge(
            d[["open_time_ms", "daily_trend_orientation"]],
            on="open_time_ms",
            how="left",
        ).fillna({"daily_trend_orientation": 0})
    else:
        out["daily_trend_orientation"] = 0

    # --- Volatility cycle phase (new) ---
    if "1d" in htf:
        d = htf["1d"].copy()
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
    else:
        out["vol_cycle_phase"] = 0

    return out


def build_htf_from_normalised(df: pd.DataFrame, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Build HTF OHLCV mini-dataframes from the normalised input, using
    the *_<tf> columns (5m, 15m, 1h, 4h, 1d).
    """
    out: Dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        time_col = f"open_time_{tf}"
        close_col = f"close_{tf}"
        if time_col not in df.columns or close_col not in df.columns:
            continue
        cols = [c for c in df.columns if c.endswith(f"_{tf}")]
        htf_df = df[cols].copy()
        # ensure time column present and converted to ms
        htf_df = htf_df.rename(columns={time_col: "open_time"})
        htf_df["open_time_ms"] = (
            pd.to_datetime(htf_df["open_time"], utc=True).astype("int64") // 1_000_000
        )
        # drop duplicate htf rows to keep unique HTF timeline
        htf_df = htf_df.drop_duplicates(subset=["open_time_ms"]).sort_values("open_time_ms")
        # rename feature columns to base names (open_5m -> open, close_5m -> close, etc.)
        rename_map = {c: c.replace(f"_{tf}", "") for c in cols if c != time_col}
        htf_df = htf_df.rename(columns=rename_map)
        out[tf] = htf_df
    return out


# ============================================================
# Main
# ============================================================
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not NORM_INPUT.exists():
        raise FileNotFoundError(f"Normalised features missing: {NORM_INPUT}")

    logging.info("Loading normalised features from %s", NORM_INPUT)
    df = pd.read_parquet(NORM_INPUT)
    if "open_time_ms" in df.columns:
        df = df.sort_values("open_time_ms").reset_index(drop=True)

    # Build HTF dict from embedded HTF columns in the normalised dataset
    htf = build_htf_from_normalised(df, ["5m", "15m", "1h", "4h", "1d"])

    logging.info("Building regime feature blocks...")
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
        "realised_vol_5m", "realised_vol_15m", "realised_vol_1h",
        "atr_percentile", "compression_flag", "vol_clustering", "variance_ratio",
        # Liquidity
        "delta_persistence_feat", "imbalance_cycles", "volume_drought",
        "wick_body_irregularity", "order_book_proxy", "cvd_slope",
        # Range
        "sweep_flag", "range_compression", "mean_reversion_distance",
        "dist_prev_swing_high", "dist_prev_swing_low", "fair_value_divergence",
        # Market Micro-Behaviour
        "chop_index", "entropy_feat", "alternating_candles", "no_trend_stat_feat",
        # Session Context
        "session_asia", "session_london", "session_ny",
        "session_vol_bias", "session_liq_compression",
        "htf_vol_slope", "daily_trend_orientation", "vol_cycle_phase",
    ]

    out = df[feature_cols].fillna(0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    logging.info("Saved → %s", OUT)


if __name__ == "__main__":
    main()