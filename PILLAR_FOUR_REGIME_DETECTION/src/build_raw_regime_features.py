import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


# ==========================================
# PATHS
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

CANDLES_PATH = RAW_DIR / "candles" / "BTCUSDT_1m_2022-11-01_2025-10-31.csv"
DELTA_PATH = RAW_DIR / "delta" / "delta_stream.parquet"
MICRO_PATH = RAW_DIR / "microstructure" / "microstructure_metrics.parquet"

HTF_FILES = {
    "5m": RAW_DIR / "htf_candles" / "BTCUSDT_5m.parquet",
    "15m": RAW_DIR / "htf_candles" / "BTCUSDT_15m.parquet",
    "1h": RAW_DIR / "htf_candles" / "BTCUSDT_1h.parquet",
    "4h": RAW_DIR / "htf_candles" / "BTCUSDT_4h.parquet",
    "1d": RAW_DIR / "htf_candles" / "BTCUSDT_1d.parquet",
}

OUT_PATH = BASE_DIR / "processed" / "part_one" / "raw_regime_features.parquet"

EPS = 1e-12


def load_inputs():
    logging.info("Loading base candles")
    candles = pd.read_csv(CANDLES_PATH)
    if "open_time_ms" not in candles.columns:
        if "open_time" in candles.columns:
            candles["open_time_ms"] = pd.to_datetime(candles["open_time"]).astype("int64") // 1_000_000
        else:
            raise ValueError("Candles missing open_time_ms/open_time")
    candles = candles.sort_values("open_time_ms").reset_index(drop=True)

    logging.info("Loading delta stream")
    delta = pd.read_parquet(DELTA_PATH)

    logging.info("Loading microstructure metrics")
    micro = pd.read_parquet(MICRO_PATH)

    logging.info("Loading HTF candles")
    htf_dict: Dict[str, pd.DataFrame] = {}
    for key, path in HTF_FILES.items():
        df = pd.read_parquet(path)
        if "open_time_ms" not in df.columns:
            if "open_time" in df.columns:
                df["open_time_ms"] = pd.to_datetime(df["open_time"]).astype("int64") // 1_000_000
            else:
                raise ValueError(f"{path} missing open_time_ms")
        htf_dict[key] = df.sort_values("open_time_ms").reset_index(drop=True)

    logging.info("Merging base inputs")
    df = candles.merge(delta, on="open_time_ms", how="left")
    df = df.merge(micro, left_on="open_time_ms", right_index=False, right_on=None, how="left")
    df = df.fillna(0)
    return df, htf_dict


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    idx = np.arange(window)
    denom = ((idx - idx.mean()) ** 2).sum()

    def _calc(x):
        if x.isnull().any():
            return np.nan
        y = x.to_numpy()
        num = ((idx - idx.mean()) * (y - y.mean())).sum()
        return num / denom if denom != 0 else 0.0

    return series.rolling(window, min_periods=window).apply(_calc, raw=False)


def build_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    returns = df["close"].pct_change().fillna(0)

    df["rolling_slope"] = _rolling_slope(df["close"], 20)
    vol_win = returns.rolling(20).std().replace(0, np.nan)
    df["trend_strength"] = (df["rolling_slope"].abs() / vol_win).fillna(0)

    df["higher_high_flag"] = (df["high"] > df["high"].shift(1)).astype(int)
    df["higher_low_flag"] = (df["low"] > df["low"].shift(1)).astype(int)

    df["regression_gradient"] = _rolling_slope(df["close"], 50)

    fast_ema = df["close"].ewm(span=20, adjust=False).mean()
    slow_ema = df["close"].ewm(span=50, adjust=False).mean()
    df["st_lt_divergence"] = (fast_ema - slow_ema) / (df["close"].abs() + EPS)

    return df


def build_vol_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rets = df["close"].pct_change().fillna(0)

    df["realised_vol_5m"] = rets.rolling(5).std().fillna(0)
    df["realised_vol_15m"] = rets.rolling(15).std().fillna(0)
    df["realised_vol_1h"] = rets.rolling(60).std().fillna(0)

    # ATR
    tr_components = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    )
    tr = tr_components.max(axis=1)
    atr14 = tr.rolling(14).mean()
    atr200 = tr.rolling(200).mean()
    df["atr_percentile"] = (atr14 / (atr200 + EPS)).fillna(0)

    vol_q20 = rets.abs().rolling(200).quantile(0.2)
    df["compression_flag"] = (rets.abs() < vol_q20).astype(int).fillna(0)

    df["vol_clustering"] = rets.rolling(50).std().fillna(0)

    fast_var = rets.rolling(20).var()
    slow_var = rets.rolling(60).var()
    df["variance_ratio"] = (fast_var / (slow_var + EPS)).fillna(0)

    return df


def build_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["delta_persistence_feat"] = df.get("delta_persistence", pd.Series(index=df.index, data=0))

    ob_sign = np.sign(df.get("order_imbalance", pd.Series(index=df.index, data=0)))
    sign_changes = (ob_sign != ob_sign.shift(1)).rolling(50).sum()
    df["imbalance_cycles"] = sign_changes.fillna(0)

    vol_q05 = df["volume"].rolling(240).quantile(0.05)
    df["volume_drought"] = (df["volume"] < vol_q05).astype(int).fillna(0)

    df["wick_body_irregularity"] = df.get("wick_body_ratio", pd.Series(index=df.index, data=0))

    df["order_book_proxy"] = df.get("order_imbalance", pd.Series(index=df.index, data=0))

    cum_delta = df.get("delta", pd.Series(index=df.index, data=0)).cumsum()
    df["cvd_slope"] = _rolling_slope(cum_delta, 20).fillna(0)

    return df


def build_range_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    prev_high = df["high"].shift(1).rolling(20).max()
    prev_low = df["low"].shift(1).rolling(20).min()
    df["sweep_flag"] = ((df["high"] > prev_high) | (df["low"] < prev_low)).astype(int).fillna(0)

    width = (df["high"] - df["low"]).rolling(50).mean()
    df["range_compression"] = (width / (df["close"].abs() + EPS)).fillna(0)

    roll_max = df["high"].rolling(50).max()
    roll_min = df["low"].rolling(50).min()
    roll_mid = (roll_max + roll_min) / 2
    df["mean_reversion_distance"] = (df["close"] - roll_mid).fillna(0)

    swing_high = df["high"].shift(1).rolling(100).max()
    swing_low = df["low"].shift(1).rolling(100).min()
    df["dist_prev_swing_high"] = (df["close"] - swing_high).fillna(0)
    df["dist_prev_swing_low"] = (df["close"] - swing_low).fillna(0)

    prev_mid = roll_mid.shift(1)
    df["fair_value_divergence"] = (df["close"] - prev_mid).fillna(0)

    return df


def build_microbehaviour_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rets = df["close"].pct_change().fillna(0)
    dir_changes = (np.sign(rets) != np.sign(rets.shift(1))).rolling(30).mean()
    df["chop_index"] = dir_changes.fillna(0)

    df["entropy_feat"] = df.get("entropy", pd.Series(index=df.index, data=0))

    candle_dir = np.sign(df["close"] - df["open"])
    flips = (candle_dir != candle_dir.shift(1)).rolling(20).mean()
    df["alternating_candles"] = flips.fillna(0)

    df["no_trend_stat_feat"] = df.get("no_trend_stat", pd.Series(index=df.index, data=0))

    return df


def build_session_context_features(df: pd.DataFrame, htf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = df.copy()
    df["open_time"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    hours = df["open_time"].dt.hour

    asia = hours.between(0, 7)
    london = hours.between(7, 13)
    ny = hours.between(13, 20)

    df["session_asia"] = asia.astype(int)
    df["session_london"] = london.astype(int)
    df["session_ny"] = ny.astype(int)

    returns = df["close"].pct_change().fillna(0)
    df["session_vol_bias"] = (
        returns.rolling(60).std().fillna(0) * (df["session_asia"] + df["session_london"] + df["session_ny"])
    )

    vol_24h = returns.rolling(1440).std().fillna(0)
    df["session_liq_compression"] = (df["session_vol_bias"] / (vol_24h + EPS)).fillna(0)

    # HTF volatility slope
    htf_vols = {}
    for key, frame in htf.items():
        f = frame.copy()
        f["ret"] = f["close"].pct_change().fillna(0)
        f["realised_vol"] = f["ret"].rolling(5).std().fillna(0)
        htf_vols[key] = f[["open_time_ms", "realised_vol"]].rename(columns={"realised_vol": f"vol_{key}"})

    df_htf = df[["open_time_ms"]].copy()
    for key, vol_df in htf_vols.items():
        df_htf = df_htf.merge(vol_df, on="open_time_ms", how="left")
        df_htf[f"vol_{key}"] = df_htf[f"vol_{key}"].fillna(method="ffill").fillna(0)

    def slope_row(row):
        vols = []
        weights = []
        for tf, minutes in [("5m", 5), ("15m", 15), ("1h", 60), ("4h", 240)]:
            vols.append(row[f"vol_{tf}"])
            weights.append(minutes)
        x = np.array(weights, dtype=float)
        y = np.array(vols, dtype=float)
        if np.all(y == 0):
            return 0.0
        coef = np.polyfit(x, y, 1)
        return coef[0]

    df["htf_vol_slope"] = df_htf.apply(slope_row, axis=1)

    # Daily trend orientation
    daily = htf["1d"].copy()
    daily["ema_fast"] = daily["close"].ewm(span=20, adjust=False).mean()
    daily["ema_slow"] = daily["close"].ewm(span=50, adjust=False).mean()
    daily["daily_trend_orientation"] = np.sign(daily["ema_fast"] - daily["ema_slow"])
    df = df.merge(
        daily[["open_time_ms", "daily_trend_orientation"]],
        on="open_time_ms",
        how="left",
    ).fillna({"daily_trend_orientation": 0})

    # Volatility cycle phase (daily ATR z-score)
    tr_components = pd.concat(
        [
            daily["high"] - daily["low"],
            (daily["high"] - daily["close"].shift(1)).abs(),
            (daily["low"] - daily["close"].shift(1)).abs(),
        ],
        axis=1,
    )
    tr_d = tr_components.max(axis=1)
    atr_d = tr_d.rolling(14).mean()
    atr_mean = atr_d.rolling(50).mean()
    atr_std = atr_d.rolling(50).std().replace(0, np.nan)
    daily["vol_cycle_phase"] = ((atr_d - atr_mean) / (atr_std + EPS)).fillna(0)
    df = df.merge(
        daily[["open_time_ms", "vol_cycle_phase"]],
        on="open_time_ms",
        how="left",
    ).fillna({"vol_cycle_phase": 0})

    return df


def save_output(df: pd.DataFrame):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    logging.info("Saved features to %s", OUT_PATH)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    base_df, htf_dict = load_inputs()

    # Trend features
    trend = build_trend_features(base_df)
    # Volatility features
    vol = build_vol_features(trend)
    # Liquidity features
    liq = build_liquidity_features(vol)
    # Range features
    rng = build_range_features(liq)
    # Micro-behaviour features
    microb = build_microbehaviour_features(rng)
    # Session context features
    final_df = build_session_context_features(microb, htf_dict)

    # Select columns
    feature_cols = [
        "open_time_ms",
        # Trend
        "rolling_slope",
        "trend_strength",
        "higher_high_flag",
        "higher_low_flag",
        "regression_gradient",
        "st_lt_divergence",
        # Vol
        "realised_vol_5m",
        "realised_vol_15m",
        "realised_vol_1h",
        "atr_percentile",
        "compression_flag",
        "vol_clustering",
        "variance_ratio",
        # Liquidity / microstructure
        "delta_persistence_feat",
        "imbalance_cycles",
        "volume_drought",
        "wick_body_irregularity",
        "order_book_proxy",
        "cvd_slope",
        # Range / structure
        "sweep_flag",
        "range_compression",
        "mean_reversion_distance",
        "dist_prev_swing_high",
        "dist_prev_swing_low",
        "fair_value_divergence",
        # Market micro-behaviour
        "chop_index",
        "entropy_feat",
        "alternating_candles",
        "no_trend_stat_feat",
        # Session context
        "session_asia",
        "session_london",
        "session_ny",
        "session_vol_bias",
        "session_liq_compression",
        "htf_vol_slope",
        "daily_trend_orientation",
        "vol_cycle_phase",
    ]

    final_df = final_df[feature_cols].fillna(0)
    save_output(final_df)

    # Summary
    print("\n=== Raw Regime Features Summary ===")
    print(f"Rows: {len(final_df):,}")
    print(f"Total features: {len(feature_cols) - 1}")
    print(final_df.head())
    groups = {
        "Trend": [
            "rolling_slope",
            "trend_strength",
            "higher_high_flag",
            "higher_low_flag",
            "regression_gradient",
            "st_lt_divergence",
        ],
        "Volatility": [
            "realised_vol_5m",
            "realised_vol_15m",
            "realised_vol_1h",
            "atr_percentile",
            "compression_flag",
            "vol_clustering",
            "variance_ratio",
        ],
        "Liquidity": [
            "delta_persistence_feat",
            "imbalance_cycles",
            "volume_drought",
            "wick_body_irregularity",
            "order_book_proxy",
            "cvd_slope",
        ],
        "Range": [
            "sweep_flag",
            "range_compression",
            "mean_reversion_distance",
            "dist_prev_swing_high",
            "dist_prev_swing_low",
            "fair_value_divergence",
        ],
        "Micro": [
            "chop_index",
            "entropy_feat",
            "alternating_candles",
            "no_trend_stat_feat",
        ],
        "Session": [
            "session_asia",
            "session_london",
            "session_ny",
            "session_vol_bias",
            "session_liq_compression",
            "htf_vol_slope",
            "daily_trend_orientation",
            "vol_cycle_phase",
        ],
    }
    for name, cols in groups.items():
        mins = final_df[cols].min().to_dict()
        maxs = final_df[cols].max().to_dict()
        print(f"\n{name} features:")
        for c in cols:
            print(f"  {c}: min={mins[c]:.6f}, max={maxs[c]:.6f}")
    print("===================================\n")


if __name__ == "__main__":
    main()
