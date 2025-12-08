"""
Pillar 1 — Feature Engine (Raw Signal Generator)

Builds the primitive feature matrix for the entry model by merging:
- Normalised Phase 0 features (PRE_PILLARFOUR)
- Pillar 4 deterministic engines (part_two)
- Pillar 4 fused regimes (part_four)
- Pillar 4 stable regimes (part_five)

Adds leakage-free primitive feature groups:
- Momentum, Trend, Volatility, Reversion, Orderflow, Structure,
  Micro-pattern, Regime-derived.

Debug mode:
- For faster iteration, you can limit to the last N rows (default 50_000).
  Set limit=None to use the full dataset.
"""

from pathlib import Path
from typing import Optional
import logging

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]

NORM_PATH = BASE_DIR / "PRE_PILLARFOUR" / "processed" / "normalised_features.parquet"
P4_BASE = BASE_DIR / "PILLAR_FOUR_REGIME_DETECTION" / "data" / "processed"
DET_PATH = P4_BASE / "part_two" / "deterministic_regime_engines.parquet"
FUSED_PATH = P4_BASE / "part_four" / "regime_labels_fused.parquet"
STABLE_PATH = P4_BASE / "part_five" / "regime_labels_stable.parquet"

DEFAULT_OUTPUT = Path("data/processed/features_pillar1_primitives.parquet")

EPS = 1e-9


# ---------------------------------------------------------------------
# Merge utilities
# ---------------------------------------------------------------------
def _merge_by_index(df: pd.DataFrame, other: pd.DataFrame, tag: str) -> pd.DataFrame:
    """
    Fast index-based merge when upstream outputs don't carry timestamps.

    Assumes df and other refer to the same chronological timeline.

    Debug-friendly:
      - If 'other' is longer than df (e.g. full history vs limited df),
        we automatically take the last len(df) rows of 'other'.
    """
    df = df.reset_index(drop=True)
    other = other.reset_index(drop=True)

    if len(other) > len(df):
        other = other.iloc[-len(df):].reset_index(drop=True)
    elif len(other) < len(df):
        raise ValueError(f"[{tag}] other shorter than df (df={len(df)}, other={len(other)}).")

    other = other.add_prefix(f"{tag}_")
    return pd.concat([df, other], axis=1)


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def load_base(limit_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Load and merge Phase 0 + Pillar 4 outputs into a single base DataFrame.

    - normalised_features.parquet
    - deterministic_regime_engines.parquet
    - regime_labels_fused.parquet
    - regime_labels_stable.parquet

    If limit_rows is not None, only the LAST 'limit_rows' bars
    of the normalised features are kept, and all merges are
    restricted to that time range.
    """
    missing = [p for p in [NORM_PATH, DET_PATH, FUSED_PATH, STABLE_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required upstream files: {missing}")

    logging.info("Loading normalised features: %s", NORM_PATH)
    df = pd.read_parquet(NORM_PATH)
    if "open_time_ms" not in df.columns:
        raise ValueError("normalised_features.parquet missing open_time_ms")
    df = df.sort_values("open_time_ms").reset_index(drop=True)

    if limit_rows is not None and limit_rows > 0 and len(df) > limit_rows:
        logging.info("Subsetting normalised features to last %d rows (from %d)", limit_rows, len(df))
        df = df.iloc[-limit_rows:].reset_index(drop=True)
    _log_mem(df, "[Base] After normalised load (possibly limited)")

    # We'll use the min timestamp in this subset to filter Pillar 4 inputs.
    min_ts = df["open_time_ms"].iloc[0]

    logging.info("Merging deterministic regime engines: %s", DET_PATH)
    det = pd.read_parquet(DET_PATH)
    if "open_time_ms" in det.columns:
        det = det.sort_values("open_time_ms")
        det = det[det["open_time_ms"] >= min_ts]
        df = df.merge(det, on="open_time_ms", how="left", suffixes=("", "_det"))
    else:
        df = _merge_by_index(df, det, tag="det")
    _log_mem(df, "[Base] After deterministic merge")

    logging.info("Merging fused regimes: %s", FUSED_PATH)
    fused = pd.read_parquet(FUSED_PATH)
    if "open_time_ms" in fused.columns:
        fused = fused.sort_values("open_time_ms")
        fused = fused[fused["open_time_ms"] >= min_ts]
        df = df.merge(fused, on="open_time_ms", how="left", suffixes=("", "_fused"))
    else:
        logging.warning("[Base] open_time_ms missing in fused labels; aligning by index (tail-matched)")
        df = _merge_by_index(df, fused, tag="fused")
    _log_mem(df, "[Base] After fused merge")

    logging.info("Merging stable regimes: %s", STABLE_PATH)
    stable = pd.read_parquet(STABLE_PATH)
    if "open_time_ms" in stable.columns:
        stable = stable.sort_values("open_time_ms")
        stable = stable[stable["open_time_ms"] >= min_ts]
        df = df.merge(stable, on="open_time_ms", how="left", suffixes=("", "_stable"))
    else:
        logging.warning("[Base] open_time_ms missing in stable labels; aligning by index (tail-matched)")
        df = _merge_by_index(df, stable, tag="stable")
    _log_mem(df, "[Base] After stable merge")

    df = df.ffill().bfill()
    return df


# ---------------------------------------------------------------------
# Core helpers (reused across engines)
# ---------------------------------------------------------------------
def _ensure_core_series(df: pd.DataFrame) -> None:
    """
    Compute core series (returns, volatility) ONCE and reuse across engines.

    Adds:
    - ret_1m
    - vol_atr_14 (canonical volatility source)
    """
    if "ret_1m" not in df.columns:
        df["ret_1m"] = df["close"].pct_change().fillna(0)

    # Canonical volatility: prefer atr_14, then atr_14_norm, then rv_15m, then rv_5m.
    if "vol_atr_14" not in df.columns:
        vol_src = None
        for cand in ["atr_14", "atr_14_norm", "rv_15m", "rv_5m"]:
            if cand in df.columns:
                vol_src = df[cand].abs()
                break
        if vol_src is None:
            raise ValueError("No ATR/volatility source found (atr_14/atr_14_norm/rv_15m/rv_5m).")
        df["vol_atr_14"] = vol_src.fillna(0.0)


def _rolling_slope(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Efficient rolling linear regression slope using raw NumPy.

    - arr: 1D numpy array
    - window: int window size
    """
    n = len(arr)
    out = np.zeros(n, dtype=float)
    if window <= 1:
        return out

    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum() + EPS

    s = pd.Series(arr)
    out_series = s.rolling(window, min_periods=window).apply(
        lambda y: ((x - x_mean) * (y - y.mean())).sum() / denom,
        raw=False,
    )
    out[:] = out_series.to_numpy()
    np.nan_to_num(out, copy=False)
    return out


# ---------------------------------------------------------------------
# Primitive feature builders (in-place)
# ---------------------------------------------------------------------
def add_momentum_primitives(df: pd.DataFrame) -> None:
    close = df["close"].to_numpy()
    ret = df["ret_1m"].to_numpy()
    volume = df.get("volume", pd.Series(0.0, index=df.index)).to_numpy()

    # ROC
    for w in (1, 5, 15):
        roc = (close / np.roll(close, w) - 1.0)
        roc[:w] = 0.0
        df[f"mom_roc_{w}"] = roc

    # z-score of returns over 20 bars
    ret_s = pd.Series(ret)
    mean20 = ret_s.rolling(20).mean()
    std20 = ret_s.rolling(20).std()
    df["mom_ret_z_20"] = ((ret_s - mean20) / (std20 + EPS)).fillna(0.0)

    # OBV and slope over 20 bars
    price_dir = np.sign(np.diff(close, prepend=close[0]))
    obv = (price_dir * volume).cumsum()
    df["mom_obv"] = obv

    obv_slope_20 = _rolling_slope(obv, window=20)
    df["mom_obv_slope_20"] = obv_slope_20

    # Volatility-adjusted 1-bar momentum
    vol = df["vol_atr_14"].to_numpy()
    df["mom_vol_adj_1"] = ret / (vol + EPS)

    # Momentum acceleration (5-bar ROC change)
    roc_5 = df["mom_roc_5"].to_numpy()
    accel_5 = roc_5 - np.roll(roc_5, 5)
    accel_5[:5] = 0.0
    df["mom_accel_5"] = accel_5


def add_trend_primitives(df: pd.DataFrame) -> None:
    close = df["close"].to_numpy()
    ret = df["ret_1m"].to_numpy()

    # Rolling slope 20
    df["trend_slope_20"] = _rolling_slope(close, window=20)

    # Trend strength: |slope| / std(ret over 20)
    ret_s = pd.Series(ret)
    std20 = ret_s.rolling(20).std()
    df["trend_strength_20"] = (np.abs(df["trend_slope_20"].to_numpy()) / (std20.to_numpy() + EPS))

    # Efficiency ratio
    s_close = pd.Series(close)
    net_change = (s_close - s_close.shift(20)).abs()
    vol_sum = s_close.diff().abs().rolling(20).sum()
    eff = (net_change / (vol_sum + EPS)).fillna(0.0)
    df["trend_efficiency_20"] = eff
    df["trend_noise_ratio_20"] = 1.0 - eff

    # MA distance: fast vs slow
    ema_fast = s_close.ewm(span=20, adjust=False).mean()
    ema_slow = s_close.ewm(span=50, adjust=False).mean()
    df["trend_ma_dist_fast_slow"] = (ema_fast - ema_slow) / (df["vol_atr_14"] + EPS)

    # HTF trend alignment: use close_5m if available
    if "close_5m" in df.columns:
        htf_close = df["close_5m"].to_numpy()
        htf_slope_5m = _rolling_slope(htf_close, window=20)
        df["trend_htf_align_5m"] = (
            np.sign(df["trend_slope_20"].to_numpy()) * np.sign(htf_slope_5m)
        )
    else:
        df["trend_htf_align_5m"] = np.sign(df["trend_slope_20"].to_numpy())


def add_volatility_primitives(df: pd.DataFrame) -> None:
    vol = df["vol_atr_14"]
    ret_s = pd.Series(df["ret_1m"].to_numpy())

    df["vol_atr_14"] = vol  # explicit canonical name

    # ATR expansion/contraction
    vol_mean_50 = vol.rolling(50).mean()
    vol_mean_200 = vol.rolling(200).mean()
    df["vol_atr_expansion_50"] = (vol / (vol_mean_50 + EPS)).fillna(1.0)
    df["vol_atr_contraction_200"] = (vol / (vol_mean_200 + EPS)).fillna(1.0)

    # Realised vol 5m / 15m (prefer Phase 0 rv_? if available)
    if "rv_5m" in df.columns:
        df["vol_realised_5m"] = df["rv_5m"].fillna(0.0)
    else:
        df["vol_realised_5m"] = ret_s.rolling(5).std().fillna(0.0)

    if "rv_15m" in df.columns:
        df["vol_realised_15m"] = df["rv_15m"].fillna(0.0)
    else:
        df["vol_realised_15m"] = ret_s.rolling(15).std().fillna(0.0)

    # Volatility percentile over a long window
    vol_s = vol.astype(float)
    df["vol_percentile_1000"] = vol_s.rolling(1000, min_periods=50).rank(pct=True).fillna(0.5)

    # Vol clustering: ratio of fast vs slow realised vol
    rv_fast = ret_s.rolling(50).std()
    rv_slow = ret_s.rolling(200).std()
    df["vol_cluster_score_50"] = (rv_fast / (rv_slow + EPS)).fillna(1.0)


def add_reversion_primitives(df: pd.DataFrame) -> None:
    close = pd.Series(df["close"].to_numpy())

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + EPS)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0)
    df["rev_rsi_14"] = rsi

    df["rev_rsi_overbought"] = np.maximum(0.0, rsi - 70.0)
    df["rev_rsi_oversold"] = np.maximum(0.0, 30.0 - rsi)

    # Z-score deviation over 50
    mean50 = close.rolling(50).mean()
    std50 = close.rolling(50).std()
    z = (close - mean50) / (std50 + EPS)
    z = z.fillna(0.0)
    df["rev_zscore_50"] = z

    # Reversion acceleration (change in z-score over 5 bars)
    z_arr = z.to_numpy()
    accel_5 = z_arr - np.roll(z_arr, 5)
    accel_5[:5] = 0.0
    df["rev_accel_5"] = accel_5

    # Volatility-normalised mean reversion metric
    df["rev_vol_norm"] = z / (df["vol_atr_14"] + EPS)


def add_orderflow_primitives(df: pd.DataFrame) -> None:
    delta = df.get("delta", pd.Series(0.0, index=df.index)).astype(float)
    volume = df.get("volume", pd.Series(0.0, index=df.index)).astype(float)
    taker_buy = df.get("taker_buy_volume", pd.Series(0.0, index=df.index)).astype(float)
    taker_sell = df.get("taker_sell_volume", pd.Series(0.0, index=df.index)).astype(float)
    trades = df.get("agg_trades", pd.Series(0.0, index=df.index)).astype(float)

    df["of_delta"] = delta
    df["of_delta_ema_20"] = delta.ewm(span=20, adjust=False).mean()

    # Taker imbalance proxy
    denom_vol = taker_buy + taker_sell
    df["of_taker_imbalance"] = ((taker_buy - taker_sell) / (denom_vol + EPS)).fillna(0.0)

    # CVD slope 50
    cvd = delta.cumsum()
    df["of_cvd"] = cvd
    df["of_cvd_slope_50"] = (cvd.diff(50) / 50.0).fillna(0.0)

    # Sweep detection: large trades vs rolling median
    roll_tr = trades.rolling(50, min_periods=10)
    med_tr = roll_tr.median()
    df["of_sweep_flag"] = (trades > med_tr * 3.0).astype(int)

    # Absorption: large trades, small net delta
    df["of_absorption_flag"] = (
        (trades > med_tr * 2.0) & (delta.abs() < volume * 0.01)
    ).astype(int)


def add_structure_primitives(df: pd.DataFrame) -> None:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_high_20 = high.shift(1).rolling(20).max()
    prev_low_20 = low.shift(1).rolling(20).min()

    df["struct_swing_high_flag_20"] = (high > prev_high_20).astype(int)
    df["struct_swing_low_flag_20"] = (low < prev_low_20).astype(int)

    # Range width 50
    df["struct_range_width_50"] = (
        high.rolling(50).max() - low.rolling(50).min()
    ).fillna(0.0)

    # BOS / CHOCH proxies
    df["struct_bos_flag"] = (
        (df["struct_swing_high_flag_20"] == 1) & (close > close.shift(1))
    ).astype(int)

    df["struct_choch_flag"] = (
        (df["struct_swing_low_flag_20"] == 1) & (close < close.shift(1))
    ).astype(int)

    # Liquidity sweep: take out prior extremes with quick rejection
    df["struct_liquidity_sweep_flag"] = (
        (high > prev_high_20 * 1.001) | (low < prev_low_20 * 0.999)
    ).astype(int)


def add_pattern_primitives(df: pd.DataFrame) -> None:
    o = df["open"].to_numpy()
    c = df["close"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()

    body = np.abs(c - o)
    rng = np.maximum(h - l, EPS)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l

    body_ratio = body / rng
    upper_ratio = upper_wick / rng
    lower_ratio = lower_wick / rng

    # Pin bars
    pin_bull = (lower_ratio > 0.6) & (body_ratio < 0.2)
    pin_bear = (upper_ratio > 0.6) & (body_ratio < 0.2)
    df["pat_pin_bar_bull"] = pin_bull.astype(int)
    df["pat_pin_bar_bear"] = pin_bear.astype(int)

    # Engulfing patterns
    prev_o = np.roll(o, 1)
    prev_c = np.roll(c, 1)
    prev_body = np.abs(prev_c - prev_o)
    prev_dir = np.sign(prev_c - prev_o)
    curr_dir = np.sign(c - o)

    engulfing_bull = (curr_dir > 0) & (prev_dir < 0) & (body > prev_body)
    engulfing_bear = (curr_dir < 0) & (prev_dir > 0) & (body > prev_body)
    engulfing_bull[0] = False
    engulfing_bear[0] = False
    df["pat_engulfing_bull"] = engulfing_bull.astype(int)
    df["pat_engulfing_bear"] = engulfing_bear.astype(int)

    # Momentum bar: body >> median body
    body_s = pd.Series(body)
    med_body_20 = body_s.rolling(20).median()
    df["pat_momentum_bar"] = (body_s > med_body_20 * 2.0).astype(int)

    # Pattern cluster score over last 10 bars
    pattern_hits = (
        df["pat_pin_bar_bull"]
        + df["pat_pin_bar_bear"]
        + df["pat_engulfing_bull"]
        + df["pat_engulfing_bear"]
        + df["pat_momentum_bar"]
    )
    df["pat_pattern_cluster_10"] = (
        pattern_hits.rolling(10).sum().astype(float).fillna(0.0)
    )

def _drop_constant_numeric_features(df: pd.DataFrame, std_eps: float = 1e-12) -> pd.DataFrame:
    """
    Drop constant (or numerically almost-constant) numeric feature columns.

    Rationale:
    - Constant columns carry no information for ML.
    - They usually indicate dead inputs, missing upstream data, or
      placeholder fields that can safely be removed.
    """
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        logging.info("[P1] No numeric columns found when checking for constants.")
        return df

    std = numeric.std()
    const_cols = std[std <= std_eps].index.tolist()

    if const_cols:
        logging.warning(
            "[P1] Dropping %d constant numeric columns: %s",
            len(const_cols),
            const_cols,
        )
        df = df.drop(columns=const_cols)

    return df

def _coerce_regime_id(series: pd.Series) -> pd.Series:
    """
    Convert mixed regime labels (e.g. '2.0', 'volatility_expansion') into
    canonical integer regime IDs 0..7.

    Mapping (from your pillar 4 plan):
        0: trend_up
        1: trend_down
        2: chop_low_vol
        3: chop_high_vol
        4: volatility_expansion
        5: volatility_compression
        6: liquidity_drought
        7: mean_reversion
    """
    name_to_id = {
        "trend_up": 0,
        "trend_down": 1,
        "chop_low_vol": 2,
        "chop_high_vol": 3,
        "volatility_expansion": 4,
        "vol_expansion": 4,        # just in case
        "volatility_compression": 5,
        "vol_compression": 5,      # just in case
        "liquidity_drought": 6,
        "mean_reversion": 7,
    }

    s = series.astype(str)

    def parse(v: str) -> int:
        v = v.strip()
        if v in name_to_id:
            return name_to_id[v]
        # Try numeric-ish strings like "2", "3.0", etc.
        try:
            return int(float(v))
        except Exception:
            # Unknown label → treat as neutral / fallback bucket
            return -1

    return s.map(parse).astype(int)

def add_regime_primitives(df: pd.DataFrame) -> None:
    """
    Regime-derived primitives (from Pillar 4):
    - regime_id_stable (regime_final)
    - regime_confidence
    - regime_is_trend
    - regime_is_chop
    - regime_is_liq_drought
    - regime_is_vol_expansion
    - regime_is_mean_reversion
    - regime_hold_length_norm
    - regime_chop_mask
    """

    # ------------------------------------------------------------------
    # regime_final id (handle direct + index-merged cases)
    # ------------------------------------------------------------------
    if "regime_label_stable" in df.columns:
        src = df["regime_label_stable"]
    elif "stable_regime_label_stable" in df.columns:  # index-merged suffix case
        src = df["stable_regime_label_stable"]
    elif "regime_label_fused" in df.columns:
        # Fallback: use fused label if for some reason stable not present
        src = df["regime_label_fused"]
    else:
        src = pd.Series(-1, index=df.index)

    df["regime_id_stable"] = _coerce_regime_id(src)

    # ------------------------------------------------------------------
    # regime_confidence (prefer existing, then ml_confidence variants)
    # ------------------------------------------------------------------
    for cand in ["regime_confidence", "ml_confidence", "fused_ml_confidence", "stable_ml_confidence"]:
        if cand in df.columns:
            df["regime_confidence"] = (
                pd.to_numeric(df[cand], errors="coerce")
                .fillna(0.0)
                .astype(float)
            )
            break

    # ------------------------------------------------------------------
    # deterministic flags
    # ------------------------------------------------------------------
    trend_flag = df.get("trend_flag", pd.Series(0, index=df.index)).astype(int)
    chop_flag = df.get("chop_flag", pd.Series(0, index=df.index)).astype(int)
    mean_rev_flag = df.get("mean_reversion_flag", pd.Series(0, index=df.index)).astype(int)
    vol_regime = df.get("volatility_regime", pd.Series("normal", index=df.index)).astype(str)
    liq_regime = df.get("liquidity_regime", pd.Series("normal", index=df.index)).astype(str)

    df["regime_is_trend"] = (trend_flag == 1).astype(int)
    df["regime_is_chop"] = (chop_flag == 1).astype(int)
    df["regime_is_mean_reversion"] = (mean_rev_flag == 1).astype(int)
    df["regime_is_vol_expansion"] = (vol_regime == "expansion").astype(int)
    df["regime_is_liq_drought"] = (liq_regime == "drought").astype(int)

    # ------------------------------------------------------------------
    # Hold length normalised (use stable_hold_length when index-merged)
    # ------------------------------------------------------------------
    hold_len = None
    for hc in ["hold_length", "stable_hold_length"]:
        if hc in df.columns:
            hold_len = pd.to_numeric(df[hc], errors="coerce").fillna(0.0)
            break

    if hold_len is None:
        hold_len = pd.Series(0.0, index=df.index)

    max_hold = (
        hold_len.rolling(2000, min_periods=50)
        .max()
        .replace(0, np.nan)
    )
    df["regime_hold_length_norm"] = (
        (hold_len / (max_hold + EPS))
        .clip(lower=0.0, upper=1.0)
        .fillna(0.0)
    )

    # ------------------------------------------------------------------
    # Chop mask: down-weight momentum during chop
    # ------------------------------------------------------------------
    chop_score = df.get("chop_score", pd.Series(0.0, index=df.index)).astype(float)
    df["regime_chop_mask"] = (1.0 - chop_score.clip(0.0, 1.0))

# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------
def build_pillar1_feature_matrix(
    save_path: Optional[Path] = DEFAULT_OUTPUT,
    limit: Optional[int] = 50_000,
) -> pd.DataFrame:
    """
    High-level builder for Pillar 1 primitive feature matrix.

    Steps:
    1) Load Phase 0 + Pillar 4 base dataset (optionally limited).
    2) Compute core series (returns, volatility) once.
    3) Apply all primitive feature engines, in-place:
       - Momentum, Trend, Volatility, Reversion,
         Orderflow, Structure, Micro-pattern, Regime-derived.
    4) Clean NaNs/inf, drop constant numeric columns, save parquet, return df.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.info("Building Pillar 1 primitive feature matrix (limit=%s)...", limit)

    # ✅ FIX: use limit_rows keyword to match load_base definition
    df = load_base(limit_rows=limit)
    _ensure_core_series(df)
    _log_mem(df, "[P1] After base + core series")

    # Primitive engines (in-place)
    add_momentum_primitives(df)
    add_trend_primitives(df)
    add_volatility_primitives(df)
    add_reversion_primitives(df)
    add_orderflow_primitives(df)
    add_structure_primitives(df)
    add_pattern_primitives(df)
    add_regime_primitives(df)

    # Final clean-up: no NaN/inf
    df.replace([np.inf, -np.inf], 0.0, inplace=True)
    df.fillna(0.0, inplace=True)

    # Drop constant numeric features (e.g. rv_5m_reg, of_taker_imbalance)
    df = _drop_constant_numeric_features(df)

    # Ensure sorted by time
    if "open_time_ms" in df.columns:
        df.sort_values("open_time_ms", inplace=True)
        df.reset_index(drop=True, inplace=True)

    _log_mem(df, "[P1] Final primitive feature matrix")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path, index=False)
        logging.info("Saved Pillar 1 primitive feature matrix -> %s", save_path)

    return df


if __name__ == "__main__":
    build_pillar1_feature_matrix()