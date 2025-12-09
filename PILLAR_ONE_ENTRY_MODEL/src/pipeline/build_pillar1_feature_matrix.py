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

Streaming mode:
- Phase 0 + Pillar 4 base is loaded once.
- Primitive features are computed in overlapping chunks and streamed
  to parquet, so we never materialise the full primitive matrix in RAM.
"""

from pathlib import Path
from typing import Optional
import logging

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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

# Max lookback across all rolling windows (RSI 14, rolling 20/50/200/1000,
# hold_length 2000, etc.). We use this for chunk overlap so windowed
# features are correct at chunk boundaries.
MAX_LOOKBACK = 2000

# Reasonable default chunk size for a 1.5M row dataset on a laptop
DEFAULT_CHUNK_SIZE = 200_000


# ---------------------------------------------------------------------
# Merge utilities
# ---------------------------------------------------------------------
def _merge_by_index(df: pd.DataFrame, other: pd.DataFrame, tag: str) -> pd.DataFrame:
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


def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggressively downcast numeric columns to reduce memory footprint.
    Keeps semantics the same for this pipeline.
    """
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    if not len(num_cols):
        return df

    for col in num_cols:
        s = df[col]
        if pd.api.types.is_float_dtype(s):
            df[col] = pd.to_numeric(s, downcast="float")
        elif pd.api.types.is_integer_dtype(s):
            df[col] = pd.to_numeric(s, downcast="integer")
    return df


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

    # Shrink memory before primitive feature computation
    df = _downcast_numeric(df)
    _log_mem(df, "[Base] After downcast_numeric")

    return df


# ---------------------------------------------------------------------
# Core helpers (reused across engines)
# ---------------------------------------------------------------------
def _ensure_core_series(df: pd.DataFrame) -> None:
    if "ret_1m" not in df.columns:
        df["ret_1m"] = df["close"].pct_change().fillna(0)

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

    for w in (1, 5, 15):
        roc = (close / np.roll(close, w) - 1.0)
        roc[:w] = 0.0
        df[f"mom_roc_{w}"] = roc

    ret_s = pd.Series(ret)
    mean20 = ret_s.rolling(20).mean()
    std20 = ret_s.rolling(20).std()
    df["mom_ret_z_20"] = ((ret_s - mean20) / (std20 + EPS)).fillna(0.0)

    price_dir = np.sign(np.diff(close, prepend=close[0]))
    obv = (price_dir * volume).cumsum()
    df["mom_obv"] = obv

    obv_slope_20 = _rolling_slope(obv, window=20)
    df["mom_obv_slope_20"] = obv_slope_20

    vol = df["vol_atr_14"].to_numpy()
    df["mom_vol_adj_1"] = ret / (vol + EPS)

    roc_5 = df["mom_roc_5"].to_numpy()
    accel_5 = roc_5 - np.roll(roc_5, 5)
    accel_5[:5] = 0.0
    df["mom_accel_5"] = accel_5


def add_trend_primitives(df: pd.DataFrame) -> None:
    close = df["close"].to_numpy()
    ret = df["ret_1m"].to_numpy()

    df["trend_slope_20"] = _rolling_slope(close, window=20)

    ret_s = pd.Series(ret)
    std20 = ret_s.rolling(20).std()
    df["trend_strength_20"] = (np.abs(df["trend_slope_20"].to_numpy()) / (std20.to_numpy() + EPS))

    s_close = pd.Series(close)
    net_change = (s_close - s_close.shift(20)).abs()
    vol_sum = s_close.diff().abs().rolling(20).sum()
    eff = (net_change / (vol_sum + EPS)).fillna(0.0)
    df["trend_efficiency_20"] = eff
    df["trend_noise_ratio_20"] = 1.0 - eff

    ema_fast = s_close.ewm(span=20, adjust=False).mean()
    ema_slow = s_close.ewm(span=50, adjust=False).mean()
    df["trend_ma_dist_fast_slow"] = (ema_fast - ema_slow) / (df["vol_atr_14"] + EPS)

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

    df["vol_atr_14"] = vol

    vol_mean_50 = vol.rolling(50).mean()
    vol_mean_200 = vol.rolling(200).mean()
    df["vol_atr_expansion_50"] = (vol / (vol_mean_50 + EPS)).fillna(1.0)
    df["vol_atr_contraction_200"] = (vol / (vol_mean_200 + EPS)).fillna(1.0)

    if "rv_5m" in df.columns:
        df["vol_realised_5m"] = df["rv_5m"].fillna(0.0)
    else:
        df["vol_realised_5m"] = ret_s.rolling(5).std().fillna(0.0)

    if "rv_15m" in df.columns:
        df["vol_realised_15m"] = df["rv_15m"].fillna(0.0)
    else:
        df["vol_realised_15m"] = ret_s.rolling(15).std().fillna(0.0)

    vol_s = vol.astype(float)
    df["vol_percentile_1000"] = vol_s.rolling(1000, min_periods=50).rank(pct=True).fillna(0.5)

    rv_fast = ret_s.rolling(50).std()
    rv_slow = ret_s.rolling(200).std()
    df["vol_cluster_score_50"] = (rv_fast / (rv_slow + EPS)).fillna(1.0)


def add_reversion_primitives(df: pd.DataFrame) -> None:
    close = pd.Series(df["close"].to_numpy())

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + EPS)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0)
    df["rev_rsi_14"] = rsi

    df["rev_rsi_overbought"] = np.maximum(0.0, rsi - 70.0)
    df["rev_rsi_oversold"] = np.maximum(0.0, 30.0 - rsi)

    mean50 = close.rolling(50).mean()
    std50 = close.rolling(50).std()
    z = (close - mean50) / (std50 + EPS)
    z = z.fillna(0.0)
    df["rev_zscore_50"] = z

    z_arr = z.to_numpy()
    accel_5 = z_arr - np.roll(z_arr, 5)
    accel_5[:5] = 0.0
    df["rev_accel_5"] = accel_5

    df["rev_vol_norm"] = z / (df["vol_atr_14"] + EPS)


def add_orderflow_primitives(df: pd.DataFrame) -> None:
    delta = df.get("delta", pd.Series(0.0, index=df.index)).astype(float)
    volume = df.get("volume", pd.Series(0.0, index=df.index)).astype(float)
    taker_buy = df.get("taker_buy_volume", pd.Series(0.0, index=df.index)).astype(float)
    taker_sell = df.get("taker_sell_volume", pd.Series(0.0, index=df.index)).astype(float)
    trades = df.get("agg_trades", pd.Series(0.0, index=df.index)).astype(float)

    df["of_delta"] = delta
    df["of_delta_ema_20"] = delta.ewm(span=20, adjust=False).mean()

    denom_vol = taker_buy + taker_sell
    df["of_taker_imbalance"] = ((taker_buy - taker_sell) / (denom_vol + EPS)).fillna(0.0)

    cvd = delta.cumsum()
    df["of_cvd"] = cvd
    df["of_cvd_slope_50"] = (cvd.diff(50) / 50.0).fillna(0.0)

    roll_tr = trades.rolling(50, min_periods=10)
    med_tr = roll_tr.median()
    df["of_sweep_flag"] = (trades > med_tr * 3.0).astype(int)

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

    df["struct_range_width_50"] = (
        high.rolling(50).max() - low.rolling(50).min()
    ).fillna(0.0)

    df["struct_bos_flag"] = (
        (df["struct_swing_high_flag_20"] == 1) & (close > close.shift(1))
    ).astype(int)

    df["struct_choch_flag"] = (
        (df["struct_swing_low_flag_20"] == 1) & (close < close.shift(1))
    ).astype(int)

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

    pin_bull = (lower_ratio > 0.6) & (body_ratio < 0.2)
    pin_bear = (upper_ratio > 0.6) & (body_ratio < 0.2)
    df["pat_pin_bar_bull"] = pin_bull.astype(int)
    df["pat_pin_bar_bear"] = pin_bear.astype(int)

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

    body_s = pd.Series(body)
    med_body_20 = body_s.rolling(20).median()
    df["pat_momentum_bar"] = (body_s > med_body_20 * 2.0).astype(int)

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


def _coerce_regime_id(series: pd.Series) -> pd.Series:
    name_to_id = {
        "trend_up": 0,
        "trend_down": 1,
        "chop_low_vol": 2,
        "chop_high_vol": 3,
        "volatility_expansion": 4,
        "vol_expansion": 4,
        "volatility_compression": 5,
        "vol_compression": 5,
        "liquidity_drought": 6,
        "mean_reversion": 7,
    }

    s = series.astype(str)

    def parse(v: str) -> int:
        v = v.strip()
        if v in name_to_id:
            return name_to_id[v]
        try:
            return int(float(v))
        except Exception:
            return -1

    return s.map(parse).astype(int)


def add_regime_primitives(df: pd.DataFrame) -> None:
    if "regime_label_stable" in df.columns:
        src = df["regime_label_stable"]
    elif "stable_regime_label_stable" in df.columns:
        src = df["stable_regime_label_stable"]
    elif "regime_label_fused" in df.columns:
        src = df["regime_label_fused"]
    else:
        src = pd.Series(-1, index=df.index)

    df["regime_id_stable"] = _coerce_regime_id(src)

    for cand in ["regime_confidence", "ml_confidence", "fused_ml_confidence", "stable_ml_confidence"]:
        if cand in df.columns:
            df["regime_confidence"] = (
                pd.to_numeric(df[cand], errors="coerce")
                .fillna(0.0)
                .astype(float)
            )
            break

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

    chop_score = df.get("chop_score", pd.Series(0.0, index=df.index)).astype(float)
    df["regime_chop_mask"] = (1.0 - chop_score.clip(0.0, 1.0))


# ---------------------------------------------------------------------
# Chunked orchestration
# ---------------------------------------------------------------------
def _compute_primitives_on_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all primitive engines in-place to a single chunk.
    """
    add_momentum_primitives(chunk)
    add_trend_primitives(chunk)
    add_volatility_primitives(chunk)
    add_reversion_primitives(chunk)
    add_orderflow_primitives(chunk)
    add_structure_primitives(chunk)
    add_pattern_primitives(chunk)
    add_regime_primitives(chunk)

    chunk.replace([np.inf, -np.inf], 0.0, inplace=True)
    chunk.fillna(0.0, inplace=True)
    return chunk


def build_pillar1_feature_matrix(
    save_path: Optional[Path] = DEFAULT_OUTPUT,
    limit: Optional[int] = None,       # None => full history
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = MAX_LOOKBACK,
) -> pd.DataFrame:
    """
    High-level builder for Pillar 1 primitive feature matrix (chunked).

    Steps:
    1) Load Phase 0 + Pillar 4 base dataset (optionally limited) into memory.
    2) Compute core series (returns, volatility) once on the full base.
    3) Stream over the base in overlapping chunks:
       - For each chunk, slice [start_overlap:end)
       - Compute all primitive features on the chunk
       - Drop the leading overlap rows (except for the first chunk)
       - Append to a parquet file via pyarrow.ParquetWriter
    4) Return a lightweight DataFrame with only the last chunk (for debug),
       while the full primitive matrix is persisted on disk.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.info(
        "Building Pillar 1 primitive feature matrix (limit=%s, chunk_size=%d, overlap=%d)...",
        limit,
        chunk_size,
        overlap,
    )

    base = load_base(limit_rows=limit)
    if "open_time_ms" in base.columns:
        base = base.sort_values("open_time_ms").reset_index(drop=True)

    _ensure_core_series(base)
    _log_mem(base, "[P1] Base after core series")

    n = len(base)
    if n == 0:
        raise ValueError("No rows in base dataset after load_base().")

    if overlap < 0:
        overlap = 0
    if overlap > n:
        overlap = n

    if save_path is None:
        save_path = DEFAULT_OUTPUT
    save_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    schema = None
    last_chunk_df = None

    for chunk_idx, start in enumerate(range(0, n, chunk_size)):
        end = min(n, start + chunk_size)
        chunk_start = max(0, start - overlap)

        logging.info(
            "[P1] Processing chunk %d: global rows [%d, %d), with warmup [%d, %d)",
            chunk_idx,
            start,
            end,
            chunk_start,
            start,
        )

        # Slice base with overlap and copy so we don't mutate the big base df
        chunk = base.iloc[chunk_start:end].copy()

        # Compute primitives on this chunk
        chunk = _compute_primitives_on_chunk(chunk)

        # Drop warmup / overlap rows at the front (except for the very first chunk)
        if chunk_idx > 0:
            trim = start - chunk_start
            if trim > 0:
                chunk = chunk.iloc[trim:].reset_index(drop=True)
        else:
            chunk = chunk.reset_index(drop=True)

        if "open_time_ms" in chunk.columns:
            chunk = chunk.sort_values("open_time_ms").reset_index(drop=True)

        _log_mem(chunk, f"[P1] Chunk {chunk_idx} after primitives")

        # Initialise writer on first chunk with its schema
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(str(save_path), schema=schema, compression="snappy")
        else:
            # Ensure consistent schema across chunks
            table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)

        writer.write_table(table)
        last_chunk_df = chunk  # keep last one around for return/debug

    if writer is not None:
        writer.close()
        logging.info("Saved Pillar 1 primitive feature matrix (streamed) -> %s", save_path)

    # Return last chunk for interactive inspection; full dataset is on disk.
    return last_chunk_df if last_chunk_df is not None else base


if __name__ == "__main__":
    # Full history by default; set limit=50_000 if you want debug mode again.
    build_pillar1_feature_matrix(limit=None)