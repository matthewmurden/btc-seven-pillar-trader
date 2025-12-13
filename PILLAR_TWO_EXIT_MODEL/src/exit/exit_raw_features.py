"""
Pillar 2 — Raw Exit Feature Generator

This module builds leakage-free, post-entry *input features* for the Pillar 2
exit models. It uses only information available at the time of entry and
summarises the recent volatility, excursions, orderflow, structure and regime
context.

The typical workflow is:

1. Start from a time-series DataFrame that already contains:
   - bar-level OHLCV
   - Pillar 1 outputs (entry_decision, entry_direction, entry_confidence, etc.)
   - Pillar 4 regime labels and flags
   - normalised primitive features (volatility, orderflow, structure, etc.)

2. Call `build_exit_raw_features(df, cfg)` to produce a row-per-entry DataFrame
   of exit-focused features.

3. Save that as `exit_raw.parquet` and feed it into:
   - mini exit engines (MFE/MAE engines, path shape engines, etc.)
   - exit label builder (TP/SL labels)
   - Pillar 2 ML models.

All features are constructed so they *only* use past or current bar data, never
future bars, so there is no label leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ExitRawConfig:
    """
    Configuration for the raw exit feature generator.

    All windows are in bars, relative to the entry bar.
    """

    # Volatility / distribution windows
    rv_windows: Tuple[int, ...] = (5, 10, 20, 30)
    range_windows: Tuple[int, ...] = (5, 10, 20)

    # Excursion / path features
    excursion_lookback: int = 50
    path_curvature_window: int = 20

    # Orderflow / liquidity features
    orderflow_window: int = 20
    micro_imbalance_window: int = 20

    # Reversion / structure context
    reversion_window: int = 50

    # Column names (so we can flexibly adapt to upstream changes)
    col_time: str = "open_time_ms"
    col_close: str = "close"
    col_high: str = "high"
    col_low: str = "low"
    col_volume: str = "volume"

    col_entry_decision: str = "entry_decision"       # usually 'long'/'short'/'none'
    col_entry_direction: str = "entry_direction"     # usually +1 / -1 / 0
    col_entry_confidence: str = "entry_confidence"   # scalar confidence
    col_entry_prob: str = "entry_prob_up"            # calibrated long probability if available
    col_valid_flag: str = "signal_is_valid"

    # Optional existing columns
    col_atr: str = "vol_atr_14"
    col_zscore: str = "rev_zscore_50"

    # Orderflow / microstructure (optional, used if present)
    col_of_delta: str = "of_delta"
    col_of_delta_ema: str = "of_delta_ema_20"
    col_of_imbalance: str = "of_imbalance"
    col_of_sweep_flag: str = "of_sweep_flag"

    # Structure (optional)
    col_fair_value: str = "struct_fair_value"
    col_nearest_level_dist: str = "struct_nearest_level_dist"
    col_last_sweep_dir: str = "struct_last_sweep_dir"

    # Regime / liquidity flags (optional)
    col_regime_id: str = "regime_id_stable"
    col_regime_conf: str = "regime_confidence"
    col_regime_vol_bucket: str = "regime_vol_bucket"
    col_regime_liq_bucket: str = "regime_liq_bucket"
    col_regime_trend_flag: str = "regime_is_trending"
    col_regime_chop_flag: str = "regime_is_choppy"
    col_regime_liq_drought_flag: str = "regime_is_liq_drought"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_cols(df: pd.DataFrame, cols: Sequence[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[ExitRaw] Missing required columns for {context}: {missing}")


def _safe_series(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    """Return column if present, else None."""
    return df[col] if col in df.columns else None


def _compute_returns(close: pd.Series) -> pd.Series:
    """Simple percentage returns, 1-bar lagged to avoid any sneaky future info."""
    ret = close.pct_change().fillna(0.0)
    return ret.replace([np.inf, -np.inf], 0.0)


def _compute_vol_scale(df: pd.DataFrame, cfg: ExitRawConfig) -> pd.Series:
    """
    Volatility scale used to normalise excursions.

    Prefer ATR / close if available, otherwise fall back to rolling std of returns.
    """
    close = pd.to_numeric(df[cfg.col_close], errors="coerce").abs()
    atr = _safe_series(df, cfg.col_atr)

    if atr is not None:
        vol_scale = (atr / (close + 1e-9)).astype(float)
    else:
        ret = _compute_returns(close)
        vol_scale = ret.rolling(20, min_periods=5).std().astype(float)

    vol_scale = vol_scale.replace([np.inf, -np.inf], np.nan)
    # Clip to avoid crazy spikes
    vol_scale = vol_scale.clip(lower=1e-5, upper=0.5)
    return vol_scale.fillna(vol_scale.median())


def _volatility_features(df: pd.DataFrame, cfg: ExitRawConfig) -> pd.DataFrame:
    """
    Realised volatility & range features over past windows.
    """
    out = pd.DataFrame(index=df.index)
    close = pd.to_numeric(df[cfg.col_close], errors="coerce")
    high = pd.to_numeric(df[cfg.col_high], errors="coerce")
    low = pd.to_numeric(df[cfg.col_low], errors="coerce")

    ret = _compute_returns(close)

    # Realised vol: std of returns over rolling windows
    for w in cfg.rv_windows:
        rv = ret.rolling(w, min_periods=max(2, w // 2)).std()
        out[f"exit_rv_{w}"] = rv.astype("float32")

    # Range-over-ATR (or range-over-close) style metrics
    atr = _safe_series(df, cfg.col_atr)
    for w in cfg.range_windows:
        roll_high = high.rolling(w, min_periods=max(2, w // 2)).max()
        roll_low = low.rolling(w, min_periods=max(2, w // 2)).min()
        rng = (roll_high - roll_low).abs()

        if atr is not None:
            denom = atr
        else:
            denom = close * 1e-2  # ~1% of price, rough fallback

        feat = rng / (denom + 1e-9)
        out[f"exit_range_norm_{w}"] = feat.astype("float32")

    # Volatility expansion: how "hot" current vol is vs recent median
    base_rv = out.get("exit_rv_20", ret.abs())
    rv_med_100 = base_rv.rolling(100, min_periods=20).median()
    vol_expansion = base_rv / (rv_med_100 + 1e-9)
    out["exit_vol_expansion_20_100"] = vol_expansion.clip(0.0, 10.0).astype("float32")

    # Volatility clustering: ratio of short-term vs long-term RV
    rv_short = out.get("exit_rv_5", base_rv)
    rv_long = out.get("exit_rv_30", base_rv)
    cluster = rv_short / (rv_long + 1e-9)
    out["exit_vol_cluster_ratio"] = cluster.clip(0.0, 10.0).astype("float32")

    return out


def _excursion_and_path_features(df: pd.DataFrame, cfg: ExitRawConfig, vol_scale: pd.Series) -> pd.DataFrame:
    """
    Excursion-focused, leakage-free features:

    - Pre-entry MFE/MAE over a lookback window
    - Normalised excursions (range / vol_scale)
    - Path curvature: compares net move vs cumulative abs move (how "wiggly" path was)
    """
    out = pd.DataFrame(index=df.index)
    close = pd.to_numeric(df[cfg.col_close], errors="coerce")
    high = pd.to_numeric(df[cfg.col_high], errors="coerce")
    low = pd.to_numeric(df[cfg.col_low], errors="coerce")

    lb = cfg.excursion_lookback
    # Pre-entry extremums
    roll_high = high.rolling(lb, min_periods=5).max()
    roll_low = low.rolling(lb, min_periods=5).min()

    pre_mfe = (roll_high / (close + 1e-9) - 1.0)   # how far price *could* have gone up recently
    pre_mae = (roll_low / (close + 1e-9) - 1.0)    # how far price *could* have gone down recently

    out["exit_pre_mfe_raw"] = pre_mfe.astype("float32")
    out["exit_pre_mae_raw"] = pre_mae.astype("float32")

    out["exit_pre_mfe_norm"] = (pre_mfe / (vol_scale + 1e-9)).clip(-10, 10).astype("float32")
    out["exit_pre_mae_norm"] = (pre_mae / (vol_scale + 1e-9)).clip(-10, 10).astype("float32")

    # Path curvature (past N bars): cumulative abs returns vs net displacement
    w = cfg.path_curvature_window
    ret = _compute_returns(close)
    cum_abs = ret.abs().rolling(w, min_periods=3).sum()
    net = (close / close.shift(w) - 1.0).abs()
    curvature = (cum_abs - net) / (cum_abs + 1e-9)
    out[f"exit_path_curvature_{w}"] = curvature.clip(0.0, 1.0).astype("float32")

    return out


def _orderflow_liquidity_features(df: pd.DataFrame, cfg: ExitRawConfig) -> pd.DataFrame:
    """
    Orderflow and liquidity-related context.

    These features only get created if the relevant columns are present.
    """
    out = pd.DataFrame(index=df.index)
    w = cfg.orderflow_window

    of_delta = _safe_series(df, cfg.col_of_delta)
    of_delta_ema = _safe_series(df, cfg.col_of_delta_ema)
    of_imbalance = _safe_series(df, cfg.col_of_imbalance)
    of_sweep = _safe_series(df, cfg.col_of_sweep_flag)

    if of_delta_ema is not None:
        out["exit_of_trend_ema"] = of_delta_ema.astype("float32")

    if of_delta is not None:
        # Approximate "signed flow vs price" absorption score
        close = pd.to_numeric(df[cfg.col_close], errors="coerce")
        ret = _compute_returns(close)
        absorption = (np.sign(of_delta) * np.sign(ret) < 0).astype(float)
        out[f"exit_absorption_rate_{w}"] = (
            absorption.rolling(w, min_periods=3).mean().astype("float32")
        )

    if of_imbalance is not None:
        out[f"exit_micro_imbalance_mean_{w}"] = (
            of_imbalance.rolling(w, min_periods=3).mean().astype("float32")
        )

    if of_sweep is not None:
        sweeps = of_sweep.astype(float)
        out[f"exit_sweep_freq_{w}"] = (
            sweeps.rolling(w, min_periods=3).mean().astype("float32")
        )

    # Liquidity drought flag simply wired through from regime if present
    liq_flag = _safe_series(df, cfg.col_regime_liq_drought_flag)
    if liq_flag is not None:
        out["exit_is_liq_drought"] = liq_flag.astype("int8")

    return out


def _structure_reversion_features(df: pd.DataFrame, cfg: ExitRawConfig, vol_scale: pd.Series) -> pd.DataFrame:
    """
    Structure + reversion context around the entry point.
    """
    out = pd.DataFrame(index=df.index)
    close = pd.to_numeric(df[cfg.col_close], errors="coerce")

    fair = _safe_series(df, cfg.col_fair_value)
    lvl_dist = _safe_series(df, cfg.col_nearest_level_dist)
    last_sweep_dir = _safe_series(df, cfg.col_last_sweep_dir)
    zscore = _safe_series(df, cfg.col_zscore)

    if fair is not None:
        fv_dev = (close - fair) / (close + 1e-9)
        out["exit_fair_value_deviation"] = fv_dev.clip(-5, 5).astype("float32")
        out["exit_fair_value_dev_norm"] = (
            fv_dev / (vol_scale + 1e-9)
        ).clip(-10, 10).astype("float32")

    if lvl_dist is not None:
        # Distance to nearest structural level, normalised by vol scale
        dist_norm = lvl_dist / (close * vol_scale + 1e-9)
        out["exit_nearest_level_dist_norm"] = dist_norm.clip(-10, 10).astype("float32")

    if last_sweep_dir is not None:
        out["exit_prior_sweep_dir"] = last_sweep_dir.astype("int8")

    if zscore is not None:
        z = pd.to_numeric(zscore, errors="coerce")
        out["exit_reversion_zscore"] = z.astype("float32")
        out["exit_reversion_extreme_flag"] = (z.abs() >= 1.5).astype("int8")

    return out


def _regime_features(df: pd.DataFrame, cfg: ExitRawConfig) -> pd.DataFrame:
    """
    Regime-derived context copied / lightly transformed from Pillar 4.
    """
    out = pd.DataFrame(index=df.index)

    for col, out_name in [
        (cfg.col_regime_id, "exit_regime_id"),
        (cfg.col_regime_conf, "exit_regime_confidence"),
        (cfg.col_regime_vol_bucket, "exit_vol_regime"),
        (cfg.col_regime_liq_bucket, "exit_liq_regime"),
        (cfg.col_regime_trend_flag, "exit_regime_is_trending"),
        (cfg.col_regime_chop_flag, "exit_regime_is_choppy"),
    ]:
        s = _safe_series(df, col)
        if s is not None:
            out[out_name] = s

    # Normalise confidence to [0,1] if numeric
    if "exit_regime_confidence" in out.columns:
        conf = pd.to_numeric(out["exit_regime_confidence"], errors="coerce")
        out["exit_regime_confidence"] = conf.clip(0.0, 1.0).astype("float32")

    return out


def _entry_mask(df: pd.DataFrame, cfg: ExitRawConfig) -> pd.Series:
    """
    Construct a boolean mask for rows that correspond to *actual entries*.

    We prioritise:
      - valid signal flag, if available
      - entry_decision being 'long'/'short', if available
      - else, non-zero entry_direction.
    """
    mask = pd.Series(True, index=df.index)

    # Validity flag
    if cfg.col_valid_flag in df.columns:
        mask &= df[cfg.col_valid_flag].astype(bool)

    if cfg.col_entry_decision in df.columns:
        dec = df[cfg.col_entry_decision].astype(str)
        mask &= dec.isin(["long", "short"])
    elif cfg.col_entry_direction in df.columns:
        mask &= df[cfg.col_entry_direction].fillna(0).astype(float) != 0.0

    return mask


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_exit_raw_features(df: pd.DataFrame, cfg: Optional[ExitRawConfig] = None) -> pd.DataFrame:
    """
    Build leakage-free raw exit features for Pillar 2.

    Parameters
    ----------
    df : pd.DataFrame
        Full time-series DataFrame containing bar-level data, Pillar 1 final
        outputs (including entry decisions), and regime / normalised features.
    cfg : ExitRawConfig, optional
        Configuration object. If None, the default configuration is used.

    Returns
    -------
    exit_raw : pd.DataFrame
        One row per *entry*, with:

        - open_time_ms
        - entry_direction, entry_confidence, entry_prob_up (if present)
        - regime context
        - volatility / range features
        - excursion / path features
        - orderflow / liquidity features
        - structure / reversion features
    """
    if cfg is None:
        cfg = ExitRawConfig()

    # Basic checks
    _require_cols(
        df,
        [cfg.col_time, cfg.col_close, cfg.col_high, cfg.col_low],
        context="base OHLC/time",
    )

    # Ensure chronological order
    df = df.sort_values(cfg.col_time).reset_index(drop=False)
    orig_index_col = "exit_raw_orig_index"
    df[orig_index_col] = df["index"].astype("int64")
    df = df.drop(columns=["index"])

    # Volatility scale
    vol_scale = _compute_vol_scale(df, cfg)

    # Build feature blocks
    vol_feats = _volatility_features(df, cfg)
    exc_feats = _excursion_and_path_features(df, cfg, vol_scale)
    of_feats = _orderflow_liquidity_features(df, cfg)
    struct_feats = _structure_reversion_features(df, cfg, vol_scale)
    regime_feats = _regime_features(df, cfg)

    # Combine all feature blocks
    feat_blocks = [vol_feats, exc_feats, of_feats, struct_feats, regime_feats]
    feat_df = pd.concat(feat_blocks, axis=1)

    # Core entry metadata we want to carry through
    meta_cols = []
    for col in [
        cfg.col_time,
        cfg.col_entry_decision,
        cfg.col_entry_direction,
        cfg.col_entry_confidence,
        cfg.col_entry_prob,
        cfg.col_valid_flag,
    ]:
        if col in df.columns:
            meta_cols.append(col)

    base_out = df[meta_cols + [orig_index_col]].copy()
    out = pd.concat([base_out, feat_df], axis=1)

    # Filter to actual entries only (no non-trade bars)
    entry_mask = _entry_mask(df, cfg)
    exit_raw = out.loc[entry_mask].reset_index(drop=True)

    # Final clean-up: enforce dtypes clean + non-leaky indices
    if cfg.col_entry_direction in exit_raw.columns:
        exit_raw[cfg.col_entry_direction] = exit_raw[cfg.col_entry_direction].astype("int8")

    if cfg.col_entry_confidence in exit_raw.columns:
        exit_raw[cfg.col_entry_confidence] = exit_raw[cfg.col_entry_confidence].astype("float32")

    if cfg.col_entry_prob in exit_raw.columns:
        exit_raw[cfg.col_entry_prob] = (
            exit_raw[cfg.col_entry_prob]
            .astype(float)
            .clip(0.0, 1.0)
            .astype("float32")
        )

    # Keep original time-series row id for later label alignment
    exit_raw.rename(columns={orig_index_col: "exit_raw_row_id"}, inplace=True)

    return exit_raw