# src/exit/exit_raw_features.py

"""
Pillar 2 — Raw Exit Feature Builder

Takes Pillar 1 final features (plus optional Pillar 4 regime info)
and builds the raw, leakage-free feature snapshot used by the
Pillar 2 exit models.

Design:

- Operates on the FULL time-series first (sorted by open_time_ms).
- Computes only features that are available at the time of entry
  (no forward-looking leakage).
- Then filters down to rows where we actually open a trade
  (entry_decision ∈ {'long', 'short'}).

Output (exit_raw):

- One row per active trade.
- Contains:
    * time + OHLC + volume
    * entry_decision / direction / confidence / probability / validity
    * volatility / reversion / orderflow primitives from Pillar 1
    * regime context from Pillar 1 / Pillar 4
    * deterministic pre-entry stats: realised volatility, ranges,
      pre-entry excursions, path curvature, volatility dynamics, etc.

This file is intentionally conservative: optional columns such as
of_imbalance / struct_* / vol/liquidity buckets are NOT required. If
they are present upstream, you can wire them in later with a small
change here.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Config + helpers
# ---------------------------------------------------------------------


@dataclass
class ExitRawConfig:
    # Realised volatility windows (in bars, backwards in time)
    rv_windows: Sequence[int] = (5, 10, 20, 30)

    # Range (high-low) aggregation windows
    range_windows: Sequence[int] = (5, 10, 20)

    # "Context" windows (backwards) for pre-entry path shape & reversion
    excursion_lookback: int = 50
    path_curvature_window: int = 20
    orderflow_window: int = 20
    reversion_window: int = 50

    # Base column names
    col_time: str = "open_time_ms"
    col_open: str = "open"
    col_close: str = "close"
    col_high: str = "high"
    col_low: str = "low"
    col_volume: str = "volume"

    # Entry info from Pillar 1 decision layer
    col_entry_decision: str = "entry_decision"
    col_entry_direction: str = "entry_direction"
    col_entry_confidence: str = "entry_confidence"
    col_entry_prob: str = "entry_prob_up"
    col_valid_flag: str = "signal_is_valid"

    # P1 primitives
    col_atr: str = "vol_atr_14"
    col_zscore: str = "rev_zscore_50"

    col_of_delta: str = "of_delta"
    col_of_delta_ema: str = "of_delta_ema_20"
    col_of_imbalance: Optional[str] = None  # not present yet, optional
    col_of_sweep_flag: str = "of_sweep_flag"

    # Structure / fair value (optional – not yet in your pipeline)
    col_fair_value: Optional[str] = None
    col_nearest_level_dist: Optional[str] = None
    col_last_sweep_dir: Optional[str] = None

    # Regime info (matching features_pillar1_final)
    col_regime_id: str = "regime_id_stable"
    col_regime_conf: str = "regime_confidence"

    # Vol/liquidity buckets are not yet built → optional
    col_regime_vol_bucket: Optional[str] = None
    col_regime_liq_bucket: Optional[str] = None

    col_regime_trend_flag: str = "regime_is_trend"
    col_regime_chop_flag: str = "regime_is_chop"
    col_regime_liq_drought_flag: str = "regime_is_liq_drought"

    # Misc regime context from P1 (if present)
    col_regime_hold_norm: str = "regime_hold_length_norm"
    col_regime_chop_mask: str = "regime_chop_mask"
    col_mean_rev_flag: str = "mean_reversion_flag"


def _require_cols(df: pd.DataFrame, cols: List[str], context: str) -> None:
    cols = [c for c in cols if c is not None]  # ignore optional Nones
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[ExitRaw] Missing required columns for {context}: {missing}")


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


# ---------------------------------------------------------------------
# Feature builders (on full time-series)
# ---------------------------------------------------------------------


def _add_realised_vol_features(df: pd.DataFrame, cfg: ExitRawConfig) -> pd.DataFrame:
    close = pd.to_numeric(df[cfg.col_close], errors="coerce")
    log_ret = np.log(close / close.shift(1))

    for w in cfg.rv_windows:
        col = f"exit_rv_{w}"
        df[col] = (
            log_ret.rolling(window=w, min_periods=max(3, w // 2))
            .std()
            .astype("float32")
        )
    return df


def _add_range_features(df: pd.DataFrame, cfg: ExitRawConfig) -> pd.DataFrame:
    high = pd.to_numeric(df[cfg.col_high], errors="coerce")
    low = pd.to_numeric(df[cfg.col_low], errors="coerce")
    close = pd.to_numeric(df[cfg.col_close], errors="coerce")

    rng = (high - low).abs()
    for w in cfg.range_windows:
        col = f"exit_range_norm_{w}"
        roll_rng = rng.rolling(window=w, min_periods=max(3, w // 2)).max()
        df[col] = (roll_rng / (close + 1e-9)).astype("float32")
    return df


def _add_excursion_context(df: pd.DataFrame, cfg: ExitRawConfig) -> pd.DataFrame:
    """
    Pre-entry excursion context: how "stretched" the recent path is.

    Uses only past bars:
      - exit_pre_mfe: recent run-up vs local minimum
      - exit_pre_mae: recent drawdown vs local maximum
    """
    close = pd.to_numeric(df[cfg.col_close], errors="coerce")

    roll_max = close.rolling(
        window=cfg.excursion_lookback, min_periods=cfg.excursion_lookback // 4
    ).max()
    roll_min = close.rolling(
        window=cfg.excursion_lookback, min_periods=cfg.excursion_lookback // 4
    ).min()

    df["exit_pre_mfe"] = (close / (roll_min + 1e-9) - 1.0).astype("float32")
    df["exit_pre_mae"] = (roll_max / (close + 1e-9) - 1.0).astype("float32")
    return df


def _add_path_shape_features(df: pd.DataFrame, cfg: ExitRawConfig) -> pd.DataFrame:
    """
    Very lightweight path-shape approximation using only pre-entry data:

      - exit_path_curvature: 2nd difference of smoothed price (convex vs concave)
      - exit_path_monotonicity: fraction of last N bars that moved with same sign
    """
    close = pd.to_numeric(df[cfg.col_close], errors="coerce")
    win = cfg.path_curvature_window

    smooth = close.rolling(window=win, min_periods=max(3, win // 2)).mean()
    curvature = smooth.diff().diff()  # 2nd derivative-ish
    df["exit_path_curvature"] = curvature.astype("float32")

    # Monotonicity in last N bars: mean sign of 1-bar returns
    sign_ret = np.sign(close.diff())
    df["exit_path_monotonicity"] = (
        sign_ret.rolling(window=win, min_periods=max(3, win // 2)).mean()
    ).astype("float32")

    return df


def _add_vol_dynamics_features(df: pd.DataFrame, cfg: ExitRawConfig) -> pd.DataFrame:
    """
    Volatility dynamics from realised vol features:

      - exit_vol_ratio_short_long: rv_5 / rv_20
      - exit_vol_short_trend:       rv_5 - rv_20
    """
    short_col = "exit_rv_5"
    long_col = "exit_rv_20"

    if short_col in df.columns and long_col in df.columns:
        short = df[short_col].astype("float32")
        long = df[long_col].astype("float32")
        eps = 1e-8

        df["exit_vol_ratio_short_long"] = (short / (long + eps)).astype("float32")
        df["exit_vol_short_trend"] = (short - long).astype("float32")
    else:
        logging.warning(
            "[ExitRaw] Vol dynamics features fall back: "
            "missing exit_rv_5 or exit_rv_20 (short_col=%s, long_col=%s)",
            short_col,
            long_col,
        )

    return df


def _add_reversion_features(df: pd.DataFrame, cfg: ExitRawConfig) -> pd.DataFrame:
    """
    Reversion context from rev_zscore_50 (if present).
    """
    if cfg.col_zscore in df.columns:
        z = pd.to_numeric(df[cfg.col_zscore], errors="coerce")
        df["exit_reversion_z"] = z.astype("float32")
        df["exit_reversion_abs_z"] = z.abs().astype("float32")
    else:
        logging.warning(
            "[ExitRaw] Reversion z-score column %s missing; reversion features skipped.",
            cfg.col_zscore,
        )
    return df


def _add_orderflow_features(df: pd.DataFrame, cfg: ExitRawConfig) -> pd.DataFrame:
    """
    Simple orderflow context from of_delta / of_delta_ema / sweep flags.
    """
    if cfg.col_of_delta in df.columns:
        df["exit_of_delta_norm"] = (
            pd.to_numeric(df[cfg.col_of_delta], errors="coerce") /
            (pd.to_numeric(df.get(cfg.col_volume, 1.0), errors="coerce").abs() + 1e-6)
        ).astype("float32")

    if cfg.col_of_delta_ema in df.columns:
        df["exit_of_delta_ema_norm"] = (
            pd.to_numeric(df[cfg.col_of_delta_ema], errors="coerce") /
            (pd.to_numeric(df.get(cfg.col_volume, 1.0), errors="coerce").abs() + 1e-6)
        ).astype("float32")

    if cfg.col_of_sweep_flag in df.columns:
        # recent sweep in last orderflow_window bars
        sweep = pd.to_numeric(df[cfg.col_of_sweep_flag], errors="coerce").fillna(0.0)
        df["exit_recent_sweep_flag"] = (
            sweep.rolling(window=cfg.orderflow_window, min_periods=1).max()
        ).astype("float32")

    return df


def _add_regime_features(df: pd.DataFrame, cfg: ExitRawConfig) -> pd.DataFrame:
    """
    Pass-through of useful regime context from P1/P4.
    """
    for col in [
        cfg.col_regime_id,
        cfg.col_regime_conf,
        cfg.col_regime_trend_flag,
        cfg.col_regime_chop_flag,
        cfg.col_regime_liq_drought_flag,
        cfg.col_regime_hold_norm,
        cfg.col_regime_chop_mask,
        cfg.col_mean_rev_flag,
        cfg.col_regime_vol_bucket,
        cfg.col_regime_liq_bucket,
    ]:
        if col is None:
            continue
        if col not in df.columns:
            logging.debug("[ExitRaw] Regime context column %s not present; skipping.", col)
            continue
        # just ensure numeric where it makes sense; leave IDs as-is
        if col == cfg.col_regime_id:
            continue
        df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def build_exit_raw_features(
    df: pd.DataFrame,
    cfg: Optional[ExitRawConfig] = None,
) -> pd.DataFrame:
    """
    Build Pillar 2 raw exit features.

    Parameters
    ----------
    df : pd.DataFrame
        Typically features_pillar1_final (optionally already merged with
        Pillar 4 regime labels). Must contain at least one row per bar.
    cfg : ExitRawConfig, optional
        Configuration object; uses sensible defaults for your pipeline.

    Returns
    -------
    exit_raw : pd.DataFrame
        One row per active trade (entry_decision in {'long', 'short'}),
        with all raw exit features at entry time.
    """
    if cfg is None:
        cfg = ExitRawConfig()

    # Basic required columns (on full stream)
    _require_cols(
        df,
        [
            cfg.col_time,
            cfg.col_open,
            cfg.col_close,
            cfg.col_high,
            cfg.col_low,
            cfg.col_volume,
            cfg.col_entry_decision,
            cfg.col_entry_direction,
            cfg.col_entry_confidence,
            cfg.col_entry_prob,
        ],
        context="Pillar 1 final features for ExitRaw",
    )

    # Sort by time to preserve chronology
    df = df.sort_values(cfg.col_time).reset_index(drop=True)
    _log_mem(df, "[ExitRaw] Input (P1 final)")

    # Build features on FULL stream (no leakage)
    df_feat = df.copy()

    df_feat = _add_realised_vol_features(df_feat, cfg)
    df_feat = _add_range_features(df_feat, cfg)
    df_feat = _add_excursion_context(df_feat, cfg)
    df_feat = _add_path_shape_features(df_feat, cfg)
    df_feat = _add_vol_dynamics_features(df_feat, cfg)
    df_feat = _add_reversion_features(df_feat, cfg)
    df_feat = _add_orderflow_features(df_feat, cfg)
    df_feat = _add_regime_features(df_feat, cfg)

    # Filter down to ACTIVE trades (long/short only)
    trade_mask = df_feat[cfg.col_entry_decision].isin(["long", "short"])
    exit_raw = df_feat.loc[trade_mask].copy()
    exit_raw.reset_index(drop=True, inplace=True)

    _log_mem(exit_raw, "[ExitRaw] Final exit_raw snapshot")

    logging.info(
        "[ExitRaw] Rows with active trade decision: %d", len(exit_raw)
    )
    logging.info(
        "[ExitRaw] entry_decision distribution: %s",
        exit_raw[cfg.col_entry_decision].value_counts().to_dict(),
    )

    return exit_raw