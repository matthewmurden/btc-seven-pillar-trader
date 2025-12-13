"""
Pillar 2 — Deterministic Exit Engines (Part 2)

This module takes the Pillar 2 mini-engine outputs (exit_with_mini_engines)
and produces a *deterministic candidate exit profile* per trade.

Outputs per trade:
  - deterministic_tp                 (unit: directional return, >0)
  - deterministic_sl                 (unit: directional return magnitude, >0)
  - deterministic_tp_first_probability   (0–1, heuristic probability TP hits before SL)
  - deterministic_exit_time_estimate     (bars)

These are used as:
  - backup logic when ML confidence is low
  - strong guidance features for ML models
  - interpretable debugging tools

All logic is *deterministic* and uses only information available at entry:
no labels, no future leakage. It operates on the mini-engine-augmented
exit dataframe produced by build_exit_mini_engines.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Config + helpers
# ---------------------------------------------------------------------


@dataclass
class ExitDeterministicConfig:
    # Core trade metadata
    col_time: str = "open_time_ms"
    col_close: str = "close"
    col_atr: str = "vol_atr_14"
    col_direction: str = "entry_direction"
    col_decision: str = "entry_decision"
    col_conf: str = "entry_confidence"   # 0–1
    col_prob: str = "entry_prob_up"      # 0–1

    # Regime flags (from Pillar 4 / P1)
    col_regime_trend_flag: str = "regime_is_trend"
    col_regime_chop_flag: str = "regime_is_chop"
    col_regime_liq_drought_flag: str = "regime_is_liq_drought"

    # Mini-engine columns
    col_excursion_net: str = "exit_excursion_net"
    col_excursion_ratio: str = "exit_excursion_ratio"
    col_conf_weighted_excursion: str = "exit_conf_weighted_excursion"

    col_path_curvature: str = "exit_path_curvature"
    col_path_monotonicity: str = "exit_path_monotonicity"
    col_path_convexity_score: str = "exit_path_convexity_score"

    col_prob_entropy: str = "exit_prob_entropy"
    col_regime_complexity: str = "exit_regime_complexity"

    col_vol_ratio_short_long: str = "exit_vol_ratio_short_long"
    col_vol_short_trend: str = "exit_vol_short_trend"
    col_vol_scaled_rv_10: str = "exit_vol_scaled_rv_10"

    col_reversion_z: str = "exit_reversion_z"
    col_reversion_abs_z: str = "exit_reversion_abs_z"
    col_reversion_alignment: str = "exit_reversion_alignment"

    # Base hyperparameters for TP/SL shape
    base_tp_rr: float = 2.0          # base TP:SL risk-reward (TP = rr * SL)
    min_tp_rr: float = 0.8
    max_tp_rr: float = 4.0

    base_sl_mult_atr: float = 1.0    # base SL ≈ 1 * (ATR/close) in return space
    min_sl_mult_atr: float = 0.3
    max_sl_mult_atr: float = 3.0

    # Exit time estimate (in bars)
    base_exit_bars: int = 50
    min_exit_bars: int = 5
    max_exit_bars: int = 150

    # Numerical safety
    eps: float = 1e-8


def _require_cols(df: pd.DataFrame, cols: List[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[ExitDeterministic] Missing required columns for {context}: {missing}")


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def _safe_numeric(
    df: pd.DataFrame, col: str, default: float = 0.0, clip: Optional[tuple] = None
) -> np.ndarray:
    """
    Convert a column to float64 numpy array with NaNs replaced by `default`.
    If column is missing, returns an array full of `default`.
    Optionally clip to [low, high].
    """
    if col not in df.columns:
        logging.warning("[ExitDeterministic] Column %s missing, using default=%s.", col, default)
        arr = np.full(len(df), default, dtype=float)
    else:
        arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        mask_nan = ~np.isfinite(arr)
        if mask_nan.any():
            arr[mask_nan] = default
    if clip is not None:
        low, high = clip
        arr = np.clip(arr, low, high)
    return arr


def _compute_vol_scale(close: np.ndarray, atr: np.ndarray, eps: float) -> np.ndarray:
    """
    vol_scale ≈ typical 1-bar move in *return space*:
        vol_scale_i = ATR_i / close_i

    Clipped to avoid crazy extremes.
    """
    close_safe = np.abs(close)
    atr_safe = np.abs(atr)
    vol = atr_safe / (close_safe + eps)
    vol[~np.isfinite(vol)] = np.nan
    # Clip to a reasonable range
    vol = np.clip(vol, 1e-5, 0.5)
    # Fill NaNs with median or small fallback
    if np.isnan(vol).any():
        med = np.nanmedian(vol)
        if not np.isfinite(med) or med <= 0:
            med = 0.01
        vol[np.isnan(vol)] = med
    return vol.astype(float)


# ---------------------------------------------------------------------
# Core deterministic logic
# ---------------------------------------------------------------------


def build_exit_deterministic_profile(
    df: pd.DataFrame,
    cfg: Optional[ExitDeterministicConfig] = None,
) -> pd.DataFrame:
    """
    Build deterministic exit profile from exit_with_mini_engines.

    Parameters
    ----------
    df : pd.DataFrame
        The mini-engine-augmented Pillar 2 dataframe (one row per active trade),
        as produced by build_exit_mini_engines.
    cfg : ExitDeterministicConfig, optional
        Configuration for column names & hyperparameters.

    Returns
    -------
    df_out : pd.DataFrame
        Same as `df` with additional columns:
          - deterministic_tp
          - deterministic_sl
          - deterministic_tp_first_probability
          - deterministic_exit_time_estimate
    """
    if cfg is None:
        cfg = ExitDeterministicConfig()

    # Core required columns (everything else is handled via defaults if missing)
    _require_cols(
        df,
        [
            cfg.col_time,
            cfg.col_close,
            cfg.col_atr,
            cfg.col_direction,
            cfg.col_decision,
            cfg.col_conf,
            cfg.col_prob,
        ],
        context="exit_with_mini_engines",
    )

    _log_mem(df, "[ExitDeterministic] Input exit_with_mini_engines")

    eps = cfg.eps

    # ------------------------------------------------------------------
    # Extract core arrays
    # ------------------------------------------------------------------
    close = _safe_numeric(df, cfg.col_close, default=1.0, clip=(1e-3, np.inf))
    atr = _safe_numeric(df, cfg.col_atr, default=0.01, clip=(0.0, np.inf))
    vol_scale = _compute_vol_scale(close, atr, eps=eps)

    # entry_direction: +1 / -1 / 0 (we only need sign)
    direction = _safe_numeric(df, cfg.col_direction, default=0.0, clip=(-1.0, 1.0))
    sign_dir = np.sign(direction)  # not used directly yet, but useful for future tweaks

    # entry_confidence (0–1)
    conf = _safe_numeric(df, cfg.col_conf, default=0.5, clip=(0.0, 1.0))

    # entry_prob_up (0–1)
    prob_up = _safe_numeric(df, cfg.col_prob, default=0.5, clip=(0.0, 1.0))

    # Regime flags
    regime_trend = _safe_numeric(df, cfg.col_regime_trend_flag, default=0.0, clip=(0.0, 1.0))
    regime_chop = _safe_numeric(df, cfg.col_regime_chop_flag, default=0.0, clip=(0.0, 1.0))
    regime_liq_drought = _safe_numeric(
        df, cfg.col_regime_liq_drought_flag, default=0.0, clip=(0.0, 1.0)
    )

    # Mini-engine features
    excursion_net = _safe_numeric(df, cfg.col_excursion_net, default=0.0)
    excursion_ratio = _safe_numeric(df, cfg.col_excursion_ratio, default=1.0)
    conf_weighted_exc = _safe_numeric(df, cfg.col_conf_weighted_excursion, default=0.0)

    path_conv = _safe_numeric(df, cfg.col_path_convexity_score, default=0.0, clip=(-1.0, 1.0))
    path_mono = _safe_numeric(df, cfg.col_path_monotonicity, default=0.0, clip=(-1.0, 1.0))

    prob_entropy = _safe_numeric(df, cfg.col_prob_entropy, default=0.5)
    regime_complexity = _safe_numeric(df, cfg.col_regime_complexity, default=0.1, clip=(0.0, 1.0))

    vol_ratio = _safe_numeric(df, cfg.col_vol_ratio_short_long, default=1.0, clip=(0.0, np.inf))
    vol_short_trend = _safe_numeric(df, cfg.col_vol_short_trend, default=0.0)
    vol_scaled_rv10 = _safe_numeric(df, cfg.col_vol_scaled_rv_10, default=0.0, clip=(0.0, np.inf))

    reversion_abs_z = _safe_numeric(df, cfg.col_reversion_abs_z, default=0.0)
    reversion_align = _safe_numeric(df, cfg.col_reversion_alignment, default=0.0, clip=(-1.0, 1.0))

    # ------------------------------------------------------------------
    # 1) Risk-Reward (TP:SL) heuristic
    # ------------------------------------------------------------------
    # Normalise entropy & complexity into ~[0,1]
    entropy_norm = np.clip(prob_entropy / 0.7, 0.0, 1.0)
    complexity_norm = np.clip(regime_complexity, 0.0, 1.0)

    # Trend vs chop impact
    trend_factor = 0.5 * regime_trend   # up to +0.5
    chop_factor = -0.5 * regime_chop    # down to -0.5

    # Reversion effect: if strongly fading extremes (alignment > 0),
    # we prefer smaller TP:SL; if trading with extreme (alignment < 0),
    # allow slightly larger trend RR.
    reversion_effect = -0.3 * reversion_align  # + alignment ⇒ smaller RR

    # Confidence + entropy: high confidence & low entropy ⇒ higher RR
    conf_term = 1.5 * (conf - 0.5)            # [-0.75, +0.75]
    entropy_term = 1.0 * (0.5 - entropy_norm) # [-0.5, +0.5]

    rr_raw = (
        cfg.base_tp_rr
        * (1.0 + trend_factor + chop_factor + reversion_effect)
        * (1.0 + 0.5 * conf_term + 0.5 * entropy_term - 0.3 * complexity_norm)
    )

    tp_rr = np.clip(rr_raw, cfg.min_tp_rr, cfg.max_tp_rr)

    # ------------------------------------------------------------------
    # 2) SL distance heuristic
    # ------------------------------------------------------------------
    # Base multiplier on ATR/close (vol_scale), adjusted by:
    #  - entropy (more uncertain ⇒ slightly larger SL)
    #  - regime complexity (messy regime ⇒ slightly smaller SL)
    #  - pre-entry excursion (already stretched ⇒ tighter SL)
    #
    # Use a gentle, bounded transform.
    exc_scaled = np.tanh(excursion_net / (vol_scale + eps))  # [-1,1]

    sl_mult = (
        cfg.base_sl_mult_atr
        * (1.0 + 0.3 * entropy_norm)           # more uncertain ⇒ up to +30%
        * (1.0 - 0.2 * complexity_norm)        # more complex regime ⇒ slightly tighter
        * (1.0 - 0.2 * exc_scaled)             # strong positive net excursion ⇒ a bit tighter
    )

    # If very short-term vol is exploding (rv10), tighten SL a bit
    vol10_effect = np.tanh(vol_scaled_rv10 * 100.0)  # small but positive
    sl_mult *= (1.0 - 0.1 * vol10_effect)

    sl_mult = np.clip(sl_mult, cfg.min_sl_mult_atr, cfg.max_sl_mult_atr)

    # SL distance in *return* space (always positive scalar)
    deterministic_sl = sl_mult * vol_scale

    # ------------------------------------------------------------------
    # 3) TP distance from RR (also positive scalar)
    # ------------------------------------------------------------------
    deterministic_tp = tp_rr * deterministic_sl

    # ------------------------------------------------------------------
    # 4) TP-first probability heuristic
    # ------------------------------------------------------------------
    # Combine:
    #  - confidence (higher ⇒ TP more likely)
    #  - low entropy (lower uncertainty)
    #  - trend vs chop
    #  - excursion & path features
    #  - regime complexity (too complex ⇒ reduce TP odds)
    #
    # Purely deterministic heuristic. No label info.
    exc_dir_sign = np.sign(excursion_net)  # positive if pre-move in favour of trade
    path_bias = 0.3 * path_mono + 0.2 * path_conv  # [-0.5, +0.5-ish]

    score = (
        1.5 * (conf - 0.5)
        + 1.0 * (0.5 - entropy_norm)
        + 0.7 * regime_trend
        - 0.7 * regime_chop
        + 0.3 * exc_dir_sign
        + 0.2 * path_bias
        - 0.4 * complexity_norm
    )

    # Dampen in liquidity drought
    score -= 0.5 * regime_liq_drought

    # Map to [0,1] via logistic
    tp_first_prob = 1.0 / (1.0 + np.exp(-score))
    tp_first_prob = np.clip(tp_first_prob, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 5) Exit time estimate (bars)
    # ------------------------------------------------------------------
    # Base around cfg.base_exit_bars, extended in clean trends with high tp_prob,
    # shortened in chop / high entropy.
    base = float(cfg.base_exit_bars)

    time_scale = (
        1.0
        + 0.5 * regime_trend
        - 0.4 * regime_chop
        - 0.3 * entropy_norm
        - 0.2 * complexity_norm
        + 0.3 * (tp_first_prob - 0.5)
    )

    time_scale = np.clip(time_scale, 0.3, 2.0)
    exit_time_estimate = base * time_scale

    exit_time_estimate = np.clip(
        exit_time_estimate, float(cfg.min_exit_bars), float(cfg.max_exit_bars)
    )

    # ------------------------------------------------------------------
    # Build output DataFrame
    # ------------------------------------------------------------------
    out = df.copy()

    out["deterministic_tp"] = deterministic_tp.astype("float32")
    out["deterministic_sl"] = deterministic_sl.astype("float32")
    out["deterministic_tp_first_probability"] = tp_first_prob.astype("float32")
    out["deterministic_exit_time_estimate"] = exit_time_estimate.astype("float32")

    _log_mem(out, "[ExitDeterministic] Output with deterministic profile")

    # Summary logs
    logging.info(
        "[ExitDeterministic] deterministic_tp: mean=%.5f, std=%.5f, min=%.5f, max=%.5f",
        float(np.nanmean(deterministic_tp)),
        float(np.nanstd(deterministic_tp)),
        float(np.nanmin(deterministic_tp)),
        float(np.nanmax(deterministic_tp)),
    )
    logging.info(
        "[ExitDeterministic] deterministic_sl: mean=%.5f, std=%.5f, min=%.5f, max=%.5f",
        float(np.nanmean(deterministic_sl)),
        float(np.nanstd(deterministic_sl)),
        float(np.nanmin(deterministic_sl)),
        float(np.nanmax(deterministic_sl)),
    )
    logging.info(
        "[ExitDeterministic] deterministic_tp_first_probability: "
        "mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
        float(np.nanmean(tp_first_prob)),
        float(np.nanstd(tp_first_prob)),
        float(np.nanmin(tp_first_prob)),
        float(np.nanmax(tp_first_prob)),
    )
    logging.info(
        "[ExitDeterministic] deterministic_exit_time_estimate: "
        "mean=%.2f, std=%.2f, min=%.2f, max=%.2f",
        float(np.nanmean(exit_time_estimate)),
        float(np.nanstd(exit_time_estimate)),
        float(np.nanmin(exit_time_estimate)),
        float(np.nanmax(exit_time_estimate)),
    )

    return out