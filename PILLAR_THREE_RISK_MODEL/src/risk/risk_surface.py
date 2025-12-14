# src/risk/risk_surface.py

"""
Pillar 3 — Risk Model (Position Sizing)

Implements the full Pillar 3 spec:

  5.1 Inputs
      - Consumes Pillar 1, Pillar 2, Pillar 4 features and normalised context.

  5.2 Core Principles
      - Never exceed max risk per trade
      - Volatility- and liquidity-aware sizing
      - Scale up only when signal + regime support it
      - Scale down in chop, extremes, low liquidity, or drawdown
      - Avoid size spikes via smoothing and governors

  5.3 Risk Surface Estimator
      - Baseline allowable risk per trade from:
          * equity fraction (max_risk_pct)
          * leverage caps
          * SL distance
          * ATR-based volatility factor
          * liquidity factor
      - Produces a safe, volatility-aware baseline size.

  5.4 Volatility-Weighted Position Sizing
      - size_volatility_multiplier from:
          * volatility_score (P1/P4)
          * expected_exit_time
          * variance proxy of MFE distribution (mfe_dir^2)
          * sl_risk_factor (from Pillar 2)

  5.5 Signal-Aligned Scaling
      - signal_alignment_multiplier from:
          * direction-calibrated entry probability
          * meta agreement/conflict
          * trend alignment
          * breakout vs mean-reversion weighting

  5.6 Regime-Aware Risk Adjustments
      - regime_multiplier from:
          * trend/chop/mean-reversion flags
          * vol expansion / compression
          * liquidity drought / regimes

  5.7 Drawdown Governor
      - drawdown_multiplier from:
          * time-ordered realised label-based returns
          * rolling equity / max-equity → drawdown
      - Higher drawdown → smaller max risk per trade
      - Recovery → gradual scaling up (implemented as discrete states)

  5.8 Exposure Caps and Leverage Governor
      - leverage_multiplier based on volatility regime
      - exposure_cap_multiplier placeholder (per-trade) for Pillar 5
        to enforce portfolio-level exposure / concurrency limits.

  5.9 Final Position Size Generator
      Final size multiplier:

          position_size_multiplier
            = volatility_multiplier
            * signal_strength_multiplier
            * regime_multiplier
            * drawdown_multiplier
            * leverage_multiplier
            * exposure_cap_multiplier

      Final risk per trade:

          risk_per_trade_frac
            = clamp(
                 baseline_risk * position_size_multiplier,
                 [min_risk_pct, max_risk_pct]
              )

      Position notional & leverage:

          position_notional ≈ risk_per_trade_frac / sl_distance
          position_leverage = clip(position_notional, 0, max_leverage)

  5.10 Outputs
      Adds (among others):

          position_size              (alias of position_notional)
          position_leverage
          volatility_adjustment_flag
          risk_mode
          drawdown_state
          scaling_metadata
          position_risk_score
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Config + helpers
# ---------------------------------------------------------------------


@dataclass
class RiskSurfaceConfig:
    # Core risk constraints
    max_risk_pct: float = 0.02      # hard-ish cap ≈2% of equity per trade
    min_risk_pct: float = 0.0005    # 0.05% minimum to avoid dust trades
    max_leverage: float = 5.0       # per-trade leverage cap

    # SL handling
    min_sl_distance: float = 0.0005  # minimum SL distance in return space (~5 bps)

    # Baseline volatility / liquidity governors (5.3)
    vol_factor_min: float = 0.5
    vol_factor_max: float = 3.0
    liq_factor_min: float = 0.2
    liq_factor_max: float = 1.5
    liq_drought_multiplier: float = 0.3  # extra cut in liq drought regime

    # Volatility score weighting (used in 5.4)
    vol_score_weight: float = 0.2

    # Exit-time & MFE scaling
    exit_time_floor: float = 10.0
    exit_time_cap: float = 150.0
    mfe_var_ref: float = 1e-4  # reference variance for mfe_dir^2

    # Bounds for advanced multipliers (kept fairly tight)
    vol_size_mult_min: float = 0.5
    vol_size_mult_max: float = 1.3
    signal_mult_min: float = 0.5
    signal_mult_max: float = 1.7
    regime_mult_min: float = 0.4
    regime_mult_max: float = 1.4
    drawdown_mult_min: float = 0.5
    drawdown_mult_max: float = 1.1
    leverage_mult_min: float = 0.5
    leverage_mult_max: float = 1.2
    exposure_mult_min: float = 0.5
    exposure_mult_max: float = 1.0

    # Column names (aligned with current pipeline)
    col_time: str = "open_time_ms"
    col_decision: str = "entry_decision"
    col_direction: str = "entry_direction"
    col_entry_price: str = "close"

    col_atr: str = "vol_atr_14"
    col_volume: str = "volume"

    # From Pillar 2 final targets
    col_tp_level: str = "optimal_tp"
    col_sl_level: str = "optimal_sl"
    col_exit_time: str = "expected_exit_time"
    col_sl_risk_factor: str = "sl_risk_factor"  # from Pillar 2 (5.4)

    # From Pillar 1 / 4
    col_volatility_score: str = "volatility_score"
    col_regime_vol_expansion_flag: str = "regime_is_vol_expansion"
    col_regime_liq_drought_flag: str = "regime_is_liq_drought"
    col_regime_trend_flag: str = "regime_is_trend"
    col_regime_chop_flag: str = "regime_is_chop"
    col_regime_mean_rev_flag: str = "regime_is_mean_reversion"

    # Label-based features (for variance + drawdown governor)
    col_mfe_dir: str = "mfe_dir"
    col_mae_dir: str = "mae_dir"
    col_exit_outcome: str = "exit_outcome"
    col_optimal_tp_distance: str = "optimal_tp_distance"
    col_optimal_sl_distance: str = "optimal_sl_distance"

    # Validity flag (optional)
    col_valid_flag: str = "signal_is_valid"

    # Signal-aligned scaling (5.5)
    col_entry_prob_up: str = "entry_prob_up"
    col_entry_prob_down: str = "entry_prob_down"
    col_meta_agreement: str = "meta_agreement_score"
    col_meta_conflict: str = "meta_conflict_score"
    col_trend_score: str = "trend_score"
    col_meta_trend_align: str = "meta_regime_trend_confirmation"
    col_breakout_prob_up: str = "breakout_prob_up"
    col_mean_reversion_prob_up: str = "mean_reversion_prob_up"

    # Numerical stability
    eps: float = 1e-8


def _require_cols(df: pd.DataFrame, cols: List[str], context: str) -> None:
    cols = [c for c in cols if c is not None]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[RiskSurface] Missing required columns for {context}: {missing}")


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def _nanmedian_safe(x: pd.Series, fallback: float) -> float:
    med = float(np.nanmedian(x.to_numpy(dtype=float)))
    if not np.isfinite(med) or med <= 0:
        return fallback
    return med


# ---------------------------------------------------------------------
# 5.4 Volatility-weighted sizing
# ---------------------------------------------------------------------


def _compute_volatility_size_multiplier(
    df_active: pd.DataFrame,
    cfg: RiskSurfaceConfig,
) -> pd.Series:
    """
    5.4 Volatility-Weighted Position Sizing

    Uses:
      - volatility_score          → high vol → shrink size
      - expected_exit_time        → longer hold → shrink size
      - mfe_dir^2 (variance proxy)→ fatter tails → shrink size
      - sl_risk_factor            → direct Pillar 2 risk hint
    """
    idx = df_active.index
    n = len(df_active)
    eps = cfg.eps

    # --- volatility_score component ---
    if cfg.col_volatility_score in df_active.columns:
        vol_score = pd.to_numeric(
            df_active[cfg.col_volatility_score], errors="coerce"
        ).fillna(0.0)
        vol_score_clip = vol_score.clip(-3.0, 3.0)
        # Positive vol_score => high vol => reduce size
        vol_component = 1.0 - cfg.vol_score_weight * vol_score_clip
    else:
        vol_component = np.ones(n, dtype=float)

    # --- expected_exit_time component ---
    if cfg.col_exit_time in df_active.columns:
        t = pd.to_numeric(df_active[cfg.col_exit_time], errors="coerce")
        t_clip = t.clip(cfg.exit_time_floor, cfg.exit_time_cap)
        # Longer expected hold → smaller
        time_component = 1.0 / np.sqrt(t_clip / cfg.exit_time_floor)
    else:
        time_component = np.ones(n, dtype=float)

    # --- MFE variance proxy component (mfe_dir^2) ---
    if cfg.col_mfe_dir in df_active.columns:
        mfe = pd.to_numeric(df_active[cfg.col_mfe_dir], errors="coerce").fillna(0.0)
        mfe_var_proxy = mfe ** 2
        mfe_ratio = (mfe_var_proxy / (cfg.mfe_var_ref + eps)).clip(0.25, 4.0)
        # Higher variance => smaller
        mfe_component = 1.0 / np.sqrt(mfe_ratio)
    else:
        mfe_component = np.ones(n, dtype=float)

    # --- SL risk factor from Pillar 2 ---
    if cfg.col_sl_risk_factor in df_active.columns:
        sl_rf = pd.to_numeric(
            df_active[cfg.col_sl_risk_factor], errors="coerce"
        ).fillna(1.0)
        sl_rf_clip = sl_rf.clip(0.5, 1.5)
    else:
        sl_rf_clip = np.ones(n, dtype=float)

    mult = vol_component * time_component * mfe_component * sl_rf_clip
    mult = np.clip(mult, cfg.vol_size_mult_min, cfg.vol_size_mult_max)

    return pd.Series(mult, index=idx, dtype="float32")


# ---------------------------------------------------------------------
# 5.5 Signal-aligned scaling
# ---------------------------------------------------------------------


def _compute_signal_alignment_multiplier(
    df_active: pd.DataFrame,
    cfg: RiskSurfaceConfig,
) -> pd.Series:
    """
    5.5 Signal-Aligned Scaling

    Uses:
      - entry probability in trade direction
      - meta_agreement_score / meta_conflict_score
      - trend alignment (trend_score, meta_regime_trend_confirmation)
      - breakout vs mean-reversion probability weighting
    """
    idx = df_active.index
    n = len(df_active)
    eps = cfg.eps

    direction = pd.to_numeric(
        df_active.get(cfg.col_direction, 0.0), errors="coerce"
    ).fillna(0.0)
    sign_dir = np.sign(direction.to_numpy(dtype=float))

    # --- Directional entry probability ---
    p_dir = np.full(n, 0.5, dtype=float)
    if (
        cfg.col_entry_prob_up in df_active.columns
        and cfg.col_entry_prob_down in df_active.columns
    ):
        p_up = pd.to_numeric(df_active[cfg.col_entry_prob_up], errors="coerce")
        p_dn = pd.to_numeric(df_active[cfg.col_entry_prob_down], errors="coerce")
        p_dir = np.where(sign_dir >= 0, p_up, p_dn)
        p_dir = np.clip(p_dir, 0.0, 1.0)

    # Map:
    #   0.5 → 1.0 (neutral)
    #   0.25 → 0.5 (low conviction)
    #   0.75 → 1.5 (high conviction)
    prob_component = 2.0 * (p_dir - 0.5) + 1.0
    prob_component = np.clip(prob_component, 0.2, 2.0)

    # --- Engine agreement vs conflict ---
    if cfg.col_meta_agreement in df_active.columns:
        agree = pd.to_numeric(
            df_active[cfg.col_meta_agreement], errors="coerce"
        ).fillna(0.0)
        agree = agree.clip(-1.0, 1.0)
        # -1 → 0.5, 0 → 1.0, +1 → 1.5
        agree_component = 1.0 + 0.5 * agree.to_numpy(dtype=float)
    else:
        agree_component = np.ones(n, dtype=float)

    if cfg.col_meta_conflict in df_active.columns:
        conflict = pd.to_numeric(
            df_active[cfg.col_meta_conflict], errors="coerce"
        ).fillna(0.0)
        conflict = conflict.clip(0.0, 5.0)
        # 0 → 1.0, 1 → 0.5, 3+ → ~0.25 (clamped later)
        conf_component = 1.0 / (1.0 + conflict.to_numpy(dtype=float))
        conf_component = np.clip(conf_component, 0.4, 1.0)
    else:
        conf_component = np.ones(n, dtype=float)

    # --- Trend alignment ---
    if cfg.col_trend_score in df_active.columns:
        trend = pd.to_numeric(
            df_active[cfg.col_trend_score], errors="coerce"
        ).fillna(0.0)
        trend_sign = np.sign(trend.to_numpy(dtype=float))

        align_raw = sign_dir * trend_sign

        if cfg.col_meta_trend_align in df_active.columns:
            meta_align = pd.to_numeric(
                df_active[cfg.col_meta_trend_align], errors="coerce"
            ).fillna(0.0)
            meta_sign = np.sign(meta_align.to_numpy(dtype=float))
            align_raw = align_raw * np.where(meta_sign == 0.0, 1.0, meta_sign)

        # +1 → 1.2 (trend-aligned), -1 → 0.8 (fighting trend), 0 → 1.0
        trend_component = np.where(
            align_raw > 0,
            1.2,
            np.where(align_raw < 0, 0.8, 1.0),
        )
    else:
        trend_component = np.ones(n, dtype=float)

    # --- Breakout vs mean-reversion weighting ---
    if (
        cfg.col_breakout_prob_up in df_active.columns
        and cfg.col_mean_reversion_prob_up in df_active.columns
    ):
        brk = pd.to_numeric(
            df_active[cfg.col_breakout_prob_up], errors="coerce"
        ).clip(0.0, 1.0)
        rev = pd.to_numeric(
            df_active[cfg.col_mean_reversion_prob_up], errors="coerce"
        ).clip(0.0, 1.0)

        brk_arr = brk.to_numpy(dtype=float)
        rev_arr = rev.to_numpy(dtype=float)

        # Project onto trade direction
        brk_dir = np.where(sign_dir >= 0, brk_arr, 1.0 - brk_arr)
        rev_dir = np.where(sign_dir >= 0, rev_arr, 1.0 - rev_arr)

        adv = np.clip(brk_dir - rev_dir, -1.0, 1.0)
        # adv = +1 → 1.3 (breakout dominates)
        # adv = -1 → 0.7 (reversion dominates)
        brk_component = 1.0 + 0.3 * adv
        brk_component = np.clip(brk_component, 0.7, 1.3)
    else:
        brk_component = np.ones(n, dtype=float)

    mult = (
        prob_component
        * agree_component
        * conf_component
        * trend_component
        * brk_component
    )
    mult = np.clip(mult, cfg.signal_mult_min, cfg.signal_mult_max)

    return pd.Series(mult.astype("float32"), index=idx)


# ---------------------------------------------------------------------
# 5.6 Regime-aware risk adjustments
# ---------------------------------------------------------------------


def _compute_regime_multiplier(
    df_active: pd.DataFrame,
    cfg: RiskSurfaceConfig,
) -> pd.Series:
    """
    5.6 Regime-Aware Risk Adjustments

    Trend regime        → allow expansion
    Chop regime         → reduce sharply
    Vol expansion       → reduce moderately
    Vol compression     → reduce sharply
    Liquidity drought   → minimum size
    Mean-reversion zone → conservative sizing
    """
    idx = df_active.index
    n = len(df_active)

    mult = np.ones(n, dtype=float)

    # Base flags
    trend = pd.to_numeric(
        df_active.get(cfg.col_regime_trend_flag, 0.0), errors="coerce"
    ).fillna(0.0)
    chop = pd.to_numeric(
        df_active.get(cfg.col_regime_chop_flag, 0.0), errors="coerce"
    ).fillna(0.0)
    mean_rev = pd.to_numeric(
        df_active.get(cfg.col_regime_mean_rev_flag, 0.0), errors="coerce"
    ).fillna(0.0)
    vol_exp = pd.to_numeric(
        df_active.get(cfg.col_regime_vol_expansion_flag, 0.0), errors="coerce"
    ).fillna(0.0)
    liq_drought = pd.to_numeric(
        df_active.get(cfg.col_regime_liq_drought_flag, 0.0), errors="coerce"
    ).fillna(0.0)

    # Approximate vol compression via very low volatility_score
    if cfg.col_volatility_score in df_active.columns:
        vol_score = pd.to_numeric(
            df_active[cfg.col_volatility_score], errors="coerce"
        ).fillna(0.0)
        vol_compress_flag = (vol_score < -1.0).astype(float)
    else:
        vol_compress_flag = np.zeros(n, dtype=float)

    # Chop regime: hard cut
    mult = np.where(chop > 0.5, mult * 0.4, mult)

    # Trend regime: expand modestly (if not chopped)
    mult = np.where(
        (trend > 0.5) & (chop <= 0.5),
        mult * 1.2,
        mult,
    )

    # Mean-reversion zone: slightly conservative
    mult = np.where(mean_rev > 0.5, mult * 0.8, mult)

    # Vol expansion: reduce size moderately
    mult = np.where(vol_exp > 0.5, mult * 0.8, mult)

    # Vol compression: reduce size sharply (no aggressive sizing)
    mult = np.where(vol_compress_flag > 0.5, mult * 0.6, mult)

    # Liquidity drought: force minimum size
    mult = np.where(liq_drought > 0.5, mult * 0.3, mult)

    mult = np.clip(mult, cfg.regime_mult_min, cfg.regime_mult_max)
    return pd.Series(mult.astype("float32"), index=idx)


# ---------------------------------------------------------------------
# 5.7 Drawdown governor
# ---------------------------------------------------------------------


def _compute_label_return_for_dd(
    df_active: pd.DataFrame,
    cfg: RiskSurfaceConfig,
) -> Optional[pd.Series]:
    """
    Build a simple realised-return proxy from labels for drawdown governor.

    Uses:
      - exit_outcome ∈ {'tp_first', 'sl_first', 'timeout'}
      - optimal_tp_distance, optimal_sl_distance
      - direction sign (entry_direction)
    """
    required = [
        cfg.col_exit_outcome,
        cfg.col_optimal_tp_distance,
        cfg.col_optimal_sl_distance,
        cfg.col_direction,
    ]
    if any(c not in df_active.columns for c in required):
        logging.warning(
            "[RiskSurface] Drawdown governor: missing label columns, "
            "using neutral drawdown (multiplier=1)."
        )
        return None

    outcome = df_active[cfg.col_exit_outcome].astype(str)
    tp_dist = pd.to_numeric(
        df_active[cfg.col_optimal_tp_distance], errors="coerce"
    ).fillna(0.0)
    sl_dist = pd.to_numeric(
        df_active[cfg.col_optimal_sl_distance], errors="coerce"
    ).fillna(0.0)
    direction = pd.to_numeric(
        df_active[cfg.col_direction], errors="coerce"
    ).fillna(0.0)
    sign_dir = np.sign(direction.to_numpy(dtype=float))

    outcome_arr = outcome.to_numpy(dtype=str)
    tp = tp_dist.to_numpy(dtype=float)
    sl = sl_dist.to_numpy(dtype=float)

    # Simple proxy: take signed TP/SL distance depending on outcome.
    ret = np.zeros(len(df_active), dtype=float)
    ret[outcome_arr == "tp_first"] = tp[outcome_arr == "tp_first"] * sign_dir[
        outcome_arr == "tp_first"
    ]
    ret[outcome_arr == "sl_first"] = -sl[outcome_arr == "sl_first"] * sign_dir[
        outcome_arr == "sl_first"
    ]
    # timeout → approx small realised outcome: 0

    return pd.Series(ret, index=df_active.index, dtype="float32")


def _compute_drawdown_multiplier(
    df_active: pd.DataFrame,
    label_return: Optional[pd.Series],
    cfg: RiskSurfaceConfig,
) -> Tuple[pd.Series, pd.Series]:
    """
    5.7 Drawdown Governor

    Builds an equity curve from past label-based returns and computes drawdown.
    Uses a simple piecewise mapping from drawdown to multiplier:

        drawdown ≥ -5%      → 1.0  (state='ok')
        -10% ≤ dd < -5%     → 0.9  (state='mild')
        -20% ≤ dd < -10%    → 0.7  (state='moderate')
        dd < -20%           → 0.5  (state='severe')

    Returns:
        drawdown_multiplier, drawdown_state
    """
    idx = df_active.index
    n = len(df_active)

    if label_return is None or label_return.empty:
        # Neutral drawdown state
        mult = np.ones(n, dtype=float)
        state = np.full(n, "none", dtype=object)
        return (
            pd.Series(mult.astype("float32"), index=idx),
            pd.Series(state, index=idx, dtype="object"),
        )

    # Sort trades by time to avoid leakage
    if cfg.col_time not in df_active.columns:
        logging.warning(
            "[RiskSurface] Drawdown governor: %s missing; using neutral.",
            cfg.col_time,
        )
        mult = np.ones(n, dtype=float)
        state = np.full(n, "none", dtype=object)
        return (
            pd.Series(mult.astype("float32"), index=idx),
            pd.Series(state, index=idx, dtype="object"),
        )

    df_dd = df_active[[cfg.col_time]].copy()
    df_dd["label_return"] = label_return
    df_dd = df_dd.sort_values(cfg.col_time)

    # Use lagged returns so trade t only sees history up to t-1
    ret = df_dd["label_return"].to_numpy(dtype=float)
    ret_shifted = np.concatenate([[0.0], ret[:-1]])
    equity = 1.0 + np.cumsum(ret_shifted)
    max_equity = np.maximum.accumulate(equity)
    dd = equity / np.maximum(max_equity, cfg.eps) - 1.0  # in [-1, 0]

    # Map drawdown to multiplier & state
    mult = np.ones_like(dd)
    state = np.full(dd.shape, "ok", dtype=object)

    # severe drawdown (< -20%)
    mask_severe = dd < -0.20
    mult[mask_severe] = 0.5
    state[mask_severe] = "severe"

    # moderate [-20%, -10%)
    mask_moderate = (dd >= -0.20) & (dd < -0.10)
    mult[mask_moderate] = 0.7
    state[mask_moderate] = "moderate"

    # mild [-10%, -5%)
    mask_mild = (dd >= -0.10) & (dd < -0.05)
    mult[mask_mild] = 0.9
    state[mask_mild] = "mild"

    # ok (≥ -5%) → 1.0

    mult = np.clip(mult, cfg.drawdown_mult_min, cfg.drawdown_mult_max)

    # Map back to original index (unsorted order)
    dd_mult_series = pd.Series(mult.astype("float32"), index=df_dd.index)
    dd_state_series = pd.Series(state, index=df_dd.index, dtype="object")

    dd_mult_series = dd_mult_series.reindex(idx)
    dd_state_series = dd_state_series.reindex(idx)

    return dd_mult_series, dd_state_series


# ---------------------------------------------------------------------
# 5.8 Exposure caps & leverage governor
# ---------------------------------------------------------------------


def _compute_leverage_multiplier(
    df_active: pd.DataFrame,
    cfg: RiskSurfaceConfig,
) -> pd.Series:
    """
    Leverage governor (per-trade).

    Higher leverage only in relatively low-vol, non-expansion regimes.
    """
    idx = df_active.index
    n = len(df_active)
    eps = cfg.eps

    if cfg.col_volatility_score in df_active.columns:
        vol_score = pd.to_numeric(
            df_active[cfg.col_volatility_score], errors="coerce"
        ).fillna(0.0)
    else:
        vol_score = pd.Series(np.zeros(n, dtype=float), index=idx)

    vol_exp = pd.to_numeric(
        df_active.get(cfg.col_regime_vol_expansion_flag, 0.0), errors="coerce"
    ).fillna(0.0)

    # Base: 1.0
    mult = np.ones(n, dtype=float)

    # Low-vol, no expansion → allow mild leverage expansion
    low_vol_mask = (vol_score < -0.5) & (vol_exp <= 0.5)
    mult[low_vol_mask] *= 1.2

    # High-vol or expansion → cut leverage
    high_vol_mask = (vol_score > 1.0) | (vol_exp > 0.5)
    mult[high_vol_mask] *= 0.8

    mult = np.clip(mult, cfg.leverage_mult_min, cfg.leverage_mult_max)
    return pd.Series(mult.astype("float32"), index=idx)


def _compute_exposure_cap_multiplier(
    df_active: pd.DataFrame,
    cfg: RiskSurfaceConfig,
) -> pd.Series:
    """
    Exposure cap multiplier (per trade).

    NOTE:
      True portfolio-level exposure caps (max open leverage, concurrent positions,
      directional exposure, cooldowns) require stateful scheduling in Pillar 5.

      Here we provide a mild *per-trade* exposure_cap_multiplier based on
      recent directional concentration as a hint to Pillar 5.

    Logic (approximate):
      - Look at recent N trades' direction sign.
      - If heavily skewed to one side, reduce new size a bit.
    """
    idx = df_active.index
    n = len(df_active)

    if cfg.col_time not in df_active.columns or cfg.col_direction not in df_active.columns:
        # Neutral exposure cap (Pillar 5 will enforce portfolio caps)
        return pd.Series(
            np.ones(n, dtype="float32"),
            index=idx,
        )

    df_exp = df_active[[cfg.col_time, cfg.col_direction]].copy()
    df_exp = df_exp.sort_values(cfg.col_time)

    direction = pd.to_numeric(df_exp[cfg.col_direction], errors="coerce").fillna(0.0)
    sign_dir = np.sign(direction.to_numpy(dtype=float))

    # Rolling window over trades, not time; small to avoid heavy compute
    window = 200
    rolling_mean = pd.Series(sign_dir).rolling(window=window, min_periods=1).mean()
    skew = rolling_mean.to_numpy(dtype=float)  # in [-1, 1]

    # If |skew| > 0.5 we are heavily biased to one side → cut size moderately
    mult = np.ones_like(skew)
    mult[np.abs(skew) > 0.7] = 0.6   # strong one-sided exposure
    mult[(np.abs(skew) > 0.5) & (np.abs(skew) <= 0.7)] = 0.8  # moderate

    mult = np.clip(mult, cfg.exposure_mult_min, cfg.exposure_mult_max)

    # Map back to active index order
    mult_series = pd.Series(mult.astype("float32"), index=df_exp.index)
    mult_series = mult_series.reindex(idx)

    return mult_series


# ---------------------------------------------------------------------
# Public API: full risk surface + sizing
# ---------------------------------------------------------------------


def compute_risk_surface_and_sizing(
    trades_df: pd.DataFrame,
    cfg: Optional[RiskSurfaceConfig] = None,
) -> pd.DataFrame:
    """
    Build Pillar 3 risk surface + position sizing.

    Adds (among others):

        sl_distance
        vol_factor
        liq_factor

        size_volatility_multiplier
        signal_alignment_multiplier
        regime_multiplier
        drawdown_multiplier
        leverage_multiplier
        exposure_cap_multiplier
        position_size_multiplier

        risk_per_trade_frac
        position_notional
        position_size
        position_leverage

        volatility_adjustment_flag
        risk_mode
        drawdown_state
        scaling_metadata
        position_risk_score
    """
    if cfg is None:
        cfg = RiskSurfaceConfig()

    df = trades_df.copy()
    _log_mem(df, "[RiskSurface] Input trades_df")

    # Basic requirements
    _require_cols(
        df,
        [
            cfg.col_decision,
            cfg.col_direction,
            cfg.col_entry_price,
            cfg.col_sl_level,
        ],
        context="base P3 sizing",
    )

    # Active trade mask: decision != 'none', direction != 0, and valid if flag exists
    active_mask = (df[cfg.col_decision] != "none") & (
        pd.to_numeric(df[cfg.col_direction], errors="coerce") != 0
    )
    if cfg.col_valid_flag in df.columns:
        valid_flag = pd.to_numeric(df[cfg.col_valid_flag], errors="coerce").fillna(1.0)
        active_mask &= valid_flag != 0

    if not active_mask.any():
        logging.warning("[RiskSurface] No active trades found in trades_df.")
        # Add empty output columns for consistency
        for col in [
            "sl_distance",
            "vol_factor",
            "liq_factor",
            "size_volatility_multiplier",
            "signal_alignment_multiplier",
            "regime_multiplier",
            "drawdown_multiplier",
            "leverage_multiplier",
            "exposure_cap_multiplier",
            "position_size_multiplier",
            "risk_per_trade_frac",
            "position_notional",
            "position_size",
            "position_leverage",
            "volatility_adjustment_flag",
            "risk_mode",
            "drawdown_state",
            "scaling_metadata",
            "position_risk_score",
        ]:
            df[col] = (
                0.0
                if "flag" not in col
                and "state" not in col
                and "mode" not in col
                and "metadata" not in col
                else ""
            )
        _log_mem(df, "[RiskSurface] Output (no active trades)")
        return df

    df_active = df.loc[active_mask].copy()

    # --------------------------------------------------------------
    # 5.3 Risk Surface Estimator: SL distance, vol_factor, liq_factor
    # --------------------------------------------------------------
    entry_price = pd.to_numeric(
        df_active[cfg.col_entry_price], errors="coerce"
    ).astype(float)
    sl_level = pd.to_numeric(df_active[cfg.col_sl_level], errors="coerce").astype(float)

    sl_distance = np.abs(sl_level - entry_price) / np.clip(
        entry_price.abs(), cfg.eps, np.inf
    )
    sl_distance = sl_distance.clip(lower=cfg.min_sl_distance)

    df["sl_distance"] = np.nan
    df.loc[active_mask, "sl_distance"] = sl_distance.astype("float32")

    # ATR-based volatility factor
    if cfg.col_atr in df_active.columns:
        atr = pd.to_numeric(df_active[cfg.col_atr], errors="coerce").abs()
        atr_pct = atr / np.clip(entry_price.abs(), cfg.eps, np.inf)
        med_atr_pct = _nanmedian_safe(atr_pct, fallback=0.005)
        atr_norm = (atr_pct / (med_atr_pct + cfg.eps)).clip(0.1, 10.0)
        vol_factor = (1.0 / np.sqrt(atr_norm)).clip(
            cfg.vol_factor_min, cfg.vol_factor_max
        )
    else:
        logging.warning(
            "[RiskSurface] ATR column %s missing; using vol_factor=1.0.", cfg.col_atr
        )
        vol_factor = np.ones(len(df_active), dtype=float)

    # Liquidity factor
    if cfg.col_volume in df_active.columns:
        vol = pd.to_numeric(df_active[cfg.col_volume], errors="coerce").abs()
        med_vol = _nanmedian_safe(vol, fallback=1.0)
        vol_norm = (vol / (med_vol + cfg.eps)).clip(0.1, 10.0)
        liq_factor = (np.sqrt(vol_norm)).clip(cfg.liq_factor_min, cfg.liq_factor_max)
    else:
        logging.warning(
            "[RiskSurface] Volume column %s missing; using liq_factor=1.0.",
            cfg.col_volume,
        )
        liq_factor = np.ones(len(df_active), dtype=float)

    if cfg.col_regime_liq_drought_flag in df_active.columns:
        liq_drought_flag = pd.to_numeric(
            df_active[cfg.col_regime_liq_drought_flag], errors="coerce"
        ).fillna(0.0)
        drought_arr = liq_drought_flag.to_numpy(dtype=float)
        liq_factor = liq_factor * np.where(
            drought_arr > 0, cfg.liq_drought_multiplier, 1.0
        )

    # Baseline risk per trade (before advanced scaling)
    baseline_risk = cfg.max_risk_pct * vol_factor * liq_factor
    baseline_risk = np.clip(
        baseline_risk, cfg.min_risk_pct, cfg.max_risk_pct * 1.5
    )

    # --------------------------------------------------------------
    # 5.4 Volatility-weighted position sizing multiplier
    # --------------------------------------------------------------
    size_vol_mult = _compute_volatility_size_multiplier(df_active, cfg)

    # --------------------------------------------------------------
    # 5.5 Signal-aligned scaling multiplier
    # --------------------------------------------------------------
    signal_mult = _compute_signal_alignment_multiplier(df_active, cfg)

    # --------------------------------------------------------------
    # 5.6 Regime-aware multiplier
    # --------------------------------------------------------------
    regime_mult = _compute_regime_multiplier(df_active, cfg)

    # --------------------------------------------------------------
    # 5.7 Drawdown governor
    # --------------------------------------------------------------
    label_return = _compute_label_return_for_dd(df_active, cfg)
    drawdown_mult, drawdown_state = _compute_drawdown_multiplier(
        df_active, label_return, cfg
    )

    # --------------------------------------------------------------
    # 5.8 Leverage governor + exposure cap
    # --------------------------------------------------------------
    leverage_mult = _compute_leverage_multiplier(df_active, cfg)
    exposure_mult = _compute_exposure_cap_multiplier(df_active, cfg)

    # Combined multiplier (regime + signal + vol + drawdown + leverage + exposure)
    combined_mult = (
        size_vol_mult.to_numpy(dtype=float)
        * signal_mult.to_numpy(dtype=float)
        * regime_mult.to_numpy(dtype=float)
        * drawdown_mult.to_numpy(dtype=float)
        * leverage_mult.to_numpy(dtype=float)
        * exposure_mult.to_numpy(dtype=float)
    )

    # Clamp combined multiplier to sane global bounds
    combined_mult = np.clip(
        combined_mult,
        cfg.vol_size_mult_min
        * cfg.signal_mult_min
        * cfg.regime_mult_min
        * cfg.drawdown_mult_min
        * cfg.leverage_mult_min
        * cfg.exposure_mult_min,
        cfg.vol_size_mult_max
        * cfg.signal_mult_max
        * cfg.regime_mult_max
        * cfg.drawdown_mult_max
        * cfg.leverage_mult_max
        * cfg.exposure_mult_max,
    )

    # --------------------------------------------------------------
    # Initial risk_per_trade_frac (P&L risk, not notional)
    # --------------------------------------------------------------
    raw_risk_frac = baseline_risk * combined_mult
    risk_per_trade_frac = np.clip(
        raw_risk_frac, cfg.min_risk_pct, cfg.max_risk_pct
    )

    # --------------------------------------------------------------
    # Initial position notional & leverage (before risk_mode gating)
    # --------------------------------------------------------------
    sl_dist_arr = sl_distance.to_numpy(dtype=float)
    position_notional = risk_per_trade_frac / np.clip(
        sl_dist_arr, cfg.min_sl_distance, np.inf
    )
    position_leverage = np.clip(position_notional, 0.0, cfg.max_leverage)

    # If leverage was clipped, adjust notional to match the capped leverage
    position_notional = position_leverage
    position_size = position_notional  # alias

    # --------------------------------------------------------------
    # Write base results back to full DataFrame
    # --------------------------------------------------------------
    df["vol_factor"] = np.nan
    df["liq_factor"] = np.nan
    df["size_volatility_multiplier"] = 1.0
    df["signal_alignment_multiplier"] = 1.0
    df["regime_multiplier"] = 1.0
    df["drawdown_multiplier"] = 1.0
    df["leverage_multiplier"] = 1.0
    df["exposure_cap_multiplier"] = 1.0
    df["position_size_multiplier"] = 1.0
    df["risk_per_trade_frac"] = 0.0
    df["position_notional"] = 0.0
    df["position_size"] = 0.0
    df["position_leverage"] = 0.0
    df["drawdown_state"] = ""
    df["volatility_adjustment_flag"] = ""
    df["risk_mode"] = ""
    df["scaling_metadata"] = ""
    df["position_risk_score"] = 0.0

    df.loc[active_mask, "vol_factor"] = vol_factor.astype("float32")
    df.loc[active_mask, "liq_factor"] = liq_factor.astype("float32")
    df.loc[active_mask, "size_volatility_multiplier"] = size_vol_mult.astype("float32")
    df.loc[active_mask, "signal_alignment_multiplier"] = signal_mult.astype("float32")
    df.loc[active_mask, "regime_multiplier"] = regime_mult.astype("float32")
    df.loc[active_mask, "drawdown_multiplier"] = drawdown_mult.astype("float32")
    df.loc[active_mask, "leverage_multiplier"] = leverage_mult.astype("float32")
    df.loc[active_mask, "exposure_cap_multiplier"] = exposure_mult.astype("float32")
    df.loc[active_mask, "position_size_multiplier"] = combined_mult.astype("float32")
    df.loc[active_mask, "risk_per_trade_frac"] = risk_per_trade_frac.astype("float32")
    df.loc[active_mask, "position_notional"] = position_notional.astype("float32")
    df.loc[active_mask, "position_size"] = position_size.astype("float32")
    df.loc[active_mask, "position_leverage"] = position_leverage.astype("float32")
    df.loc[active_mask, "drawdown_state"] = drawdown_state.astype("object")

    # Volatility adjustment flag (cut / neutral / expanded)
    size_vol = df.loc[active_mask, "size_volatility_multiplier"].to_numpy(dtype=float)
    vol_flag = np.full(size_vol.shape, "neutral", dtype=object)
    vol_flag[size_vol < 0.8] = "cut"
    vol_flag[size_vol > 1.1] = "expanded"
    df.loc[active_mask, "volatility_adjustment_flag"] = vol_flag

    # --------------------------------------------------------------
    # Risk mode classification
    # --------------------------------------------------------------
    risk_mode = np.full(size_vol.shape, "normal", dtype=object)
    dd_state_arr = df.loc[active_mask, "drawdown_state"].to_numpy(dtype=object)

    # crisis → severe drawdown
    risk_mode[dd_state_arr == "severe"] = "crisis"

    # defensive → moderate dd, chop, or vol expansion
    chop_flag = pd.to_numeric(
        df_active.get(cfg.col_regime_chop_flag, 0.0), errors="coerce"
    ).fillna(0.0)
    vol_exp_flag = pd.to_numeric(
        df_active.get(cfg.col_regime_vol_expansion_flag, 0.0), errors="coerce"
    ).fillna(0.0)
    defensive_mask = (
        (dd_state_arr == "moderate")
        | (chop_flag.to_numpy(dtype=float) > 0.5)
        | (vol_exp_flag.to_numpy(dtype=float) > 0.5)
    )
    risk_mode[defensive_mask & (risk_mode != "crisis")] = "defensive"

    # aggressive → strong signal + low drawdown
    sig_mult_arr = signal_mult.to_numpy(dtype=float)
    aggressive_mask = (
        (sig_mult_arr > 1.3) & (dd_state_arr == "ok") & (size_vol >= 0.9)
    )
    risk_mode[aggressive_mask & (risk_mode == "normal")] = "aggressive"

    df.loc[active_mask, "risk_mode"] = risk_mode

    # --------------------------------------------------------------
    # Risk-mode-based adjustment of risk_per_trade_frac
    # --------------------------------------------------------------
    mode_arr = df.loc[active_mask, "risk_mode"].to_numpy(dtype=object)
    rpt = df.loc[active_mask, "risk_per_trade_frac"].to_numpy(dtype=float)

    mode_mult = np.ones_like(rpt)
    mode_mult[mode_arr == "crisis"] = 0.4
    mode_mult[mode_arr == "defensive"] = 0.7
    mode_mult[mode_arr == "normal"] = 1.0
    mode_mult[mode_arr == "aggressive"] = 1.0  # currently no extra boost

    rpt = rpt * mode_mult
    rpt = np.clip(rpt, cfg.min_risk_pct, cfg.max_risk_pct)
    df.loc[active_mask, "risk_per_trade_frac"] = rpt.astype("float32")

    # Recompute position_notional / size / leverage after mode gating
    sl_dist_arr_final = df.loc[active_mask, "sl_distance"].to_numpy(dtype=float)
    position_notional_final = rpt / np.clip(
        sl_dist_arr_final, cfg.min_sl_distance, np.inf
    )
    position_leverage_final = np.clip(position_notional_final, 0.0, cfg.max_leverage)
    position_notional_final = position_leverage_final
    position_size_final = position_notional_final

    df.loc[active_mask, "position_notional"] = position_notional_final.astype("float32")
    df.loc[active_mask, "position_size"] = position_size_final.astype("float32")
    df.loc[active_mask, "position_leverage"] = position_leverage_final.astype("float32")

    # Position risk score (0–1, where 1 ~ at max_risk_pct) AFTER risk_mode gating
    pos_risk_score = np.clip(
        rpt / (cfg.max_risk_pct + cfg.eps),
        0.0,
        1.0,
    )
    df.loc[active_mask, "position_risk_score"] = pos_risk_score.astype("float32")

    # Scaling metadata string (per-component multipliers only)
    reg_mult_arr = regime_mult.to_numpy(dtype=float)
    dd_mult_arr = drawdown_mult.to_numpy(dtype=float)
    lev_mult_arr = leverage_mult.to_numpy(dtype=float)
    meta = []
    for sv, sg, rg, ddm, lm in zip(
        size_vol, sig_mult_arr, reg_mult_arr, dd_mult_arr, lev_mult_arr
    ):
        meta.append(
            f"vol={sv:.2f}|sig={sg:.2f}|reg={rg:.2f}|dd={ddm:.2f}|lev={lm:.2f}"
        )
    df.loc[active_mask, "scaling_metadata"] = meta

    _log_mem(df, "[RiskSurface] Output trades_df + full P3 sizing")

    # Some summary stats (after all adjustments)
    active_leverage = df.loc[active_mask, "position_leverage"]
    logging.info(
        "[RiskSurface] Active trades: %d, mean leverage=%.3f, max leverage=%.3f",
        int(active_mask.sum()),
        float(active_leverage.mean()),
        float(active_leverage.max()),
    )
    logging.info(
        "[RiskSurface] risk_per_trade_frac: mean=%.4f, max=%.4f",
        float(df.loc[active_mask, "risk_per_trade_frac"].mean()),
        float(df.loc[active_mask, "risk_per_trade_frac"].max()),
    )
    logging.info(
        "[RiskSurface] size_volatility_multiplier: mean=%.3f, std=%.3f",
        float(df.loc[active_mask, "size_volatility_multiplier"].mean()),
        float(df.loc[active_mask, "size_volatility_multiplier"].std()),
    )
    logging.info(
        "[RiskSurface] signal_alignment_multiplier: mean=%.3f, std=%.3f",
        float(df.loc[active_mask, "signal_alignment_multiplier"].mean()),
        float(df.loc[active_mask, "signal_alignment_multiplier"].std()),
    )

    return df


# Backwards-compat alias if your pipeline still imports this name
build_risk_surface_and_sizing = compute_risk_surface_and_sizing