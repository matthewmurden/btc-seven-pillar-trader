"""
Pillar 1 — Regime-Weighted Master Aggregator

This module integrates:
- Mini-engine scores (momentum/trend/vol/reversion/orderflow/structure/pattern)
- Multi-horizon ML outputs (short_dir, dir_5_10, dir_15_30, breakout, mean_reversion)
- Meta-features (agreement, conflict, regime-aligned summaries)
- Regime state (trend/chop/vol-expansion/liquidity-drought/mean-reversion)

Into a single unified directional signal:

    raw_entry_confidence ∈ [-1, 1]

Design principles:
- Direction is always:  +ve → long bias,  -ve → short bias.
- Regime steers *weights*, not labels (no regime leakage into targets).
- Strong gating in volatility compression & liquidity drought.
- Uses tanh squashing for stability and interpretable [-1, 1] range.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class MasterAggregatorConfig:
    # Deterministic engines: base weights
    w_momentum: float = 0.35
    w_trend: float = 0.40
    w_volatility: float = 0.05
    w_reversion: float = 0.30
    w_orderflow: float = 0.20
    w_structure: float = 0.20
    w_pattern: float = 0.05

    # ML directional weights (per horizon)
    w_ml_short: float = 0.40      # 1–2 bars
    w_ml_5_10: float = 0.35       # 5–10 bars
    w_ml_15_30: float = 0.25      # 15–30 bars
    w_ml_global: float = 0.60     # overall scale for ML composite

    # Breakout / mean-reversion models
    w_breakout_dir: float = 0.40      # breakout directional contribution
    w_mean_reversion_strength: float = 0.30  # scales reversion block via prob

    # Meta-feature weights
    w_meta_agreement: float = 0.50
    w_meta_conflict: float = -0.60
    w_meta_vol_weighted_mom: float = 0.25
    w_meta_structure_aligned_mom: float = 0.25
    w_meta_regime_trend_conf: float = 0.30
    w_meta_breakout_align: float = 0.35
    w_meta_horizon_consistency: float = 0.30
    w_meta_directional_persistence: float = 0.30

    # Regime-driven re-weighting
    trend_regime_boost: float = 0.75        # boost trend+momentum in trend regimes
    trend_chop_suppress: float = 0.75       # suppress trend in chop
    rev_chop_boost: float = 0.60           # boost reversion in chop
    rev_mr_boost: float = 0.80             # boost reversion in MR regime
    rev_mr_prob_boost: float = 0.60        # boost reversion using MR model prob
    breakout_volexp_boost: float = 0.80    # boost breakout in volatility expansion

    # Global gating
    vol_compression_suppress: float = 0.80  # suppress all direction in vol compression
    liq_drought_kill: float = 1.00          # 1.0 → fully kill signal in liq drought

    # Tanh squashing
    base_scale: float = 2.0            # larger → flatter tanh, smaller → more saturation
    max_abs_clip: float = 1.0          # final hard clip in [-max_abs_clip, max_abs_clip]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _get_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    """
    Safe getter with numeric coercion and NaN fill.
    Logs a warning once per missing column name.
    """
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    logging.warning("[Agg] Column %s missing; using default=%s", col, default)
    return pd.Series(default, index=df.index, dtype=float)


def _directional_from_probs(df: pd.DataFrame, prefix: str) -> pd.Series:
    """
    Turn (prob_up, prob_down, prob_neutral) into a single directional score in [-1, 1].

    Formula:
        dir = (prob_up - prob_down) * (1 - prob_neutral)

    This shrinks the magnitude when the model is "neutral-heavy".
    """
    up = _get_series(df, f"{prefix}_prob_up", default=0.5)
    down = _get_series(df, f"{prefix}_prob_down", default=0.5)
    neutral = _get_series(df, f"{prefix}_prob_neutral", default=0.0)
    dir_score = (up - down) * (1.0 - neutral).clip(0.0, 1.0)
    return dir_score.astype(float)


def _tanh_squash(x: pd.Series, scale: float, max_abs: float) -> pd.Series:
    y = np.tanh(x.to_numpy() / (scale + 1e-9))
    y = np.clip(y, -max_abs, max_abs)
    return pd.Series(y, index=x.index, dtype=float)


# ---------------------------------------------------------------------
# Core aggregator
# ---------------------------------------------------------------------
def compute_master_aggregator(
    df: pd.DataFrame,
    cfg: Optional[MasterAggregatorConfig] = None,
) -> pd.DataFrame:
    """
    Compute raw_entry_confidence and diagnostic components in-place.

    Adds:
      - raw_entry_confidence
      - agg_trend_block
      - agg_reversion_block
      - agg_ml_block
      - agg_breakout_block
      - agg_meta_block
      - agg_global_gate
    """
    if cfg is None:
        cfg = MasterAggregatorConfig()

    idx = df.index

    # ----------------------------
    # 1) Deterministic engine blocks
    # ----------------------------
    momentum = _get_series(df, "momentum_score")
    trend = _get_series(df, "trend_score")
    vol = _get_series(df, "volatility_score")
    reversion = _get_series(df, "reversion_score")
    orderflow = _get_series(df, "orderflow_score")
    structure = _get_series(df, "structure_score")
    pattern = _get_series(df, "pattern_score")

    # Base linear combos (directional)
    det_trend_block = cfg.w_momentum * momentum + cfg.w_trend * trend
    det_reversion_block = cfg.w_reversion * reversion + cfg.w_structure * structure
    det_ofpatt_block = cfg.w_orderflow * orderflow + cfg.w_pattern * pattern
    det_vol_block = cfg.w_volatility * vol

    # ----------------------------
    # 2) ML directional blocks
    # ----------------------------
    short_dir = _directional_from_probs(df, "short_dir")
    dir_5_10 = _directional_from_probs(df, "dir_5_10")
    dir_15_30 = _directional_from_probs(df, "dir_15_30")

    ml_dir = (
        cfg.w_ml_short * short_dir
        + cfg.w_ml_5_10 * dir_5_10
        + cfg.w_ml_15_30 * dir_15_30
    )

    # Breakout directional: binary trend continuation flavour
    breakout_dir = _directional_from_probs(df, "breakout")

    # Mean-reversion strength (NOT directly directional; used to boost reversion)
    mr_prob_up = _get_series(df, "mean_reversion_prob_up", default=0.0)
    mr_strength = ((mr_prob_up - 0.5) * 2.0).clip(-1.0, 1.0)

    # ----------------------------
    # 3) Meta-features
    # ----------------------------
    meta_agree = _get_series(df, "meta_agreement_score")
    meta_conf = _get_series(df, "meta_conflict_score")
    meta_vol_mom = _get_series(df, "meta_vol_weighted_mom")
    meta_struct_mom = _get_series(df, "meta_structure_aligned_mom")
    meta_trend_conf = _get_series(df, "meta_regime_trend_confirmation")
    meta_brk_align = _get_series(df, "meta_breakout_alignment")
    meta_h_cons = _get_series(df, "meta_horizon_consistency")
    meta_persist = _get_series(df, "meta_directional_persistence")

    meta_term = (
        cfg.w_meta_agreement * meta_agree
        + cfg.w_meta_conflict * meta_conf
        + cfg.w_meta_vol_weighted_mom * meta_vol_mom
        + cfg.w_meta_structure_aligned_mom * meta_struct_mom
        + cfg.w_meta_regime_trend_conf * meta_trend_conf
        + cfg.w_meta_breakout_align * meta_brk_align
        + cfg.w_meta_horizon_consistency * meta_h_cons
        + cfg.w_meta_directional_persistence * meta_persist
    )

    # ----------------------------
    # 4) Regime state & gating
    # ----------------------------
    regime_conf = _get_series(df, "regime_confidence", default=0.0).clip(0.0, 1.0)
    is_trend = _get_series(df, "regime_is_trend", default=0.0)
    is_chop = _get_series(df, "regime_is_chop", default=0.0)
    is_mr = _get_series(df, "regime_is_mean_reversion", default=0.0)
    is_vol_expansion = _get_series(df, "regime_is_vol_expansion", default=0.0)
    is_liq_drought = _get_series(df, "regime_is_liq_drought", default=0.0)

    if "regime_id_stable" in df.columns:
        regime_id = pd.to_numeric(df["regime_id_stable"], errors="coerce").fillna(-1)
    else:
        regime_id = pd.Series(-1, index=idx, dtype=float)

    # volatility_compression is regime_id == 5 in your mapping
    vol_compression_mask = (regime_id == 5).astype(float)

    # --- Component-specific regime weights ---

    # Trend/momentum block:
    trend_block_weight = (
        1.0
        + cfg.trend_regime_boost * is_trend * regime_conf
        - cfg.trend_chop_suppress * is_chop * regime_conf
    )

    # Reversion/structure block:
    rev_block_weight = (
        1.0
        + cfg.rev_chop_boost * is_chop * regime_conf
        + cfg.rev_mr_boost * is_mr * regime_conf
        + cfg.rev_mr_prob_boost * mr_strength.clip(lower=0.0)  # only boost when MR prob > 0.5
    )

    # Breakout block:
    breakout_weight = 1.0 + cfg.breakout_volexp_boost * is_vol_expansion * regime_conf

    # ML gate from meta-agreement/conflict
    agreement_gate = ((meta_agree + 1.0) / 2.0).clip(0.0, 1.0)  # [-1,1] → [0,1]
    conflict_gate = (1.0 - meta_conf.clip(0.0, 1.0) ** 2).clip(0.0, 1.0)
    ml_gate = (0.5 + 0.5 * agreement_gate) * conflict_gate  # emphasise ML when agreement high & conflict low

    # Global gates
    vol_compression_gate = (1.0 - cfg.vol_compression_suppress * vol_compression_mask).clip(0.0, 1.0)
    liq_gate = (1.0 - cfg.liq_drought_kill * is_liq_drought.clip(0.0, 1.0)).clip(0.0, 1.0)
    global_gate = (vol_compression_gate * liq_gate).clip(0.0, 1.0)

    # ----------------------------
    # 5) Build final linear combination
    # ----------------------------
    agg_trend_block = trend_block_weight * det_trend_block
    agg_reversion_block = rev_block_weight * det_reversion_block
    agg_ofpatt_block = det_ofpatt_block
    agg_vol_block = det_vol_block
    agg_ml_block = cfg.w_ml_global * ml_dir * ml_gate
    agg_breakout_block = cfg.w_breakout_dir * breakout_dir * breakout_weight
    agg_meta_block = meta_term

    raw_linear = (
        agg_trend_block
        + agg_reversion_block
        + agg_ofpatt_block
        + agg_vol_block
        + agg_ml_block
        + agg_breakout_block
        + agg_meta_block
    )

    raw_linear_gated = raw_linear * global_gate

    # ----------------------------
    # 6) Squash to [-1, 1] and attach
    # ----------------------------
    raw_entry_confidence = _tanh_squash(
        raw_linear_gated,
        scale=cfg.base_scale,
        max_abs=cfg.max_abs_clip,
    )

    df["raw_entry_confidence"] = raw_entry_confidence.astype(float)

    # Diagnostics
    df["agg_trend_block"] = agg_trend_block.astype(float)
    df["agg_reversion_block"] = agg_reversion_block.astype(float)
    df["agg_ml_block"] = agg_ml_block.astype(float)
    df["agg_breakout_block"] = agg_breakout_block.astype(float)
    df["agg_meta_block"] = agg_meta_block.astype(float)
    df["agg_global_gate"] = global_gate.astype(float)

    return df

