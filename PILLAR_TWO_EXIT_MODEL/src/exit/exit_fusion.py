# src/exit/exit_fusion.py

"""
Pillar 2 — Fusion Layer

Combines:
  - deterministic exits (from deterministic profile)
  - ML exit model predictions (when available)
  - regime context (Pillar 4)
  - breakout probabilities (Pillar 1)

into a single, stable, regime-aware, confidence-aware exit profile.

Outputs:
  - fused_tp  (relative TP distance, >0)
  - fused_sl  (relative SL distance, >0)
  - fused_tp_first_probability
  - fused_exit_time  (bars)
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class ExitFusionConfig:
    # Core columns
    col_time: str = "open_time_ms"
    col_direction: str = "entry_direction"

    # Deterministic profile inputs
    col_det_tp: str = "deterministic_tp"
    col_det_sl: str = "deterministic_sl"
    col_det_tp_prob: str = "deterministic_tp_first_probability"
    col_det_exit_time: str = "deterministic_exit_time_estimate"

    # --- ML model outputs (preferred if present) ---

    # TP distance
    ml_tp_candidates: Sequence[str] = (
        "predicted_tp_distance",
        "tp_distance_reg_pred",
    )

    # SL distance
    ml_sl_candidates: Sequence[str] = (
        "predicted_sl_distance",
        "sl_distance_reg_pred",
    )

    # TP-first probability
    ml_tp_prob_candidates: Sequence[str] = (
        "predicted_TP_first_probability",  # your current column
        "tp_first_clf_prob",
        "tp_first_clf_prob_tp_first",
    )

    # Exit time
    ml_exit_time_candidates: Sequence[str] = (
        "predicted_exit_time",
        "exit_time_reg_pred",
    )

    # Volatility impact
    ml_vol_impact_candidates: Sequence[str] = (
        "predicted_volatility_effect",
        "vol_impact_reg_pred",
    )

    # Label fallbacks (if ML preds missing)
    col_label_tp: str = "optimal_tp_distance"
    col_label_sl: str = "optimal_sl_distance"
    col_label_exit_time: str = "expected_exit_time"

    # Pillar 1 / regime context
    col_breakout_prob_up: str = "breakout_prob_up"
    col_breakout_prob_down: str = "breakout_prob_down"

    col_regime_trend: str = "regime_is_trend"
    col_regime_chop: str = "regime_is_chop"
    col_regime_mean_rev: str = "regime_is_mean_reversion"
    col_regime_vol_expansion: str = "regime_is_vol_expansion"
    col_regime_liq_drought: str = "regime_is_liq_drought"

    # Fusion hyperparameters
    low_conf: float = 0.10
    high_conf: float = 0.25

    weight_ml_low: float = 0.30
    weight_ml_mid: float = 0.50
    weight_ml_high: float = 0.80

    # Regime multipliers
    trend_tp_mult: float = 1.30
    trend_sl_mult: float = 1.10
    trend_time_mult: float = 1.15

    chop_tp_mult: float = 0.75
    chop_sl_mult: float = 0.80
    chop_time_mult: float = 0.80

    mean_rev_tp_mult: float = 0.70
    mean_rev_sl_mult: float = 0.85
    mean_rev_time_mult: float = 0.80

    vol_exp_tp_mult: float = 1.40
    vol_exp_sl_mult: float = 1.15
    vol_exp_time_mult: float = 1.10

    liq_drought_tp_mult: float = 0.80
    liq_drought_sl_mult: float = 0.95
    liq_drought_time_mult: float = 0.90
    liq_drought_max_rr: float = 2.5

    # Breakout adjustments
    breakout_prob_thr: float = 0.60
    breakout_tp_extra_mult: float = 1.25
    breakout_time_mult: float = 1.10

    # Global RR constraints
    min_rr: float = 0.50
    max_rr: float = 5.00

    # Safety eps
    eps: float = 1e-6


def _require_cols(df: pd.DataFrame, cols: List[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[ExitFusion] Missing required columns for {context}: {missing}")


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def _pick_first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _resolve_ml_or_label(
    df: pd.DataFrame,
    ml_candidates: Sequence[str],
    label_col: str,
    desc: str,
) -> str:
    """
    Prefer one of the ML prediction columns; if none exist, fall back to label.
    """
    ml_col = _pick_first_existing(df, ml_candidates)
    if ml_col is not None:
        return ml_col

    if label_col in df.columns:
        logging.warning(
            "[ExitFusion] No ML column found for %s (%s); falling back to label '%s' "
            "(this is leakage, OK for offline analysis / debugging).",
            desc,
            ", ".join(ml_candidates),
            label_col,
        )
        return label_col

    raise ValueError(
        f"[ExitFusion] Could not resolve column for {desc}. "
        f"Tried ML candidates {list(ml_candidates)} and label '{label_col}'."
    )


def _get_ml_tp_prob(df: pd.DataFrame, cfg: ExitFusionConfig, det_prob: pd.Series) -> pd.Series:
    """
    Resolve which ML probability column to use (robust to naming choice).
    Falls back to deterministic TP prob if none found.
    """
    col = _pick_first_existing(df, cfg.ml_tp_prob_candidates)
    if col is None:
        logging.warning(
            "[ExitFusion] No ML TP-first probability column found; "
            "falling back to deterministic TP-first probability."
        )
        return det_prob.copy()

    return pd.to_numeric(df[col], errors="coerce")


def fuse_exit_signals(
    df: pd.DataFrame,
    cfg: Optional[ExitFusionConfig] = None,
) -> pd.DataFrame:
    """
    Main fusion routine.

    Expects:
      - deterministic_* columns
      - ML preds (predicted_*) if present
      - labels (for fallback)
      - regime flags
      - breakout probabilities
    """
    if cfg is None:
        cfg = ExitFusionConfig()

    # Require deterministic profile + basic trade info
    _require_cols(
        df,
        [
            cfg.col_time,
            cfg.col_direction,
            cfg.col_det_tp,
            cfg.col_det_sl,
            cfg.col_det_tp_prob,
            cfg.col_det_exit_time,
        ],
        context="deterministic profile",
    )

    _log_mem(df, "[ExitFusion] exit_with_ml input")

    # ------------------------------------------------------------------
    # 1) Base deterministic signals
    # ------------------------------------------------------------------
    det_tp = pd.to_numeric(df[cfg.col_det_tp], errors="coerce").clip(lower=cfg.eps)
    det_sl = pd.to_numeric(df[cfg.col_det_sl], errors="coerce").clip(lower=cfg.eps)
    det_prob = pd.to_numeric(df[cfg.col_det_tp_prob], errors="coerce").clip(0.0, 1.0)
    det_time = pd.to_numeric(df[cfg.col_det_exit_time], errors="coerce").clip(lower=1.0)

    # ------------------------------------------------------------------
    # 2) Resolve which columns to use for ML (or fall back to labels)
    # ------------------------------------------------------------------
    ml_tp_col = _resolve_ml_or_label(
        df,
        cfg.ml_tp_candidates,
        cfg.col_label_tp,
        desc="TP distance",
    )
    ml_sl_col = _resolve_ml_or_label(
        df,
        cfg.ml_sl_candidates,
        cfg.col_label_sl,
        desc="SL distance",
    )
    ml_time_col = _resolve_ml_or_label(
        df,
        cfg.ml_exit_time_candidates,
        cfg.col_label_exit_time,
        desc="exit time",
    )

    ml_tp = pd.to_numeric(df[ml_tp_col], errors="coerce").clip(lower=cfg.eps)
    ml_sl = pd.to_numeric(df[ml_sl_col], errors="coerce").clip(lower=cfg.eps)
    ml_time = pd.to_numeric(df[ml_time_col], errors="coerce").clip(lower=1.0)

    ml_prob = _get_ml_tp_prob(df, cfg, det_prob).clip(0.0, 1.0)

    vol_col = _pick_first_existing(df, cfg.ml_vol_impact_candidates)
    if vol_col is not None:
        ml_vol_imp = pd.to_numeric(df[vol_col], errors="coerce").fillna(0.0)
    else:
        ml_vol_imp = pd.Series(0.0, index=df.index)

    # ------------------------------------------------------------------
    # 3) Confidence-based blend between deterministic + ML
    # ------------------------------------------------------------------
    ml_conf = (ml_prob - 0.5).abs()  # how far from confusion

    w_ml = np.where(
        ml_conf >= cfg.high_conf,
        cfg.weight_ml_high,
        np.where(
            ml_conf <= cfg.low_conf,
            cfg.weight_ml_low,
            cfg.weight_ml_mid,
        ),
    )
    w_det = 1.0 - w_ml

    base_tp = (w_det * det_tp + w_ml * ml_tp).astype("float32")
    base_sl = (w_det * det_sl + w_ml * ml_sl).astype("float32")
    base_prob = (w_det * det_prob + w_ml * ml_prob).clip(0.0, 1.0).astype("float32")
    base_time = (w_det * det_time + w_ml * ml_time).astype("float32")

    # Vol impact nudges
    base_tp = (base_tp * (1.0 + 2.0 * ml_vol_imp)).astype("float32")
    base_sl = (base_sl * (1.0 - 1.0 * ml_vol_imp)).astype("float32")

    # ------------------------------------------------------------------
    # 4) Regime-based warping
    # ------------------------------------------------------------------
    def _get_flag(col: str) -> pd.Series:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=df.index)

    regime_trend = _get_flag(cfg.col_regime_trend)
    regime_chop = _get_flag(cfg.col_regime_chop)
    regime_mean_rev = _get_flag(cfg.col_regime_mean_rev)
    regime_vol_exp = _get_flag(cfg.col_regime_vol_expansion)
    regime_liq = _get_flag(cfg.col_regime_liq_drought)

    fused_tp = base_tp.copy()
    fused_sl = base_sl.copy()
    fused_time = base_time.copy()

    # Trend
    mask_trend = regime_trend > 0.5
    fused_tp[mask_trend] *= cfg.trend_tp_mult
    fused_sl[mask_trend] *= cfg.trend_sl_mult
    fused_time[mask_trend] *= cfg.trend_time_mult

    # Chop
    mask_chop = regime_chop > 0.5
    fused_tp[mask_chop] *= cfg.chop_tp_mult
    fused_sl[mask_chop] *= cfg.chop_sl_mult
    fused_time[mask_chop] *= cfg.chop_time_mult

    # Mean-reversion
    mask_mean_rev = regime_mean_rev > 0.5
    fused_tp[mask_mean_rev] *= cfg.mean_rev_tp_mult
    fused_sl[mask_mean_rev] *= cfg.mean_rev_sl_mult
    fused_time[mask_mean_rev] *= cfg.mean_rev_time_mult

    # Vol expansion
    mask_vol_exp = regime_vol_exp > 0.5
    fused_tp[mask_vol_exp] *= cfg.vol_exp_tp_mult
    fused_sl[mask_vol_exp] *= cfg.vol_exp_sl_mult
    fused_time[mask_vol_exp] *= cfg.vol_exp_time_mult

    # Liquidity drought
    mask_liq = regime_liq > 0.5
    fused_tp[mask_liq] *= cfg.liq_drought_tp_mult
    fused_sl[mask_liq] *= cfg.liq_drought_sl_mult
    fused_time[mask_liq] *= cfg.liq_drought_time_mult

    # ------------------------------------------------------------------
    # 5) Breakout-aware tweak
    # ------------------------------------------------------------------
    if cfg.col_breakout_prob_up in df.columns:
        p_up = pd.to_numeric(df[cfg.col_breakout_prob_up], errors="coerce").clip(0.0, 1.0)
    else:
        p_up = pd.Series(0.0, index=df.index)

    if cfg.col_breakout_prob_down in df.columns:
        p_down = pd.to_numeric(df[cfg.col_breakout_prob_down], errors="coerce").clip(0.0, 1.0)
    else:
        p_down = pd.Series(0.0, index=df.index)

    direction = pd.to_numeric(df[cfg.col_direction], errors="coerce").fillna(0.0)
    breakout_aligned = np.where(direction >= 0.0, p_up, p_down)

    mask_breakout = breakout_aligned >= cfg.breakout_prob_thr
    fused_tp[mask_breakout] *= cfg.breakout_tp_extra_mult
    fused_time[mask_breakout] *= cfg.breakout_time_mult

    # ------------------------------------------------------------------
    # 6) Global RR + safety
    # ------------------------------------------------------------------
    fused_tp = fused_tp.clip(lower=cfg.eps)
    fused_sl = fused_sl.clip(lower=cfg.eps)
    fused_time = fused_time.clip(lower=1.0)

    rr = fused_tp / fused_sl
    rr = rr.clip(lower=cfg.min_rr, upper=cfg.max_rr)

    # Extra cap in liq drought
    rr[mask_liq] = rr[mask_liq].clip(upper=cfg.liq_drought_max_rr)

    # Recompute TP from RR & SL
    fused_tp = (fused_sl * rr).astype("float32")

    # Fused TP-first prob: small RR-based tilt
    fused_prob = base_prob * (1.0 + 0.05 * (rr - 1.0))
    fused_prob = fused_prob.clip(0.0, 1.0).astype("float32")

    # ------------------------------------------------------------------
    # 7) Assemble output
    # ------------------------------------------------------------------
    df_out = df.copy()
    df_out["fused_tp"] = fused_tp.astype("float32")
    df_out["fused_sl"] = fused_sl.astype("float32")
    df_out["fused_tp_first_probability"] = fused_prob.astype("float32")
    df_out["fused_exit_time"] = fused_time.astype("float32")

    _log_mem(df_out, "[ExitFusion] exit_with_ml + fused exits")

    logging.info(
        "[ExitFusion] fused_tp: mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
        float(fused_tp.mean()),
        float(fused_tp.std()),
        float(fused_tp.min()),
        float(fused_tp.max()),
    )
    logging.info(
        "[ExitFusion] fused_sl: mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
        float(fused_sl.mean()),
        float(fused_sl.std()),
        float(fused_sl.min()),
        float(fused_sl.max()),
    )
    logging.info(
        "[ExitFusion] fused_tp_first_probability: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
        float(fused_prob.mean()),
        float(fused_prob.std()),
        float(fused_prob.min()),
        float(fused_prob.max()),
    )
    logging.info(
        "[ExitFusion] fused_exit_time: mean=%.2f, std=%.2f, min=%.2f, max=%.2f",
        float(fused_time.mean()),
        float(fused_time.std()),
        float(fused_time.min()),
        float(fused_time.max()),
    )

    return df_out