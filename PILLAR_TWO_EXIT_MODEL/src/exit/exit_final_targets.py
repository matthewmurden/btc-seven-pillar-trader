# src/exit/exit_final_targets.py

"""
Pillar 2 — Final TP/SL Generator

Takes:
  - fused TP/SL distances and probabilities (from fusion layer)
  - entry price / direction / confidence
  - volatility and regime context
  - optional Pillar 3 risk constraints
  - optional Pillar 5 execution priority

Produces actionable, regime-aware TP/SL *levels* to hand off to:
  - Pillar 3 (risk & sizing)
  - Pillar 5 (execution)

Outputs per row:
  - optimal_tp                (absolute price level)
  - optimal_sl                (absolute price level)
  - optimal_tp_distance       (final TP distance, fraction of price)
  - optimal_sl_distance       (final SL distance, fraction of price)
  - r_multiple                (optimal_tp_distance / optimal_sl_distance)
  - tp_first_probability      (prob TP hit before SL; from fusion)
  - expected_exit_time        (bars; from fusion)
  - sl_risk_factor            (how much SL was scaled vs fused base)
  - tp_extension_factor       (how much TP was scaled vs fused base)

NOTE:
  This pillar NO LONGER decides which trades are eligible to trade.
  All trade-level safety / filters (including the old system_filter_v1)
  now live in Pillar 6 (safety layer).
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class ExitFinalConfig:
    # Core time / price / direction
    col_time: str = "open_time_ms"
    col_entry_price: str = "close"  # entry bar close
    col_direction: str = "entry_direction"
    col_decision: str = "entry_decision"

    # Fused distances & probabilities (from fusion layer)
    col_fused_tp: str = "fused_tp"
    col_fused_sl: str = "fused_sl"
    col_fused_tp_prob: str = "fused_tp_first_probability"
    col_fused_exit_time: str = "fused_exit_time"

    # Volatility / regime context (from Pillar 1 / Pillar 4)
    col_atr: str = "vol_atr_14"
    col_regime_trend: str = "regime_is_trend"
    col_regime_chop: str = "regime_is_chop"
    col_regime_meanrev: str = "regime_is_mean_reversion"
    col_regime_volexp: str = "regime_is_vol_expansion"
    col_regime_liq_drought: str = "regime_is_liq_drought"
    col_regime_conf: str = "regime_confidence"
    # Stable regime ID (for diagnostics; no filtering here)
    col_regime_id_stable: Optional[str] = "regime_id_stable"

    # Pillar 1 confidence / probabilities
    col_entry_conf: str = "entry_confidence"
    col_entry_prob_up: str = "entry_prob_up"

    # Optional Pillar 3 risk model constraint (scalar)
    # e.g. p3_risk_scalar in [0.5, 1.5]
    col_p3_risk_scalar: Optional[str] = None

    # Optional Pillar 5 execution priority / urgency (0–1)
    col_p5_exec_priority: Optional[str] = None

    # Bounds / safety rails for base distances
    min_tp_dist: float = 1e-4   # 1 bp
    max_tp_dist: float = 0.25   # 25%
    min_sl_dist: float = 1e-4   # 1 bp
    max_sl_dist: float = 0.10   # 10%

    # How strongly to adjust using confidence
    # final_factor ≈ 1 + conf_alpha * (entry_conf - conf_center)
    conf_alpha_tp: float = 0.75
    conf_alpha_sl: float = 0.50
    conf_center: float = 0.5

    # Regime multipliers (relative to fused baseline)
    trend_tp_boost: float = 1.25
    trend_sl_widen: float = 1.10

    chop_tp_compress: float = 0.70
    chop_sl_tighten: float = 0.85

    meanrev_tp_compress: float = 0.65
    meanrev_sl_tighten: float = 0.95

    volexp_tp_boost: float = 1.35
    volexp_sl_widen: float = 1.20

    liq_tp_cap_factor: float = 0.65
    liq_sl_conservative: float = 1.15

    # Exec priority effect (Pillar 5)
    # factor ~= 1 + exec_alpha * (priority - 0.5)
    exec_alpha_tp: float = 0.40
    exec_alpha_sl: float = 0.10

    # Risk scalar bounds (if Pillar 3 provides a scalar)
    min_risk_scalar: float = 0.5
    max_risk_scalar: float = 1.5

    # Final factor clamps
    min_tp_factor: float = 0.4
    max_tp_factor: float = 3.0
    min_sl_factor: float = 0.5
    max_sl_factor: float = 3.0

    # Numerical eps
    eps: float = 1e-8


def _require_cols(df: pd.DataFrame, cols: List[str], context: str) -> None:
    cols = [c for c in cols if c is not None]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[ExitFinal] Missing required columns for {context}: {missing}")


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024**2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def build_final_exit_targets(df: pd.DataFrame, cfg: Optional[ExitFinalConfig] = None) -> pd.DataFrame:
    """
    Build final, actionable TP/SL *levels* from fused distances + context.

    Parameters
    ----------
    df : pd.DataFrame
        Output of fusion layer (exit_with_fusion), one row per entry bar
        / active trade.
    cfg : ExitFinalConfig, optional
        Column names and scaling parameters.

    Returns
    -------
    df_out : pd.DataFrame
        df with additional columns:
          - optimal_tp
          - optimal_sl
          - optimal_tp_distance
          - optimal_sl_distance
          - r_multiple
          - tp_first_probability
          - expected_exit_time
          - sl_risk_factor
          - tp_extension_factor

        NOTE: No eligibility filters are applied here. All safety /
        eligibility tagging is handled by Pillar 6.
    """
    if cfg is None:
        cfg = ExitFinalConfig()

    # ------------------------------------------------------------------
    # Basic requirements
    # ------------------------------------------------------------------
    _require_cols(
        df,
        [
            cfg.col_time,
            cfg.col_entry_price,
            cfg.col_direction,
            cfg.col_decision,
            cfg.col_fused_tp,
            cfg.col_fused_sl,
            cfg.col_fused_tp_prob,
            cfg.col_fused_exit_time,
        ],
        context="final exit targets",
    )

    _log_mem(df, "[ExitFinal] Input (fusion)")

    # Ensure numeric base arrays
    entry_price = pd.to_numeric(df[cfg.col_entry_price], errors="coerce").to_numpy(dtype=float)
    direction = pd.to_numeric(df[cfg.col_direction], errors="coerce").to_numpy(dtype=float)
    decision = df[cfg.col_decision].astype(str).to_numpy()

    fused_tp = pd.to_numeric(df[cfg.col_fused_tp], errors="coerce").to_numpy(dtype=float)
    fused_sl = pd.to_numeric(df[cfg.col_fused_sl], errors="coerce").to_numpy(dtype=float)

    fused_tp_prob = pd.to_numeric(df[cfg.col_fused_tp_prob], errors="coerce").to_numpy(dtype=float)
    fused_exit_time = pd.to_numeric(df[cfg.col_fused_exit_time], errors="coerce").to_numpy(dtype=float)

    n = len(df)

    # ------------------------------------------------------------------
    # 1) Start from bounded fused distances
    # ------------------------------------------------------------------
    tp_dist = np.clip(fused_tp, cfg.min_tp_dist, cfg.max_tp_dist)
    sl_dist = np.clip(fused_sl, cfg.min_sl_dist, cfg.max_sl_dist)

    # ------------------------------------------------------------------
    # 2) Initialise factors (1.0 = no change from fused)
    # ------------------------------------------------------------------
    tp_factor = np.ones(n, dtype=float)
    sl_factor = np.ones(n, dtype=float)

    # ------------------------------------------------------------------
    # 3) Confidence-based scaling
    # ------------------------------------------------------------------
    if cfg.col_entry_conf in df.columns:
        conf = pd.to_numeric(df[cfg.col_entry_conf], errors="coerce").fillna(cfg.conf_center)
        conf = conf.clip(0.0, 1.0).to_numpy(dtype=float)
        conf_delta = conf - cfg.conf_center

        tp_factor *= 1.0 + cfg.conf_alpha_tp * conf_delta
        # high conf → relatively looser SL, low conf → tighter SL
        sl_factor *= 1.0 + cfg.conf_alpha_sl * (-conf_delta)
    else:
        logging.warning(
            "[ExitFinal] entry_confidence column %s missing; skipping confidence scaling.",
            cfg.col_entry_conf,
        )

    # ------------------------------------------------------------------
    # 4) Regime-conditioned scaling
    # ------------------------------------------------------------------
    def _get_flag(col_name: str) -> np.ndarray:
        if col_name in df.columns:
            return pd.to_numeric(df[col_name], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return np.zeros(n, dtype=float)

    trend_flag = _get_flag(cfg.col_regime_trend)
    chop_flag = _get_flag(cfg.col_regime_chop)
    meanrev_flag = _get_flag(cfg.col_regime_meanrev)
    volexp_flag = _get_flag(cfg.col_regime_volexp)
    liq_flag = _get_flag(cfg.col_regime_liq_drought)

    # Trend regime → extend winners a bit, allow slightly wider SL
    tp_factor *= 1.0 + trend_flag * (cfg.trend_tp_boost - 1.0)
    sl_factor *= 1.0 + trend_flag * (cfg.trend_sl_widen - 1.0)

    # Chop regime → compress TP/SL
    tp_factor *= 1.0 + chop_flag * (cfg.chop_tp_compress - 1.0)
    sl_factor *= 1.0 + chop_flag * (cfg.chop_sl_tighten - 1.0)

    # Mean reversion regime → compress TP more than SL
    tp_factor *= 1.0 + meanrev_flag * (cfg.meanrev_tp_compress - 1.0)
    sl_factor *= 1.0 + meanrev_flag * (cfg.meanrev_sl_tighten - 1.0)

    # Vol expansion → widen both
    tp_factor *= 1.0 + volexp_flag * (cfg.volexp_tp_boost - 1.0)
    sl_factor *= 1.0 + volexp_flag * (cfg.volexp_sl_widen - 1.0)

    # Low liquidity → cap TP extension & be conservative on SL
    if np.any(liq_flag != 0.0):
        tp_factor *= 1.0 - liq_flag * (1.0 - cfg.liq_tp_cap_factor)
        sl_factor *= 1.0 + liq_flag * (cfg.liq_sl_conservative - 1.0)

    # ------------------------------------------------------------------
    # 5) Pillar 3 risk scalar (optional)
    # ------------------------------------------------------------------
    if cfg.col_p3_risk_scalar and cfg.col_p3_risk_scalar in df.columns:
        risk_scalar = pd.to_numeric(df[cfg.col_p3_risk_scalar], errors="coerce").fillna(1.0)
        risk_scalar = risk_scalar.clip(cfg.min_risk_scalar, cfg.max_risk_scalar).to_numpy(dtype=float)
        # Risk scalar primarily affects SL (how tight/wide), but we can bleed a little into TP.
        sl_factor *= risk_scalar
        tp_factor *= np.sqrt(risk_scalar)  # gentle effect on TP
    elif cfg.col_p3_risk_scalar:
        logging.warning(
            "[ExitFinal] Pillar 3 risk scalar column %s missing; skipping risk-model scaling.",
            cfg.col_p3_risk_scalar,
        )

    # ------------------------------------------------------------------
    # 6) Pillar 5 execution priority (optional)
    # ------------------------------------------------------------------
    if cfg.col_p5_exec_priority and cfg.col_p5_exec_priority in df.columns:
        priority = pd.to_numeric(df[cfg.col_p5_exec_priority], errors="coerce").fillna(0.5)
        priority = priority.clip(0.0, 1.0).to_numpy(dtype=float)
        pr_delta = priority - 0.5
        # High priority / urgency → favour slightly closer TP and slightly tighter SL
        tp_factor *= 1.0 - cfg.exec_alpha_tp * pr_delta
        sl_factor *= 1.0 - cfg.exec_alpha_sl * pr_delta
    elif cfg.col_p5_exec_priority:
        logging.warning(
            "[ExitFinal] Pillar 5 execution priority column %s missing; skipping exec scaling.",
            cfg.col_p5_exec_priority,
        )

    # ------------------------------------------------------------------
    # 7) Clamp final factors and build distances
    # ------------------------------------------------------------------
    tp_factor = np.clip(tp_factor, cfg.min_tp_factor, cfg.max_tp_factor)
    sl_factor = np.clip(sl_factor, cfg.min_sl_factor, cfg.max_sl_factor)

    final_tp_dist = np.clip(tp_dist * tp_factor, cfg.min_tp_dist, cfg.max_tp_dist)
    final_sl_dist = np.clip(sl_dist * sl_factor, cfg.min_sl_dist, cfg.max_sl_dist)

    # ------------------------------------------------------------------
    # 8) Convert distances to *price levels* by side
    # ------------------------------------------------------------------
    optimal_tp = np.full(n, np.nan, dtype=float)
    optimal_sl = np.full(n, np.nan, dtype=float)

    # Side masks
    is_long = direction > 0
    is_short = direction < 0
    active = (decision != "none") & np.isfinite(entry_price)

    # Longs: TP = price * (1 + dTP), SL = price * (1 - dSL)
    mask_long = active & is_long
    optimal_tp[mask_long] = entry_price[mask_long] * (1.0 + final_tp_dist[mask_long])
    optimal_sl[mask_long] = entry_price[mask_long] * (1.0 - final_sl_dist[mask_long])

    # Shorts: TP = price * (1 - dTP), SL = price * (1 + dSL)
    mask_short = active & is_short
    optimal_tp[mask_short] = entry_price[mask_short] * (1.0 - final_tp_dist[mask_short])
    optimal_sl[mask_short] = entry_price[mask_short] * (1.0 + final_sl_dist[mask_short])

    # ------------------------------------------------------------------
    # 9) Build output DataFrame with new columns
    # ------------------------------------------------------------------
    out = df.copy()

    out["optimal_tp"] = optimal_tp.astype("float64")
    out["optimal_sl"] = optimal_sl.astype("float64")
    out["tp_first_probability"] = fused_tp_prob.astype("float64")
    out["expected_exit_time"] = fused_exit_time.astype("float64")
    out["sl_risk_factor"] = sl_factor.astype("float32")
    out["tp_extension_factor"] = tp_factor.astype("float32")

    # Expose final distances and R-multiple for diagnostics / P3 / backtests
    out["optimal_tp_distance"] = final_tp_dist.astype("float32")
    out["optimal_sl_distance"] = final_sl_dist.astype("float32")
    r_multiple = final_tp_dist / np.maximum(final_sl_dist, cfg.eps)
    out["r_multiple"] = r_multiple.astype("float32")

    # ------------------------------------------------------------------
    # 10) Logging stats (purely diagnostic now)
    # ------------------------------------------------------------------
    valid_mask = active & np.isfinite(optimal_tp) & np.isfinite(optimal_sl)
    if valid_mask.any():
        rr_valid = r_multiple[valid_mask]
        logging.info(
            "[ExitFinal] Final distances: TP mean=%.5f, SL mean=%.5f, RR mean=%.3f",
            float(final_tp_dist[valid_mask].mean()),
            float(final_sl_dist[valid_mask].mean()),
            float(rr_valid.mean()),
        )
        logging.info(
            "[ExitFinal] TP levels: mean=%.2f, SL levels: mean=%.2f (on active trades)",
            float(optimal_tp[valid_mask].mean()),
            float(optimal_sl[valid_mask].mean()),
        )
    else:
        logging.warning("[ExitFinal] No valid trades with non-NaN optimal TP/SL found.")

    return out