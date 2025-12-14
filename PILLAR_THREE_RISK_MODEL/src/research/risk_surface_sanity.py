# src/research/risk_surface_sanity.py

"""
Sanity script for Pillar 3 — Risk Model (Position Sizing).

Loads:
  - PILLAR_TWO_EXIT_MODEL/data/processed/exit_with_sizing.parquet

Checks:
  - Core column presence
  - Risk per trade range and distribution
  - Leverage range and distribution
  - Multiplier ranges (volatility, signal, regime, drawdown, leverage, exposure)
  - Risk modes / drawdown states / volatility flags distributions
  - Consistency between risk_per_trade_frac and position_risk_score

Run:
  python -m src.research.risk_surface_sanity
"""

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.risk.risk_surface import RiskSurfaceConfig


BASE_DIR = Path(__file__).resolve().parents[3]
P2_BASE = BASE_DIR / "PILLAR_TWO_EXIT_MODEL"
INPUT_PATH = P2_BASE / "data" / "processed" / "exit_with_sizing.parquet"


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def _check_columns(df: pd.DataFrame, cols: List[str], label: str) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logging.error("[FAIL] Missing %s columns: %s", label, missing)
        return False
    logging.info("[PASS] All %s columns present.", label)
    return True


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    cfg = RiskSurfaceConfig()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"exit_with_sizing parquet not found: {INPUT_PATH}")

    logging.info("[RiskSurfaceSanity] Loading sized trades: %s", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)
    _log_mem(df, "[RiskSurfaceSanity] Loaded")

    n_rows, n_cols = df.shape
    logging.info("[RiskSurfaceSanity] n_rows=%d, n_cols=%d", n_rows, n_cols)

    if n_rows == 0:
        logging.error("[FAIL] exit_with_sizing has zero rows.")
        return
    else:
        logging.info("[PASS] File has at least one row.")

    fails = 0
    warns = 0

    # --------------------------------------------------------------
    # Core column presence
    # --------------------------------------------------------------
    core_cols = [
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
    ]
    if not _check_columns(df, core_cols, "core P3"):
        fails += 1

    # Active trades as per sizing: risk_per_trade_frac > 0
    active_mask = df["risk_per_trade_frac"] > 0
    n_active = int(active_mask.sum())
    logging.info("[RiskSurfaceSanity] Active sized trades (risk_per_trade_frac>0): %d", n_active)

    if n_active == 0:
        logging.error("[FAIL] No trades with positive risk_per_trade_frac.")
        fails += 1

    df_active = df.loc[active_mask].copy()

    # --------------------------------------------------------------
    # Risk per trade statistics
    # --------------------------------------------------------------
    r = pd.to_numeric(df_active["risk_per_trade_frac"], errors="coerce")
    r_min, r_max, r_mean = float(r.min()), float(r.max()), float(r.mean())
    r_med = float(r.median())
    r_q95 = float(r.quantile(0.95))

    logging.info(
        "[RiskSurfaceSanity] risk_per_trade_frac: "
        "min=%.5f, max=%.5f, mean=%.5f, median=%.5f, p95=%.5f",
        r_min,
        r_max,
        r_mean,
        r_med,
        r_q95,
    )

    if r_min < 0:
        logging.error("[FAIL] risk_per_trade_frac has negative values (min=%.6f).", r_min)
        fails += 1
    else:
        logging.info("[PASS] risk_per_trade_frac is non-negative.")

    # Allow a tiny epsilon above max_risk_pct for numerical noise
    tol = cfg.max_risk_pct * 0.05
    if r_max > cfg.max_risk_pct + tol:
        logging.error(
            "[FAIL] risk_per_trade_frac exceeds max_risk_pct (max=%.5f, cfg.max_risk_pct=%.5f).",
            r_max,
            cfg.max_risk_pct,
        )
        fails += 1
    else:
        logging.info(
            "[PASS] risk_per_trade_frac <= max_risk_pct + tol "
            "(max=%.5f, cfg.max_risk_pct=%.5f).",
            r_max,
            cfg.max_risk_pct,
        )

    # --------------------------------------------------------------
    # Leverage statistics
    # --------------------------------------------------------------
    lev = pd.to_numeric(df_active["position_leverage"], errors="coerce")
    lev_min, lev_max, lev_mean = float(lev.min()), float(lev.max()), float(lev.mean())
    lev_med = float(lev.median())
    logging.info(
        "[RiskSurfaceSanity] position_leverage: min=%.3f, max=%.3f, mean=%.3f, median=%.3f",
        lev_min,
        lev_max,
        lev_mean,
        lev_med,
    )

    if lev_min < 0:
        logging.error("[FAIL] position_leverage has negative values (min=%.6f).", lev_min)
        fails += 1
    else:
        logging.info("[PASS] position_leverage is non-negative.")

    if lev_max > cfg.max_leverage + 1e-6:
        logging.error(
            "[FAIL] position_leverage exceeds cfg.max_leverage (max=%.3f, cfg=%.3f).",
            lev_max,
            cfg.max_leverage,
        )
        fails += 1
    else:
        logging.info(
            "[PASS] position_leverage <= cfg.max_leverage (max=%.3f, cfg=%.3f).",
            lev_max,
            cfg.max_leverage,
        )

    # --------------------------------------------------------------
    # Multiplier ranges
    # --------------------------------------------------------------
    def check_mult(col: str, lo: float, hi: float) -> None:
        nonlocal fails, warns
        if col not in df_active.columns:
            logging.error("[FAIL] Multiplier column missing: %s", col)
            fails += 1
            return

        x = pd.to_numeric(df_active[col], errors="coerce")
        x_min, x_max, x_mean, x_std = float(x.min()), float(x.max()), float(x.mean()), float(x.std())
        logging.info(
            "[RiskSurfaceSanity] %s: min=%.3f, max=%.3f, mean=%.3f, std=%.3f",
            col,
            x_min,
            x_max,
            x_mean,
            x_std,
        )

        tol_lo = lo - 0.05
        tol_hi = hi + 0.05

        if x_min < tol_lo or x_max > tol_hi:
            logging.error(
                "[FAIL] %s outside expected range (min=%.3f, max=%.3f, expected≈[%.3f, %.3f]).",
                col,
                x_min,
                x_max,
                lo,
                hi,
            )
            fails += 1
        elif x_min < lo or x_max > hi:
            logging.warning(
                "[WARN] %s slightly outside nominal range (min=%.3f, max=%.3f, expected≈[%.3f, %.3f]).",
                col,
                x_min,
                x_max,
                lo,
                hi,
            )
            warns += 1
        else:
            logging.info(
                "[PASS] %s within expected range [%.3f, %.3f].", col, lo, hi
            )

    check_mult(
        "size_volatility_multiplier",
        cfg.vol_size_mult_min,
        cfg.vol_size_mult_max,
    )
    check_mult(
        "signal_alignment_multiplier",
        cfg.signal_mult_min,
        cfg.signal_mult_max,
    )
    check_mult("regime_multiplier", cfg.regime_mult_min, cfg.regime_mult_max)
    check_mult("drawdown_multiplier", cfg.drawdown_mult_min, cfg.drawdown_mult_max)
    check_mult("leverage_multiplier", cfg.leverage_mult_min, cfg.leverage_mult_max)
    check_mult("exposure_cap_multiplier", cfg.exposure_mult_min, cfg.exposure_mult_max)

    # --------------------------------------------------------------
    # Check combined multiplier and position_size_multiplier
    # --------------------------------------------------------------
    if "position_size_multiplier" in df_active.columns:
        pm = pd.to_numeric(
            df_active["position_size_multiplier"], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        pm_min, pm_max, pm_mean, pm_std = (
            float(pm.min()),
            float(pm.max()),
            float(pm.mean()),
            float(pm.std()),
        )
        logging.info(
            "[RiskSurfaceSanity] position_size_multiplier: "
            "min=%.3f, max=%.3f, mean=%.3f, std=%.3f",
            pm_min,
            pm_max,
            pm_mean,
            pm_std,
        )
        if not np.isfinite(pm_min) or not np.isfinite(pm_max):
            logging.error("[FAIL] position_size_multiplier contains non-finite values.")
            fails += 1
        elif pm_min <= 0:
            logging.error(
                "[FAIL] position_size_multiplier has non-positive values (min=%.6f).",
                pm_min,
            )
            fails += 1
        else:
            logging.info("[PASS] position_size_multiplier positive and finite.")
    else:
        logging.error("[FAIL] position_size_multiplier missing.")
        fails += 1

    # --------------------------------------------------------------
    # Risk mode / drawdown / volatility flag distributions
    # --------------------------------------------------------------
    def log_value_counts(col: str, allowed: list) -> None:
        nonlocal fails, warns
        if col not in df_active.columns:
            logging.error("[FAIL] Missing categorical column: %s", col)
            fails += 1
            return
        vc = df_active[col].astype(str).value_counts().to_dict()
        logging.info("[RiskSurfaceSanity] %s distribution: %s", col, vc)
        extra = set(vc.keys()) - set(allowed)
        if extra:
            logging.error(
                "[FAIL] %s has unexpected values: %s (allowed=%s)",
                col,
                extra,
                allowed,
            )
            fails += 1
        else:
            logging.info("[PASS] %s only contains allowed states.", col)

    log_value_counts(
        "drawdown_state",
        ["none", "ok", "mild", "moderate", "severe"],
    )
    log_value_counts(
        "volatility_adjustment_flag",
        ["cut", "neutral", "expanded"],
    )
    log_value_counts(
        "risk_mode",
        ["", "normal", "defensive", "aggressive", "crisis"],
    )

    # --------------------------------------------------------------
    # position_risk_score consistency
    # --------------------------------------------------------------
    prs = pd.to_numeric(df_active["position_risk_score"], errors="coerce")
    prs_min, prs_max, prs_mean = float(prs.min()), float(prs.max()), float(prs.mean())
    logging.info(
        "[RiskSurfaceSanity] position_risk_score: min=%.3f, max=%.3f, mean=%.3f",
        prs_min,
        prs_max,
        prs_mean,
    )

    if prs_min < -1e-3 or prs_max > 1.1:
        logging.error(
            "[FAIL] position_risk_score outside [0,1.1] (min=%.3f, max=%.3f).",
            prs_min,
            prs_max,
        )
        fails += 1
    else:
        logging.info("[PASS] position_risk_score reasonably in [0,1].")

    # Check correlation with risk_per_trade_frac/max_risk_pct
    r_scaled = r.to_numpy(dtype=float) / (cfg.max_risk_pct + 1e-12)
    mask_finite = np.isfinite(r_scaled) & np.isfinite(prs.to_numpy(dtype=float))
    if mask_finite.sum() > 10:
        corr = np.corrcoef(r_scaled[mask_finite], prs.to_numpy(dtype=float)[mask_finite])[0, 1]
        logging.info(
            "[RiskSurfaceSanity] Corr(risk_per_trade_frac/max_risk_pct, position_risk_score)=%.3f",
            float(corr),
        )
        if corr < 0.9:
            logging.warning(
                "[WARN] position_risk_score not tightly aligned with risk_per_trade_frac (corr=%.3f).",
                float(corr),
            )
            warns += 1
        else:
            logging.info(
                "[PASS] position_risk_score tightly aligned with risk_per_trade_frac (corr=%.3f).",
                float(corr),
            )

    # --------------------------------------------------------------
    # Final summary
    # --------------------------------------------------------------
    logging.info("[RiskSurfaceSanity] ===== SUMMARY =====")
    logging.info(
        "[RiskSurfaceSanity] total_rows=%d | active=%d | fails=%d | warns=%d",
        n_rows,
        n_active,
        fails,
        warns,
    )
    if fails == 0:
        if warns == 0:
            logging.info(
                "[RiskSurfaceSanity] ✅ PASS: Pillar 3 sizing looks consistent (no warnings)."
            )
        else:
            logging.info(
                "[RiskSurfaceSanity] ✅ PASS (with warnings): Check logs for details."
            )
    else:
        logging.error(
            "[RiskSurfaceSanity] ❌ FAIL: Pillar 3 sizing has %d hard failures. Inspect logs.",
            fails,
        )


if __name__ == "__main__":
    main()