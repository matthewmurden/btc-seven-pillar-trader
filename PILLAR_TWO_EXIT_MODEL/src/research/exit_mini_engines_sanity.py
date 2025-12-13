# src/research/exit_mini_engines_sanity.py

"""
Sanity checker for Pillar 2 — Mini-Engines parquet.

Loads:
  - PILLAR_TWO_EXIT_MODEL/data/processed/exit_with_mini_engines.parquet

Runs a battery of checks on:
  - core trade metadata
  - mini-engine feature presence
  - NaN / Inf rates
  - value ranges & basic consistency per engine

Outputs:
  - Lots of INFO / WARNING / ERROR lines
  - Final PASS / FAIL summary

Run:
  python -m src.research.exit_mini_engines_sanity
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[3]
P2_BASE = BASE_DIR / "PILLAR_TWO_EXIT_MODEL"

MINI_PATH = P2_BASE / "data" / "processed" / "exit_with_mini_engines.parquet"


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not MINI_PATH.exists():
        raise FileNotFoundError(f"Mini-engines parquet not found: {MINI_PATH}")

    logging.info("[ExitMiniSanity] Loading exit_with_mini_engines: %s", MINI_PATH)
    df = pd.read_parquet(MINI_PATH)
    _log_mem(df, "[ExitMiniSanity] Loaded mini-engines parquet")

    n_rows, n_cols = df.shape

    checks = 0
    fails = 0
    warns = 0

    def check(condition: bool, ok_msg: str, fail_msg: str, hard: bool = True) -> None:
        nonlocal checks, fails, warns
        checks += 1
        if condition:
            logging.info("[PASS] %s", ok_msg)
        else:
            if hard:
                fails += 1
                logging.error("[FAIL] %s", fail_msg)
            else:
                warns += 1
                logging.warning("[WARN] %s", fail_msg)

    # ------------------------------------------------------------------
    # Basic shape / existence
    # ------------------------------------------------------------------
    logging.info("[ExitMiniSanity] ===== BASIC INFO =====")
    logging.info("[ExitMiniSanity] n_rows=%d, n_cols=%d", n_rows, n_cols)

    check(
        n_rows > 0,
        "Mini-engines file has at least one row.",
        "Mini-engines file is empty.",
    )

    # ------------------------------------------------------------------
    # Required core columns (from exit_raw)
    # ------------------------------------------------------------------
    logging.info("[ExitMiniSanity] ===== CORE COLUMN PRESENCE =====")
    required_core = [
        "open_time_ms",
        "entry_decision",
        "entry_direction",
        "entry_confidence",
        "entry_prob_up",
        "signal_is_valid",
    ]

    missing_core = [c for c in required_core if c not in df.columns]
    check(
        len(missing_core) == 0,
        f"All core columns present: {required_core}",
        f"Missing core columns: {missing_core}",
    )

    if "entry_decision" in df.columns:
        dist_dec = df["entry_decision"].value_counts(dropna=False).to_dict()
        logging.info("[ExitMiniSanity] entry_decision distribution: %s", dist_dec)

    if "entry_direction" in df.columns:
        dist_dir = df["entry_direction"].value_counts(dropna=False).to_dict()
        logging.info("[ExitMiniSanity] entry_direction distribution: %s", dist_dir)

    # ------------------------------------------------------------------
    # Expected mini-engine columns
    # ------------------------------------------------------------------
    logging.info("[ExitMiniSanity] ===== MINI-ENGINE COLUMN PRESENCE =====")

    expected_engine_cols = [
        # Excursion engine
        "exit_excursion_net",
        "exit_excursion_ratio",
        "exit_conf_weighted_excursion",
        # Path shape engine
        "exit_path_curvature",
        "exit_path_monotonicity",
        "exit_path_convexity_score",
        # Entropy engine
        "exit_prob_entropy",
        "exit_regime_complexity",
        # Vol dynamics engine
        "exit_vol_ratio_short_long",
        "exit_vol_short_trend",
        "exit_vol_scaled_rv_10",
        # Reversion distance engine
        "exit_reversion_z",
        "exit_reversion_abs_z",
        "exit_reversion_alignment",
    ]

    missing_engine = [c for c in expected_engine_cols if c not in df.columns]
    check(
        len(missing_engine) == 0,
        "All expected mini-engine columns are present.",
        f"Missing mini-engine columns: {missing_engine}",
        hard=False,  # soft – we can still inspect what *is* there
    )

    logging.info(
        "[ExitMiniSanity] Engine columns actually present: %s",
        [c for c in expected_engine_cols if c in df.columns],
    )

    # ------------------------------------------------------------------
    # Utility helpers for numeric column diagnostics
    # ------------------------------------------------------------------
    def col_stats(col: str):
        series = pd.to_numeric(df[col], errors="coerce")
        n = len(series)
        n_nan = int(series.isna().sum())
        n_inf = int(np.isinf(series).sum())
        nan_frac = n_nan / max(n, 1)
        finite = series.replace([np.inf, -np.inf], np.nan)
        v_min = float(finite.min(skipna=True)) if nan_frac < 1.0 else np.nan
        v_max = float(finite.max(skipna=True)) if nan_frac < 1.0 else np.nan
        v_mean = float(finite.mean(skipna=True)) if nan_frac < 1.0 else np.nan
        v_std = float(finite.std(skipna=True)) if nan_frac < 1.0 else np.nan
        return {
            "n": n,
            "n_nan": n_nan,
            "nan_frac": nan_frac,
            "n_inf": n_inf,
            "min": v_min,
            "max": v_max,
            "mean": v_mean,
            "std": v_std,
        }

    # ------------------------------------------------------------------
    # Stats + sanity bounds per engine
    # ------------------------------------------------------------------
    logging.info("[ExitMiniSanity] ===== PER-COLUMN STATS & CHECKS =====")

    numeric_engine_cols = [
        c
        for c in expected_engine_cols
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]

    for col in numeric_engine_cols:
        stats = col_stats(col)
        logging.info(
            "[ExitMiniSanity] %s: n=%d, nan_frac=%.4f, n_inf=%d, min=%.6f, max=%.6f, mean=%.6f, std=%.6f",
            col,
            stats["n"],
            stats["nan_frac"],
            stats["n_inf"],
            stats["min"],
            stats["max"],
            stats["mean"],
            stats["std"],
        )

        # General sanity: avoid insane NaN/Inf rates
        check(
            stats["nan_frac"] < 0.5,
            f"{col}: <50% NaNs (nan_frac={stats['nan_frac']:.4f})",
            f"{col}: >=50% NaNs (nan_frac={stats['nan_frac']:.4f})",
            hard=False,
        )

        check(
            stats["n_inf"] == 0,
            f"{col}: no +/-Inf values.",
            f"{col}: has {stats['n_inf']} +/-Inf values.",
            hard=True,
        )

    # ------------------------------------------------------------------
    # Engine-specific tighter checks
    # ------------------------------------------------------------------

    # 1) Excursion Engine
    if "exit_excursion_ratio" in df.columns:
        stats = col_stats("exit_excursion_ratio")
        # ratio should be >= 0 in most cases
        neg_frac = float(
            (pd.to_numeric(df["exit_excursion_ratio"], errors="coerce") < 0).mean()
        )
        logging.info(
            "[ExitMiniSanity] exit_excursion_ratio negative fraction: %.4f",
            neg_frac,
        )
        check(
            neg_frac < 0.05,
            f"exit_excursion_ratio negative fraction <5%% (neg_frac={neg_frac:.4f})",
            f"exit_excursion_ratio has high negative fraction (neg_frac={neg_frac:.4f})",
            hard=False,
        )

    # 2) Path Shape Engine
    if "exit_path_convexity_score" in df.columns:
        stats = col_stats("exit_path_convexity_score")
        # tanh → must be in [-1, 1]
        check(
            stats["min"] >= -1.001 and stats["max"] <= 1.001,
            (
                f"exit_path_convexity_score within [-1,1] bounds "
                f"(min={stats['min']:.4f}, max={stats['max']:.4f})"
            ),
            (
                "exit_path_convexity_score out of [-1,1] bounds: "
                f"min={stats['min']:.4f}, max={stats['max']:.4f}"
            ),
            hard=True,
        )

    # 3) Entropy Engine
    if "exit_prob_entropy" in df.columns:
        stats = col_stats("exit_prob_entropy")
        # Entropy for Bernoulli ∈ [0, ~0.693]
        check(
            stats["min"] >= -1e-6 and stats["max"] <= 1.0,
            (
                f"exit_prob_entropy in [0,1]ish range "
                f"(min={stats['min']:.4f}, max={stats['max']:.4f})"
            ),
            (
                "exit_prob_entropy out of [0,1] range: "
                f"min={stats['min']:.4f}, max={stats['max']:.4f}"
            ),
            hard=True,
        )

    if "exit_regime_complexity" in df.columns:
        stats = col_stats("exit_regime_complexity")
        # By construction (sum of |flags| / 3), should be in [0, 1]
        check(
            stats["min"] >= -1e-3 and stats["max"] <= 1.1,
            (
                f"exit_regime_complexity in [0,1]ish range "
                f"(min={stats['min']:.4f}, max={stats['max']:.4f})"
            ),
            (
                "exit_regime_complexity out of [0,1]ish range: "
                f"min={stats['min']:.4f}, max={stats['max']:.4f}"
            ),
            hard=False,
        )

    # 4) Volatility Dynamics Engine
    if "exit_vol_ratio_short_long" in df.columns:
        series = pd.to_numeric(df["exit_vol_ratio_short_long"], errors="coerce")
        stats = col_stats("exit_vol_ratio_short_long")
        non_nan = series.dropna()
        frac_nonpos = float((non_nan <= 0).mean()) if len(non_nan) > 0 else 0.0
        logging.info(
            "[ExitMiniSanity] exit_vol_ratio_short_long nonpos_fraction: %.4f",
            frac_nonpos,
        )
        check(
            frac_nonpos < 0.1,
            f"exit_vol_ratio_short_long mostly positive (nonpos_frac={frac_nonpos:.4f})",
            f"exit_vol_ratio_short_long has many <=0 values (nonpos_frac={frac_nonpos:.4f})",
            hard=False,
        )

    if "exit_vol_scaled_rv_10" in df.columns:
        stats = col_stats("exit_vol_scaled_rv_10")
        check(
            stats["min"] >= -1e-6,
            (
                f"exit_vol_scaled_rv_10 non-negative (min={stats['min']:.6f}) "
                "(allowing tiny numerical noise)"
            ),
            f"exit_vol_scaled_rv_10 has negative values (min={stats['min']:.6f})",
            hard=False,
        )

    # 5) Reversion Distance Engine
    if "exit_reversion_abs_z" in df.columns:
        stats = col_stats("exit_reversion_abs_z")
        check(
            stats["min"] >= -1e-6,
            (
                f"exit_reversion_abs_z non-negative (min={stats['min']:.6f}) "
                "(allowing tiny numerical noise)"
            ),
            f"exit_reversion_abs_z has negative values (min={stats['min']:.6f})",
            hard=True,
        )

    if "exit_reversion_alignment" in df.columns:
        stats = col_stats("exit_reversion_alignment")
        # Should be in {-1, 0, 1} basically; allow tiny numerical noise
        check(
            stats["min"] >= -1.01 and stats["max"] <= 1.01,
            (
                "exit_reversion_alignment roughly in [-1,1] "
                f"(min={stats['min']:.4f}, max={stats['max']:.4f})"
            ),
            (
                "exit_reversion_alignment outside [-1,1]: "
                f"min={stats['min']:.4f}, max={stats['max']:.4f}"
            ),
            hard=False,
        )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    logging.info("[ExitMiniSanity] ===== SUMMARY =====")
    logging.info(
        "[ExitMiniSanity] Total checks run: %d | fails: %d | warnings: %d",
        checks,
        fails,
        warns,
    )

    if fails == 0:
        logging.info("[ExitMiniSanity] ✅ PASS: exit_with_mini_engines parquet looks consistent.")
    else:
        logging.error(
            "[ExitMiniSanity] ❌ FAIL: exit_with_mini_engines has %d hard failures. Inspect logs.",
            fails,
        )


if __name__ == "__main__":
    main()