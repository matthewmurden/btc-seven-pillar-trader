# src/research/exit_deterministic_sanity.py

"""
Sanity checks for Pillar 2 deterministic exit profile.

Loads:
  - data/processed/exit_with_deterministic.parquet

Checks:
  - Core columns present
  - Deterministic TP/SL > 0 and finite
  - TP-first probability in [0,1]
  - Exit time estimate within configured range
  - Basic distribution stats

Run:
  python -m src.research.exit_deterministic_sanity
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.exit.exit_deterministic import ExitDeterministicConfig

BASE_DIR = Path(__file__).resolve().parents[3]
P2_BASE = BASE_DIR / "PILLAR_TWO_EXIT_MODEL"

INPUT_PATH = P2_BASE / "data" / "processed" / "exit_with_deterministic.parquet"


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    cfg = ExitDeterministicConfig()

    logging.info(
        "[ExitDeterministicSanity] Loading deterministic exit parquet: %s",
        INPUT_PATH,
    )
    df = pd.read_parquet(INPUT_PATH)
    _log_mem(df, "[ExitDeterministicSanity] Loaded")

    n_rows, n_cols = df.shape
    logging.info("[ExitDeterministicSanity] n_rows=%d, n_cols=%d", n_rows, n_cols)

    fails = 0
    warns = 0

    if n_rows == 0:
        logging.error("[FAIL] File has zero rows.")
        fails += 1
    else:
        logging.info("[PASS] File has at least one row.")

    # Core columns
    core_cols = [
        cfg.col_time,
        cfg.col_close,
        cfg.col_atr,
        cfg.col_direction,
        cfg.col_decision,
        "deterministic_tp",
        "deterministic_sl",
        "deterministic_tp_first_probability",
        "deterministic_exit_time_estimate",
    ]
    missing_core = [c for c in core_cols if c not in df.columns]
    if missing_core:
        logging.error("[FAIL] Missing core deterministic columns: %s", missing_core)
        fails += 1
    else:
        logging.info("[PASS] All core deterministic columns present.")

    # Helper for numeric stats
    def col_stats(col: str) -> None:
        nonlocal fails, warns
        s = df[col].astype(float)
        nan_frac = float(s.isna().mean())
        n_inf = np.isinf(s).sum()
        if nan_frac > 0.5:
            logging.error(
                "[FAIL] %s: >50%% NaN (nan_frac=%.4f)",
                col,
                nan_frac,
            )
            fails += 1
        else:
            logging.info(
                "[ExitDeterministicSanity] %s: n=%d, nan_frac=%.4f, n_inf=%d, "
                "min=%.6f, max=%.6f, mean=%.6f, std=%.6f",
                col,
                len(s),
                nan_frac,
                n_inf,
                float(s.min(skipna=True)),
                float(s.max(skipna=True)),
                float(s.mean(skipna=True)),
                float(s.std(skipna=True)),
            )
            if n_inf > 0:
                logging.error("[FAIL] %s: contains +/-Inf values.", col)
                fails += 1

    # Deterministic TP/SL stats
    for col in [
        "deterministic_tp",
        "deterministic_sl",
    ]:
        if col in df.columns:
            col_stats(col)

    # TP/SL positivity checks
    for col in ["deterministic_tp", "deterministic_sl"]:
        if col in df.columns:
            s = df[col].astype(float)
            nonpos_frac = float((s <= 0.0).mean())
            logging.info(
                "[ExitDeterministicSanity] %s nonpos_fraction=%.4f",
                col,
                nonpos_frac,
            )
            if nonpos_frac > 0.05:
                logging.warning(
                    "[WARN] %s has >5%% non-positive values (nonpos_frac=%.4f).",
                    col,
                    nonpos_frac,
                )
                warns += 1
            else:
                logging.info(
                    "[PASS] %s mostly positive (nonpos_frac=%.4f).",
                    col,
                    nonpos_frac,
                )

    # TP-first probability checks
    if "deterministic_tp_first_probability" in df.columns:
        p = df["deterministic_tp_first_probability"].astype(float)
        col_stats("deterministic_tp_first_probability")
        if (p < -0.01).any() or (p > 1.01).any():
            logging.error(
                "[FAIL] deterministic_tp_first_probability out of [0,1] "
                "(min=%.4f, max=%.4f).",
                float(p.min(skipna=True)),
                float(p.max(skipna=True)),
            )
            fails += 1
        else:
            logging.info(
                "[PASS] deterministic_tp_first_probability ~in [0,1] "
                "(min=%.4f, max=%.4f).",
                float(p.min(skipna=True)),
                float(p.max(skipna=True)),
            )

    # Exit time estimate checks
    if "deterministic_exit_time_estimate" in df.columns:
        t = df["deterministic_exit_time_estimate"].astype(float)
        col_stats("deterministic_exit_time_estimate")

        t_min = float(t.min(skipna=True))
        t_max = float(t.max(skipna=True))
        if t_min < cfg.min_exit_bars - 1 or t_max > cfg.max_exit_bars + 1:
            logging.warning(
                "[WARN] deterministic_exit_time_estimate outside expected "
                "range [%d, %d] (min=%.2f, max=%.2f).",
                cfg.min_exit_bars,
                cfg.max_exit_bars,
                t_min,
                t_max,
            )
            warns += 1
        else:
            logging.info(
                "[PASS] deterministic_exit_time_estimate roughly in "
                "[%d, %d] (min=%.2f, max=%.2f).",
                cfg.min_exit_bars,
                cfg.max_exit_bars,
                t_min,
                t_max,
            )

    logging.info(
        "[ExitDeterministicSanity] ===== SUMMARY =====\n"
        "  total_rows=%d\n  total_cols=%d\n  fails=%d\n  warnings=%d",
        n_rows,
        n_cols,
        fails,
        warns,
    )

    if fails > 0:
        logging.error(
            "[ExitDeterministicSanity] ❌ FAIL: deterministic exit profile has %d hard failures.",
            fails,
        )
    else:
        logging.info(
            "[ExitDeterministicSanity] ✅ PASS: deterministic exit profile looks consistent "
            "(warnings=%d).",
            warns,
        )


if __name__ == "__main__":
    main()