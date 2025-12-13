# src/research/exit_fusion_sanity.py

"""
Sanity checks for Pillar 2 Fusion Layer output.

Loads:
  - data/processed/exit_with_fusion.parquet

Checks:
  - presence of fused_* columns
  - basic NaN / Inf stats
  - ranges for TP/SL/prob/time
  - RR distribution
  - simple comparison vs deterministic TP/SL

Run:
  python -m src.research.exit_fusion_sanity
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
P2_BASE = BASE_DIR / "PILLAR_TWO_EXIT_MODEL"

FUSION_PATH = P2_BASE / "data" / "processed" / "exit_with_fusion.parquet"


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logging.info("[ExitFusionSanity] Loading fused exit parquet: %s", FUSION_PATH)
    df = pd.read_parquet(FUSION_PATH)
    _log_mem(df, "[ExitFusionSanity] Loaded")

    n_rows, n_cols = df.shape
    logging.info("[ExitFusionSanity] n_rows=%d, n_cols=%d", n_rows, n_cols)

    if n_rows == 0:
        logging.error("[FAIL] Fused file has zero rows.")
        return
    else:
        logging.info("[PASS] Fused file has at least one row.")

    required_cols = [
        "deterministic_tp",
        "deterministic_sl",
        "fused_tp",
        "fused_sl",
        "fused_tp_first_probability",
        "fused_exit_time",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logging.error("[FAIL] Missing required fused columns: %s", missing)
        return
    else:
        logging.info("[PASS] All core fused columns present.")

    # Helper to dump stats
    def col_stats(name: str) -> None:
        s = pd.to_numeric(df[name], errors="coerce")
        n = len(s)
        nan_frac = float(s.isna().mean())
        n_inf = int(np.isinf(s).sum())
        logging.info(
            "[ExitFusionSanity] %s: n=%d, nan_frac=%.4f, n_inf=%d, min=%.6f, max=%.6f, mean=%.6f, std=%.6f",
            name,
            n,
            nan_frac,
            n_inf,
            float(s.min(skipna=True)),
            float(s.max(skipna=True)),
            float(s.mean(skipna=True)),
            float(s.std(skipna=True)),
        )
        if nan_frac > 0.1:
            logging.warning("[WARN] %s has >10%% NaNs (nan_frac=%.4f)", name, nan_frac)
        if n_inf > 0:
            logging.error("[FAIL] %s contains Inf values.", name)

    logging.info("[ExitFusionSanity] ===== BASIC STATS =====")
    col_stats("deterministic_tp")
    col_stats("deterministic_sl")
    col_stats("fused_tp")
    col_stats("fused_sl")
    col_stats("fused_tp_first_probability")
    col_stats("fused_exit_time")

    # Probability checks
    prob = pd.to_numeric(df["fused_tp_first_probability"], errors="coerce")
    if (prob < -1e-3).any() or (prob > 1.0 + 1e-3).any():
        logging.error(
            "[FAIL] fused_tp_first_probability outside [0,1] (min=%.4f, max=%.4f)",
            float(prob.min(skipna=True)),
            float(prob.max(skipna=True)),
        )
    else:
        logging.info(
            "[PASS] fused_tp_first_probability in [0,1]ish range (min=%.4f, max=%.4f)",
            float(prob.min(skipna=True)),
            float(prob.max(skipna=True)),
        )

    # TP/SL positivity + RR distribution
    fused_tp = pd.to_numeric(df["fused_tp"], errors="coerce")
    fused_sl = pd.to_numeric(df["fused_sl"], errors="coerce")

    nonpos_tp = (fused_tp <= 0).mean()
    nonpos_sl = (fused_sl <= 0).mean()

    logging.info("[ExitFusionSanity] fused_tp nonpos_fraction=%.4f", nonpos_tp)
    logging.info("[ExitFusionSanity] fused_sl nonpos_fraction=%.4f", nonpos_sl)

    if nonpos_tp > 0.0:
        logging.error("[FAIL] Some fused_tp <= 0 (nonpos_frac=%.4f).", nonpos_tp)
    else:
        logging.info("[PASS] fused_tp strictly positive (within numerical tolerances).")

    if nonpos_sl > 0.0:
        logging.error("[FAIL] Some fused_sl <= 0 (nonpos_frac=%.4f).", nonpos_sl)
    else:
        logging.info("[PASS] fused_sl strictly positive (within numerical tolerances).")

    rr = fused_tp / fused_sl.replace(0.0, np.nan)
    rr = rr.replace([np.inf, -np.inf], np.nan)
    logging.info(
        "[ExitFusionSanity] fused RR (tp/sl): min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
        float(rr.min(skipna=True)),
        float(rr.max(skipna=True)),
        float(rr.mean(skipna=True)),
        float(rr.std(skipna=True)),
    )

    # Compare fused vs deterministic TP/SL to ensure we're not identical or insane
    det_tp = pd.to_numeric(df["deterministic_tp"], errors="coerce")
    det_sl = pd.to_numeric(df["deterministic_sl"], errors="coerce")

    tp_ratio = fused_tp / det_tp.replace(0.0, np.nan)
    sl_ratio = fused_sl / det_sl.replace(0.0, np.nan)

    tp_ratio = tp_ratio.replace([np.inf, -np.inf], np.nan)
    sl_ratio = sl_ratio.replace([np.inf, -np.inf], np.nan)

    logging.info(
        "[ExitFusionSanity] fused_tp / det_tp: min=%.4f, max=%.4f, mean=%.4f",
        float(tp_ratio.min(skipna=True)),
        float(tp_ratio.max(skipna=True)),
        float(tp_ratio.mean(skipna=True)),
    )
    logging.info(
        "[ExitFusionSanity] fused_sl / det_sl: min=%.4f, max=%.4f, mean=%.4f",
        float(sl_ratio.min(skipna=True)),
        float(sl_ratio.max(skipna=True)),
        float(sl_ratio.mean(skipna=True)),
    )

    # Light sanity: ratios shouldn't be all exactly 1, and shouldn't be routinely insane
    if np.isclose(tp_ratio.dropna(), 1.0).all() and np.isclose(sl_ratio.dropna(), 1.0).all():
        logging.warning(
            "[WARN] fused TP/SL are almost identical to deterministic TP/SL; "
            "fusion may not be doing much."
        )

    logging.info("[ExitFusionSanity] ===== SUMMARY =====")
    logging.info(
        "[ExitFusionSanity] DONE. Inspect log for WARN/FAIL. If only PASS/WARN → probably fine."
    )


if __name__ == "__main__":
    main()