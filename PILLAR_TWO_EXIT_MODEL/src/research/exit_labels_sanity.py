# src/research/exit_labels_sanity.py

"""
Sanity checks for Pillar 2 exit labels.

Run from PILLAR_TWO_EXIT_MODEL:

    (venv) python -m src.research.exit_labels_sanity

Checks:
  - exit_labels.parquet exists and is non-empty
  - required columns are present
  - alignment with exit_raw (same number of rows, 1–1 on open_time_ms)
  - basic type / NaN sanity for all label columns
  - value ranges (mfe_dir, mae_dir, optimal_tp/SL, RR)
  - exit_outcome only in {tp_first, sl_first, timeout}
  - expected_exit_time within [1, horizon_bars] (ignoring NaNs)
"""

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.exit.exit_labels import ExitLabelConfig

BASE_DIR = Path(__file__).resolve().parents[3]
P2_BASE = BASE_DIR / "PILLAR_TWO_EXIT_MODEL"

EXIT_LABELS_PATH = P2_BASE / "data" / "processed" / "exit_labels.parquet"
EXIT_RAW_PATH = P2_BASE / "data" / "processed" / "exit_raw.parquet"


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def _require_cols(df: pd.DataFrame, cols: List[str], context: str, errors: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        msg = f"[ExitLabelsSanity] Missing required columns in {context}: {missing}"
        logging.error(msg)
        errors.append(msg)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    errors: List[str] = []
    cfg = ExitLabelConfig()

    # ------------------------------------------------------------------
    # 1) Load exit_labels
    # ------------------------------------------------------------------
    if not EXIT_LABELS_PATH.exists():
        logging.error("[ExitLabelsSanity] exit_labels parquet not found: %s", EXIT_LABELS_PATH)
        raise SystemExit(1)

    logging.info("[ExitLabelsSanity] Loading exit_labels: %s", EXIT_LABELS_PATH)
    labels = pd.read_parquet(EXIT_LABELS_PATH)
    _log_mem(labels, "[ExitLabelsSanity] Loaded exit_labels")

    if labels.empty:
        logging.error("[ExitLabelsSanity] exit_labels is empty!")
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # 2) Required columns
    # ------------------------------------------------------------------
    required_cols = [
        cfg.col_time,
        "mfe_dir",
        "mae_dir",
        "expected_exit_time",
        "exit_outcome",
        "optimal_tp_distance",
        "optimal_sl_distance",
        "optimal_risk_reward",
    ]
    _require_cols(labels, required_cols, "exit_labels", errors)

    # If key columns are missing, no point going further
    if errors:
        logging.error("[ExitLabelsSanity] FAIL: critical columns missing.")
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # 3) Alignment with exit_raw.parquet
    # ------------------------------------------------------------------
    if not EXIT_RAW_PATH.exists():
        msg = f"[ExitLabelsSanity] exit_raw parquet not found (expected for alignment): {EXIT_RAW_PATH}"
        logging.error(msg)
        errors.append(msg)
    else:
        logging.info("[ExitLabelsSanity] Loading exit_raw: %s", EXIT_RAW_PATH)
        exit_raw = pd.read_parquet(EXIT_RAW_PATH)
        _log_mem(exit_raw, "[ExitLabelsSanity] Loaded exit_raw")

        if exit_raw.empty:
            msg = "[ExitLabelsSanity] exit_raw is empty!"
            logging.error(msg)
            errors.append(msg)
        else:
            if len(exit_raw) != len(labels):
                msg = (
                    f"[ExitLabelsSanity] Row mismatch: exit_raw={len(exit_raw)}, "
                    f"exit_labels={len(labels)}"
                )
                logging.error(msg)
                errors.append(msg)

            # Check 1–1 on open_time_ms (ignoring order)
            if cfg.col_time in exit_raw.columns:
                t_raw = set(exit_raw[cfg.col_time].astype(int).tolist())
                t_lab = set(labels[cfg.col_time].astype(int).tolist())
                if t_raw != t_lab:
                    missing_in_lab = sorted(t_raw - t_lab)[:5]
                    missing_in_raw = sorted(t_lab - t_raw)[:5]
                    msg = (
                        "[ExitLabelsSanity] open_time_ms mismatch between exit_raw and labels. "
                        f"examples missing_in_labels={missing_in_lab}, "
                        f"missing_in_exit_raw={missing_in_raw}"
                    )
                    logging.error(msg)
                    errors.append(msg)
            else:
                msg = f"[ExitLabelsSanity] {cfg.col_time} missing from exit_raw; cannot verify alignment."
                logging.error(msg)
                errors.append(msg)

    # ------------------------------------------------------------------
    # 4) Basic NaN / type sanity
    # ------------------------------------------------------------------
    numeric_cols = [
        "mfe_dir",
        "mae_dir",
        "expected_exit_time",
        "optimal_tp_distance",
        "optimal_sl_distance",
        "optimal_risk_reward",
    ]

    for col in numeric_cols:
        series = labels[col]
        if not np.issubdtype(series.dtype, np.number):
            msg = f"[ExitLabelsSanity] Column {col} is not numeric (dtype={series.dtype})"
            logging.error(msg)
            errors.append(msg)
            continue

        n_total = len(series)
        n_nan = int(series.isna().sum())
        frac_nan = n_nan / n_total
        logging.info(
            "[ExitLabelsSanity] NaN stats for %s: %d/%d (%.4f)",
            col,
            n_nan,
            n_total,
            frac_nan,
        )

        # Allow some NaNs (e.g. tail, bad rows) but warn if too many
        if frac_nan > 0.05:
            msg = (
                f"[ExitLabelsSanity] Column {col} has >5% NaNs "
                f"({n_nan}/{n_total}, {frac_nan:.3f})"
            )
            logging.warning(msg)
            # Not a hard fail, just warning.

    # ------------------------------------------------------------------
    # 5) Value range checks
    # ------------------------------------------------------------------
    # expected_exit_time within [1, horizon] ignoring NaNs
    if "expected_exit_time" in labels.columns:
        e = labels["expected_exit_time"].astype(float)
        mask = e.notna()
        if mask.any():
            e_valid = e[mask]
            min_e = float(e_valid.min())
            max_e = float(e_valid.max())
            logging.info(
                "[ExitLabelsSanity] expected_exit_time range (non-NaN): [%.1f, %.1f]",
                min_e,
                max_e,
            )

            if min_e < 1 - 1e-6:
                msg = (
                    f"[ExitLabelsSanity] expected_exit_time has values < 1 (min={min_e:.3f}). "
                    "This usually indicates an indexing bug."
                )
                logging.error(msg)
                errors.append(msg)

            # give a bit of slack above horizon in case config changed between build/check
            if max_e > cfg.horizon_bars * 1.1:
                msg = (
                    f"[ExitLabelsSanity] expected_exit_time max={max_e:.3f} "
                    f"exceeds 1.1 * horizon_bars ({cfg.horizon_bars})."
                )
                logging.error(msg)
                errors.append(msg)

    # mfe_dir should be mostly >= 0, mae_dir mostly <= 0 in directional terms
    if "mfe_dir" in labels.columns:
        mfe = labels["mfe_dir"].astype(float)
        neg_mfe_frac = float((mfe < -1e-8).sum()) / len(mfe)
        logging.info(
            "[ExitLabelsSanity] mfe_dir: mean=%.4f, std=%.4f, neg_fraction=%.4f",
            float(np.nanmean(mfe)),
            float(np.nanstd(mfe)),
            neg_mfe_frac,
        )
        if neg_mfe_frac > 0.01:
            msg = (
                f"[ExitLabelsSanity] mfe_dir has >1%% negative values "
                f"(neg_fraction={neg_mfe_frac:.3f}); directional logic may be flipped."
            )
            logging.warning(msg)

    if "mae_dir" in labels.columns:
        mae = labels["mae_dir"].astype(float)
        pos_mae_frac = float((mae > 1e-8).sum()) / len(mae)
        logging.info(
            "[ExitLabelsSanity] mae_dir: mean=%.4f, std=%.4f, pos_fraction=%.4f",
            float(np.nanmean(mae)),
            float(np.nanstd(mae)),
            pos_mae_frac,
        )
        if pos_mae_frac > 0.01:
            msg = (
                f"[ExitLabelsSanity] mae_dir has >1%% positive values "
                f"(pos_fraction={pos_mae_frac:.3f}); directional logic may be flipped."
            )
            logging.warning(msg)

    # TP/SL and RR sanity
    if "optimal_tp_distance" in labels.columns:
        tp = labels["optimal_tp_distance"].astype(float)
        frac_neg_tp = float((tp < -1e-8).sum()) / len(tp)
        logging.info(
            "[ExitLabelsSanity] optimal_tp_distance: mean=%.4f, std=%.4f, neg_fraction=%.4f",
            float(np.nanmean(tp)),
            float(np.nanstd(tp)),
            frac_neg_tp,
        )
        if frac_neg_tp > 0.0:
            msg = "[ExitLabelsSanity] optimal_tp_distance has negative values."
            logging.error(msg)
            errors.append(msg)

    if "optimal_sl_distance" in labels.columns:
        sl = labels["optimal_sl_distance"].astype(float)
        frac_nonpos_sl = float((sl <= 0).sum()) / len(sl)
        logging.info(
            "[ExitLabelsSanity] optimal_sl_distance: mean=%.4f, std=%.4f, nonpos_fraction=%.4f",
            float(np.nanmean(sl)),
            float(np.nanstd(sl)),
            frac_nonpos_sl,
        )
        if frac_nonpos_sl > 0.01:
            msg = (
                "[ExitLabelsSanity] optimal_sl_distance has >1% non-positive values; "
                "this can cause division issues for RR."
            )
            logging.error(msg)
            errors.append(msg)

    if "optimal_risk_reward" in labels.columns:
        rr = labels["optimal_risk_reward"].astype(float)
        frac_neg_rr = float((rr < -1e-8).sum()) / len(rr)
        huge_rr_frac = float((rr > 100).sum()) / len(rr)
        logging.info(
            "[ExitLabelsSanity] optimal_risk_reward: mean=%.4f, std=%.4f, "
            "neg_fraction=%.4f, rr>100_fraction=%.4f",
            float(np.nanmean(rr)),
            float(np.nanstd(rr)),
            frac_neg_rr,
            huge_rr_frac,
        )
        if frac_neg_rr > 0.0:
            msg = "[ExitLabelsSanity] optimal_risk_reward has negative values."
            logging.error(msg)
            errors.append(msg)
        if huge_rr_frac > 0.01:
            logging.warning(
                "[ExitLabelsSanity] >1%% of RR values exceed 100; check for outliers."
            )

    # ------------------------------------------------------------------
    # 6) exit_outcome domain check
    # ------------------------------------------------------------------
    if "exit_outcome" in labels.columns:
        valid_outcomes = {"tp_first", "sl_first", "timeout"}
        vals = labels["exit_outcome"].dropna().astype(str)
        unique_vals = set(vals.unique().tolist())
        logging.info(
            "[ExitLabelsSanity] exit_outcome unique values: %s",
            unique_vals,
        )
        bad = unique_vals - valid_outcomes
        if bad:
            msg = f"[ExitLabelsSanity] exit_outcome has unexpected values: {bad}"
            logging.error(msg)
            errors.append(msg)

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------
    if errors:
        logging.error("[ExitLabelsSanity] FAIL: exit_labels parquet has issues:")
        for e in errors:
            logging.error("  - %s", e)
        raise SystemExit(1)
    else:
        logging.info("[ExitLabelsSanity] PASS: exit_labels parquet looks consistent.")


if __name__ == "__main__":
    main()