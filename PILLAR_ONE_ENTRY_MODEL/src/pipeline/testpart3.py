"""
Diagnostic / validation script for Pillar 1 mini-engines (deterministic signal layers).

Target parquet:
    PILLAR_ONE_ENTRY_MODEL/data/processed/features_pillar1_with_engines.parquet

Checks (non-pytest, pure terminal output):
    - File exists and loads, shape/memory summary
    - Time index sanity (open_time_ms sorted, no NaNs/dupes)
    - Presence of all mini-engine score columns
    - No NaNs / infs in mini-engine scores
    - Score ranges: scores should be in [-1, 1] (tanh-bounded)
    - Scores are not constant (std > 0)
    - Saturation diagnostics: % of values near -1 or +1
    - Correlations of each score with key primitives
    - Regime-aware sanity checks (means per regime where available)

Outputs:
    - PASS/FAIL/WARN per check
    - Final summary with counts and overall verdict
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import logging

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = (
    BASE_DIR
    / "PILLAR_ONE_ENTRY_MODEL"
    / "data"
    / "processed"
    / "features_pillar1_with_engines.parquet"
)


# ---------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------
@dataclass
class CheckResult:
    name: str
    status: str  # "PASS" | "FAIL" | "WARN"
    details: List[str]


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"{msg} | shape={df.shape} | approx_mem={mb:.1f} MB")


# ---------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------
def check_time_index(df: pd.DataFrame) -> CheckResult:
    name = "Time index (open_time_ms) sanity"
    details: List[str] = []

    if "open_time_ms" not in df.columns:
        return CheckResult(
            name=name,
            status="FAIL",
            details=["Column 'open_time_ms' missing – required for 1m bar alignment."],
        )

    ts = df["open_time_ms"]
    has_nans = ts.isna().any()
    is_sorted = ts.is_monotonic_increasing
    has_dupes = ts.duplicated().any()

    details.append("  OK" if (not has_nans and is_sorted and not has_dupes) else "  Issues detected:")
    details.append(f"  - is_sorted: {is_sorted}")
    details.append(f"  - has_nans: {has_nans}")
    details.append(f"  - has_duplicates: {has_dupes}")

    status = "PASS" if (not has_nans and is_sorted and not has_dupes) else "FAIL"
    return CheckResult(name=name, status=status, details=details)


def check_score_columns_present(df: pd.DataFrame) -> CheckResult:
    name = "Mini-engine score columns present"
    details: List[str] = []

    expected = [
        "momentum_score",
        "trend_score",
        "volatility_score",
        "reversion_score",
        "orderflow_score",
        "structure_score",
        "pattern_score",
    ]
    missing = [c for c in expected if c not in df.columns]

    if missing:
        details.append("Missing mini-engine score columns:")
        details.append(f"  - {missing}")
        return CheckResult(name=name, status="FAIL", details=details)

    details.append("All expected mini-engine scores present.")
    details.append(f"  - columns: {expected}")
    return CheckResult(name=name, status="PASS", details=details)


def check_scores_finite(df: pd.DataFrame) -> CheckResult:
    name = "No NaNs / infs in mini-engine scores"
    details: List[str] = []
    score_cols = [
        c
        for c in [
            "momentum_score",
            "trend_score",
            "volatility_score",
            "reversion_score",
            "orderflow_score",
            "structure_score",
            "pattern_score",
        ]
        if c in df.columns
    ]
    if not score_cols:
        return CheckResult(
            name=name,
            status="FAIL",
            details=["No mini-engine score columns found at all."],
        )

    sub = df[score_cols]
    n_nans = int(sub.isna().sum().sum())
    n_infs = int(np.isinf(sub.to_numpy()).sum())

    if n_nans == 0 and n_infs == 0:
        details.append("  All score columns finite.")
        details.append(f"  - score_cols={len(score_cols)}, NaNs={n_nans}, infs={n_infs}")
        status = "PASS"
    else:
        details.append("  Non-finite values detected in score columns.")
        details.append(f"  - score_cols={len(score_cols)}, NaNs={n_nans}, infs={n_infs}")
        status = "FAIL"

    return CheckResult(name=name, status=status, details=details)


def check_score_ranges(df: pd.DataFrame, tol: float = 1e-3) -> CheckResult:
    """
    Scores are tanh-bounded → should live in [-1, 1], maybe with tiny numerical
    overshoot. We allow a small tolerance around [-1, 1].
    """
    name = "Mini-engine score ranges ~[-1, 1]"
    details: List[str] = []

    cols = [
        c
        for c in [
            "momentum_score",
            "trend_score",
            "volatility_score",
            "reversion_score",
            "orderflow_score",
            "structure_score",
            "pattern_score",
        ]
        if c in df.columns
    ]
    if not cols:
        return CheckResult(
            name=name,
            status="FAIL",
            details=["No mini-engine score columns to test ranges."],
        )

    violators = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        min_v = float(s.min())
        max_v = float(s.max())
        if min_v < -1.0 - tol or max_v > 1.0 + tol:
            violators[c] = (min_v, max_v)
        details.append(f"  - {c}: min={min_v:.4f}, max={max_v:.4f}")

    if violators:
        details.append("Scores with values outside [-1 - tol, 1 + tol]:")
        for c, (mn, mx) in violators.items():
            details.append(f"  * {c}: min={mn:.4f}, max={mx:.4f}")
        status = "FAIL"
    else:
        status = "PASS"

    return CheckResult(name=name, status=status, details=details)


def check_score_non_constant(df: pd.DataFrame) -> CheckResult:
    name = "Mini-engine scores are not constant"
    details: List[str] = []
    cols = [
        c
        for c in [
            "momentum_score",
            "trend_score",
            "volatility_score",
            "reversion_score",
            "orderflow_score",
            "structure_score",
            "pattern_score",
        ]
        if c in df.columns
    ]
    if not cols:
        return CheckResult(
            name=name,
            status="FAIL",
            details=["No mini-engine score columns found."],
        )

    std = df[cols].std()
    const = std[std <= 1e-9]
    if const.empty:
        details.append("No constant score columns detected.")
        for c in cols:
            details.append(f"  - {c}: std={std[c]:.6f}")
        status = "PASS"
    else:
        details.append("Constant (or near-constant) score columns detected:")
        for c in const.index.tolist():
            details.append(f"  - {c}: std={std[c]:.6e}")
        status = "FAIL"

    return CheckResult(name=name, status=status, details=details)


def check_score_saturation(df: pd.DataFrame, thresh: float = 0.99) -> CheckResult:
    """
    Diagnostics for how many scores are near -1 or +1. This is a WARN-only
    check (never hard FAIL) – too much saturation might indicate that
    weights or scale are too aggressive.
    """
    name = "Score saturation diagnostics (near tanh bounds)"
    details: List[str] = []
    cols = [
        c
        for c in [
            "momentum_score",
            "trend_score",
            "volatility_score",
            "reversion_score",
            "orderflow_score",
            "structure_score",
            "pattern_score",
        ]
        if c in df.columns
    ]
    if not cols:
        return CheckResult(
            name=name,
            status="WARN",
            details=["No mini-engine score columns found for saturation check."],
        )

    warn = False
    n = len(df)
    details.append(f"Total rows: {n}")
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        hi = (s >= thresh).sum()
        lo = (s <= -thresh).sum()
        frac_hi = hi / n if n > 0 else 0.0
        frac_lo = lo / n if n > 0 else 0.0
        details.append(
            f"  - {c}: >= {thresh:.2f} → {frac_hi:.3%}, <= -{thresh:.2f} → {frac_lo:.3%}"
        )
        if frac_hi > 0.10 or frac_lo > 0.10:
            warn = True

    if warn:
        details.append(
            "One or more scores have >10% of values very close to ±1; "
            "consider reviewing weights/scale if this is unintended."
        )
        status = "WARN"
    else:
        status = "PASS"

    return CheckResult(name=name, status=status, details=details)


def check_score_correlations(df: pd.DataFrame) -> CheckResult:
    """
    Check that each mini-engine score has a reasonable correlation
    with its primary primitive(s). If *all* correlations for an engine
    are near zero, it might indicate miswiring.

    This is a FAIL if |corr| < min_abs_corr for all primitives for a given score.
    """
    name = "Score ↔ primitive correlation sanity"
    details: List[str] = []
    min_abs_corr = 0.05  # extremely low corr everywhere → suspicious

    corr_pairs = {
        "momentum_score": ["mom_roc_5", "mom_ret_z_20"],
        "trend_score": ["trend_slope_20", "trend_efficiency_20"],
        "volatility_score": ["vol_atr_14", "vol_percentile_1000"],
        "reversion_score": ["rev_zscore_50", "rev_rsi_14"],
        "orderflow_score": ["of_delta_ema_20", "of_cvd_slope_50"],
        "structure_score": ["struct_range_width_50"],
        "pattern_score": ["pat_pattern_cluster_10"],
    }

    any_fail = False

    for score, prims in corr_pairs.items():
        if score not in df.columns:
            details.append(f"- {score}: missing score column, skipping.")
            continue

        cols = [c for c in prims if c in df.columns]
        if not cols:
            details.append(f"- {score}: missing all primitives {prims}, skipping.")
            continue

        sub = df[[score] + cols].dropna()
        if len(sub) == 0:
            details.append(f"- {score}: no data after dropping NaNs, skipping.")
            continue

        corr = sub.corr().iloc[0, 1:]
        details.append(f"- {score} correlations:")
        for c in cols:
            details.append(f"    {c}: corr={corr[c]:.3f}")

        # Fail if all abs(corr) below threshold
        if all(abs(corr[c]) < min_abs_corr for c in cols):
            any_fail = True
            details.append(
                f"  -> All |corr| < {min_abs_corr:.3f} for {score}; "
                "engine may be miswired or primitives not informative."
            )

    status = "FAIL" if any_fail else "PASS"
    return CheckResult(name=name, status=status, details=details)


def check_regime_interactions(df: pd.DataFrame) -> CheckResult:
    """
    Soft check: do regimes roughly influence the right scores?

    - trend_score should differ between regime_is_trend == 1 vs 0
    - reversion_score should differ between regime_is_mean_reversion == 1 vs 0
    - orderflow_score should differ between regime_is_vol_expansion == 1 vs 0

    This is WARN-only (not hard FAIL).
    """
    name = "Regime interaction sanity (trend/reversion/vol_expansion)"
    details: List[str] = []
    warn = False

    def _regime_delta(
        score_col: str,
        flag_col: str,
        pretty: str,
    ):
        nonlocal warn
        if score_col not in df.columns or flag_col not in df.columns:
            details.append(f"- {pretty}: missing {score_col} or {flag_col}, skipping.")
            return

        mask_flag = df[flag_col] == 1
        mask_not = df[flag_col] == 0

        if mask_flag.sum() < 100 or mask_not.sum() < 100:
            details.append(
                f"- {pretty}: not enough samples in one of the buckets "
                f"(flag={mask_flag.sum()}, not={mask_not.sum()}), skipping."
            )
            return

        m_flag = df.loc[mask_flag, score_col].mean()
        m_not = df.loc[mask_not, score_col].mean()
        delta = m_flag - m_not
        details.append(
            f"- {pretty}: mean(flag=1)={m_flag:.4f}, mean(flag=0)={m_not:.4f}, Δ={delta:.4f}"
        )
        if abs(delta) < 0.02:
            warn = True
            details.append(
                f"  -> |Δ| < 0.02; regime flag doesn't visibly tilt {score_col}."
            )

    _regime_delta("trend_score", "regime_is_trend", "trend regime vs trend_score")
    _regime_delta(
        "reversion_score",
        "regime_is_mean_reversion",
        "mean-reversion regime vs reversion_score",
    )
    _regime_delta(
        "orderflow_score",
        "regime_is_vol_expansion",
        "vol-expansion regime vs orderflow_score",
    )

    if warn:
        status = "WARN"
    else:
        status = "PASS"

    return CheckResult(name=name, status=status, details=details)


def summarize_scores(df: pd.DataFrame) -> CheckResult:
    name = "Mini-engine score distribution snapshot"
    details: List[str] = []

    cols = [
        c
        for c in [
            "momentum_score",
            "trend_score",
            "volatility_score",
            "reversion_score",
            "orderflow_score",
            "structure_score",
            "pattern_score",
        ]
        if c in df.columns
    ]
    if not cols:
        return CheckResult(
            name=name,
            status="WARN",
            details=["No mini-engine score columns present to summarize."],
        )

    desc = df[cols].describe().round(4)
    details.append("Descriptive stats for mini-engine scores:")
    details.append(desc.to_string())
    return CheckResult(name=name, status="PASS", details=details)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    _print_header("Loading Pillar 1 mini-engine features parquet")
    print(f"Path: {DATA_PATH}")
    if not DATA_PATH.exists():
        print("ERROR: Parquet file not found.")
        print("Make sure you have run the mini-engine pipeline first.")
        return

    df = pd.read_parquet(DATA_PATH)
    _log_mem(df, "[MiniEnginesTest] Loaded features+scores")

    _print_header("Running Pillar 1 mini-engine validation checks")
    results: List[CheckResult] = []

    # Core checks
    results.append(check_time_index(df))
    results.append(check_score_columns_present(df))
    results.append(check_scores_finite(df))
    results.append(check_score_ranges(df))
    results.append(check_score_non_constant(df))
    results.append(check_score_saturation(df))
    results.append(check_score_correlations(df))
    results.append(check_regime_interactions(df))
    results.append(summarize_scores(df))

    # Print per-check output
    for res in results:
        tag = f"[{res.status}]"
        print(f"{tag} {res.name}")
        for line in res.details:
            print(line)

    # Summary
    _print_header("Summary")
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_warn = sum(1 for r in results if r.status == "WARN")
    total = len(results)

    print(f"Total checks: {total}")
    print(f"Passed      : {n_pass}")
    print(f"Failed      : {n_fail}")
    print(f"Warnings    : {n_warn}")
    print()

    if n_fail == 0:
        if n_warn == 0:
            print("Overall verdict: ✅ All mini-engine checks passed with no warnings.")
        else:
            print("Overall verdict: ✅ No hard failures, but there are WARNINGS above to review.")
    else:
        print("Overall verdict: ❌ One or more checks FAILED.")
        print(
            "Advanced reasoning: look at FAILED checks above. Typical causes:\n"
            "- score columns missing → pipeline not wired to write mini-engines\n"
            "- NaNs / infs → missing fills or numeric overflow\n"
            "- scores out of [-1,1] → tanh or scale misconfigured\n"
            "- constant scores → engine inputs are dead, or weights=0\n"
            "- near-zero correlations → engine not actually using its primitives\n"
            "- weak regime deltas → regime flags not influencing scores as intended"
        )


if __name__ == "__main__":
    main()