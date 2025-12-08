"""
Pillar 1 — features_pillar1_primitives sanity + quality hub.

Usage:
    python -m src.pipeline.test_pillar1_primitives

What it does:
- Loads data/processed/features_pillar1_primitives.parquet
- Verifies:
    * File exists and loads correctly
    * Core time structure (open_time_ms) is sane
    * No NaNs / infs in numeric columns
    * Required feature groups exist (Momentum, Trend, Vol, Reversion,
      Orderflow, Structure, Pattern, Regime-derived)
    * No obvious label leakage (no next_* / target_* columns)
    * Regime-related columns are consistent with Pillar 4 design
    * Distributions of key features look sane (no all-zeros, no constants)
- Prints PASS/FAIL for each check and a final summary.

This is NOT a pytest; it's a manual diagnostic hub to give you confidence
that Pillar 1 Parts 1–2 are behaving as expected.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_pillar1_primitives.parquet"

MAX_ROWS_FOR_STATS = 200_000  # downsample for heavier stats if needed


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str = ""
    extra: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _print_check(cr: CheckResult) -> None:
    status = "PASS" if cr.passed else "FAIL"
    print(f"[{status}] {cr.name}")
    if cr.details:
        print(f"  {cr.details}")
    if cr.extra:
        for k, v in cr.extra.items():
            print(f"  - {k}: {v}")


def load_features(limit_stats: bool = True) -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Expected parquet not found: {FEATURES_PATH}")

    _print_header("Loading Pillar 1 primitives parquet")
    print(f"Path: {FEATURES_PATH}")

    df = pd.read_parquet(FEATURES_PATH)
    print(f"Loaded shape: {df.shape}")

    # Optionally downsample for heavy stats (but keep original available)
    if limit_stats and len(df) > MAX_ROWS_FOR_STATS:
        print(f"Downsampling for heavy stats: using last {MAX_ROWS_FOR_STATS:,} rows "
              f"out of {len(df):,}")
        df = df.tail(MAX_ROWS_FOR_STATS).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------
def check_basic_structure(df: pd.DataFrame) -> List[CheckResult]:
    results: List[CheckResult] = []

    # 1) open_time_ms presence
    if "open_time_ms" not in df.columns:
        results.append(CheckResult(
            name="Core column 'open_time_ms' exists",
            passed=False,
            details="open_time_ms is missing. Pillar 1 must be time-indexed by open_time_ms."
        ))
        return results

    ts = df["open_time_ms"]
    is_sorted = ts.is_monotonic_increasing
    has_nans = ts.isna().any()
    has_dupes = ts.duplicated().any()

    details = (f"rows={len(df):,}, cols={df.shape[1]}, "
               f"ts_range=[{int(ts.iloc[0])} .. {int(ts.iloc[-1])}]")
    extra = {
        "is_sorted": bool(is_sorted),
        "has_nans": bool(has_nans),
        "has_duplicates": bool(has_dupes),
    }
    passed = is_sorted and not has_nans and not has_dupes
    results.append(CheckResult(
        name="Time index (open_time_ms) sanity",
        passed=passed,
        details=("OK" if passed else
                 "Time index has issues: "
                 f"sorted={is_sorted}, nans={has_nans}, dupes={has_dupes}. "
                 "This can break any time-based logic and regressions."),
        extra=extra,
    ))

    # Check approximate 1m spacing (not strict fail, just informative)
    diffs = ts.diff().dropna()
    median_diff = int(diffs.median())
    unique_diffs = sorted(diffs.value_counts().head(3).index.tolist())
    results.append(CheckResult(
        name="Approximate bar spacing",
        passed=True,
        details=("Median delta_ms ~ 60_000 is expected for 1m; "
                 "this is informational, not a hard constraint."),
        extra={
            "median_delta_ms": median_diff,
            "top_deltas_ms": unique_diffs,
        },
    ))

    return results


def check_nan_inf(df: pd.DataFrame) -> List[CheckResult]:
    results: List[CheckResult] = []
    num_cols = df.select_dtypes(include=[np.number]).columns

    n_nan = df[num_cols].isna().sum().sum()
    n_inf = np.isinf(df[num_cols].to_numpy()).sum()

    passed = (n_nan == 0) and (n_inf == 0)
    details = (f"numeric_cols={len(num_cols)}, NaNs={int(n_nan)}, infs={int(n_inf)}")
    if not passed:
        details += (" → Pillar 1 primitives must be fully numeric and finite. "
                    "NaN/inf indicates missing fills or divide-by-zero issues.")

    results.append(CheckResult(
        name="No NaNs / infs in numeric features",
        passed=passed,
        details=details,
    ))
    return results


def check_feature_groups(df: pd.DataFrame) -> List[CheckResult]:
    """
    Verify that each feature category in the Pillar 1 plan is present and non-empty.
    """
    results: List[CheckResult] = []

    # Define expected groups by prefix / substrings
    group_specs = {
        "Momentum primitives": [
            "mom_roc_", "mom_ret_z_", "mom_obv", "mom_vol_adj", "mom_accel_"
        ],
        "Trend primitives": [
            "trend_slope_", "trend_strength_", "trend_efficiency_",
            "trend_noise_ratio_", "trend_ma_dist_", "trend_htf_align_"
        ],
        "Volatility primitives": [
            "vol_atr_14", "vol_atr_expansion_", "vol_atr_contraction_",
            "vol_realised_", "vol_percentile_", "vol_cluster_"
        ],
        "Reversion primitives": [
            "rev_rsi_", "rev_zscore_", "rev_accel_", "rev_vol_norm"
        ],
        "Orderflow primitives": [
            "of_delta", "of_delta_ema_", "of_taker_imbalance",
            "of_sweep_flag", "of_absorption_flag", "of_cvd"
        ],
        "Structure primitives": [
            "struct_swing_", "struct_range_", "struct_bos_",
            "struct_choch_", "struct_liquidity_sweep_"
        ],
        "Pattern primitives": [
            "pat_pin_", "pat_engulfing_", "pat_momentum_bar",
            "pat_pattern_cluster_"
        ],
        "Regime-derived primitives": [
            "regime_id_stable", "regime_confidence", "regime_is_",
            "regime_hold_length_norm", "regime_chop_mask"
        ],
    }

    cols = df.columns.tolist()

    for group_name, patterns in group_specs.items():
        matched: List[str] = []
        for pat in patterns:
            matched.extend([c for c in cols if pat in c])

        matched = sorted(set(matched))
        passed = len(matched) > 0
        if not passed:
            details = (f"No columns found for {group_name}. "
                       "This suggests the corresponding primitive engine "
                       "did not run or was renamed.")
        else:
            details = f"Found {len(matched)} columns."

        results.append(CheckResult(
            name=f"Feature group present: {group_name}",
            passed=passed,
            details=details,
            extra={"columns": matched[:20] + (["..."] if len(matched) > 20 else [])},
        ))

    return results


def check_label_leakage(df: pd.DataFrame) -> List[CheckResult]:
    """
    Ensure no obvious label/target columns are present in the primitive matrix.
    """
    results: List[CheckResult] = []

    leak_prefixes = ("next_", "target_", "label_", "y_")
    suspected = [c for c in df.columns if c.startswith(leak_prefixes)]

    passed = len(suspected) == 0
    details = ("No obvious label columns detected."
               if passed else
               "Found columns that look like labels / targets. "
               "They MUST NOT be present in primitive features; "
               "this would cause catastrophic leakage in model training.")
    results.append(CheckResult(
        name="No label leakage (next_*/target_*/label_*/y_*)",
        passed=passed,
        details=details,
        extra={"suspected_columns": suspected},
    ))

    return results


def check_regime_consistency(df: pd.DataFrame) -> List[CheckResult]:
    results: List[CheckResult] = []

    # regime_id_stable
    if "regime_id_stable" in df.columns:
        rid = df["regime_id_stable"]
        unique_vals = sorted(rid.unique().tolist())
        bad_mask = ~rid.isin([-1, 0, 1, 2, 3, 4, 5, 6, 7])
        n_bad = int(bad_mask.sum())
        passed = n_bad == 0
        details = (f"Unique regime_id_stable values: {unique_vals}. "
                   "Allowed: -1 (fallback), 0..7.")
        if not passed:
            details += f" Found {n_bad} rows with invalid ids."
        results.append(CheckResult(
            name="Regime ID stable is within expected 0..7 / -1",
            passed=passed,
            details=details,
        ))
    else:
        results.append(CheckResult(
            name="Regime ID stable exists",
            passed=False,
            details="regime_id_stable not found. Pillar 1 should have a numeric regime id "
                    "derived from Pillar 4 stable regime labels.",
        ))

    # regime_confidence
    if "regime_confidence" in df.columns:
        rc = df["regime_confidence"].astype(float)
        min_v, max_v = float(rc.min()), float(rc.max())
        passed = (min_v >= 0.0 - 1e-6) and (max_v <= 1.0 + 1e-6)
        details = f"range=[{min_v:.4f}, {max_v:.4f}] (expected ~[0,1])"
        if not passed:
            details += " → Regime confidence should be a probability in [0,1]."
        results.append(CheckResult(
            name="Regime confidence in [0,1]",
            passed=passed,
            details=details,
        ))

    # regime_hold_length_norm
    if "regime_hold_length_norm" in df.columns:
        hl = df["regime_hold_length_norm"].astype(float)
        min_v, max_v = float(hl.min()), float(hl.max())
        passed = (min_v >= -1e-6) and (max_v <= 1.0 + 1e-6)
        details = f"range=[{min_v:.4f}, {max_v:.4f}] (expected ~[0,1])"
        if not passed:
            details += " → This is meant to be a normalised hold length."
        results.append(CheckResult(
            name="Regime hold_length_norm in [0,1]",
            passed=passed,
            details=details,
        ))

    # regime_chop_mask
    if "regime_chop_mask" in df.columns:
        cm = df["regime_chop_mask"].astype(float)
        min_v, max_v = float(cm.min()), float(cm.max())
        passed = (min_v >= 0.0 - 1e-6) and (max_v <= 1.0 + 1e-6)
        details = f"range=[{min_v:.4f}, {max_v:.4f}] (expected ~[0,1])"
        if not passed:
            details += " → Chop mask should be a damping factor in [0,1]."
        results.append(CheckResult(
            name="Regime chop_mask in [0,1]",
            passed=passed,
            details=details,
        ))

    # deterministic flags
    for flag_name in ["regime_is_trend", "regime_is_chop",
                      "regime_is_mean_reversion",
                      "regime_is_vol_expansion",
                      "regime_is_liq_drought"]:
        if flag_name in df.columns:
            v = df[flag_name]
            unique_vals = sorted(v.unique().tolist())
            passed = set(unique_vals).issubset({0, 1})
            details = f"unique={unique_vals} (expected subset of {{0,1}})"
            results.append(CheckResult(
                name=f"Deterministic regime flag '{flag_name}' is binary",
                passed=passed,
                details=details,
            ))
        # If missing, we don't hard-fail, but you can add a check if you want.

    return results


def check_constant_and_sparse(df: pd.DataFrame) -> List[CheckResult]:
    """
    Detect:
    - Fully constant numeric columns (probably useless / bugged)
    - Extremely sparse binary flags (e.g. <0.01% ones) – not fail, just warn.
    """
    results: List[CheckResult] = []

    num_cols = df.select_dtypes(include=[np.number]).columns
    sample = df[num_cols]

    # Constant columns
    stds = sample.std(axis=0)
    const_cols = stds[stds == 0].index.tolist()

    passed_const = len(const_cols) == 0
    details_const = (f"{len(const_cols)} constant numeric columns."
                     " Constant columns may indicate dead features or bugs.")
    results.append(CheckResult(
        name="No constant numeric feature columns",
        passed=passed_const,
        details=details_const,
        extra={"constant_columns": const_cols[:50]},
    ))

    # Sparse flags
    sparse_info = []
    for c in num_cols:
        if sample[c].dropna().isin([0, 1]).all():
            rate = float(sample[c].mean())  # fraction of ones
            if 0 < rate < 0.0001:  # <0.01% ones
                sparse_info.append((c, rate))

    results.append(CheckResult(
        name="Sparse binary flags summary (not a failure, just info)",
        passed=True,
        details=f"Found {len(sparse_info)} very sparse binary flags (<0.01% ones). "
                "Inspect if they match extremely rare patterns like sweeps.",
        extra={"sparse_flags": [(c, f"{rate:.6f}") for c, rate in sparse_info[:50]]},
    ))

    return results


def check_key_distributions(df: pd.DataFrame) -> List[CheckResult]:
    """
    High-level distributions for a few key representative features.
    Only informative; doesn't hard-fail unless something is obviously insane.
    """
    results: List[CheckResult] = []

    key_cols = [
        "mom_roc_1",
        "mom_ret_z_20",
        "mom_obv_slope_20",
        "trend_slope_20",
        "trend_efficiency_20",
        "vol_atr_14",
        "vol_realised_5m",
        "rev_rsi_14",
        "of_delta",
        "struct_range_width_50",
        "pat_pattern_cluster_10",
    ]

    present = [c for c in key_cols if c in df.columns]
    missing = [c for c in key_cols if c not in df.columns]

    if present:
        desc = df[present].describe(percentiles=[0.01, 0.5, 0.99]).to_string()
        results.append(CheckResult(
            name="Key feature distribution snapshot",
            passed=True,
            details="Descriptive stats for a subset of key primitives:\n" + desc,
            extra={"missing_key_features": missing},
        ))
    else:
        results.append(CheckResult(
            name="Key feature distribution snapshot",
            passed=False,
            details="None of the expected key primitive columns were found. "
                    "This suggests the primitive builder is not wired to your plan.",
            extra={"missing_key_features": missing},
        ))

    return results


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------
def run_all_checks(df: pd.DataFrame) -> List[CheckResult]:
    all_results: List[CheckResult] = []

    all_results.extend(check_basic_structure(df))
    all_results.extend(check_nan_inf(df))
    all_results.extend(check_feature_groups(df))
    all_results.extend(check_label_leakage(df))
    all_results.extend(check_regime_consistency(df))
    all_results.extend(check_constant_and_sparse(df))
    all_results.extend(check_key_distributions(df))

    return all_results


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    df = load_features(limit_stats=True)
    _print_header("Running Pillar 1 primitives validation checks")

    results = run_all_checks(df)

    n_pass = sum(r.passed for r in results)
    n_fail = len(results) - n_pass

    for r in results:
        _print_check(r)

    _print_header("Summary")
    print(f"Total checks: {len(results)}")
    print(f"Passed      : {n_pass}")
    print(f"Failed      : {n_fail}")

    if n_fail == 0:
        print("\nOverall verdict: ✅ Pillar 1 primitives look structurally sound "
              "and aligned with the plan (based on these checks).")
    else:
        print("\nOverall verdict: ❌ One or more checks failed.")
        print("Advanced reasoning: look at FAILED checks above. Typical causes:")
        print("- Missing feature groups → primitive engine not wired / column names changed")
        print("- NaNs / infs → missing fills, divide-by-zero, or merge issues")
        print("- Bad regime ids / probabilities → Pillar 4 outputs not propagated correctly")
        print("- Constant columns → dead features or unintended broadcasting")
        print("\nFix the underlying issue, rebuild features_pillar1_primitives.parquet, "
              "and re-run this test script until all checks pass.")
    if n_fail > 0:
        # Non-zero exit code for CI / automation
        raise SystemExit(1)


if __name__ == "__main__":
    main()