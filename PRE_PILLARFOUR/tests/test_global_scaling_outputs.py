import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

# Make sure project root is on sys.path so we can import the engine module
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from global_scaling_engine import (  # noqa: E402
    GlobalScalingEngine,
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    DEFAULT_PARAMS,
    DEFAULT_REGIME_COL,
    CLIP_DRIFT,
    collect_feature_list,
    ensure_open_time_ms,
)

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"{GREEN}[PASS]{RESET} {msg}")


def fail(msg: str, failures: list[str]) -> None:
    print(f"{RED}[FAIL]{RESET} {msg}")
    failures.append(msg)


def load_files():
    """Load canonical input, normalised output, and params JSON."""
    canon_path = DEFAULT_INPUT
    norm_path = DEFAULT_OUTPUT
    params_path = DEFAULT_PARAMS

    missing = [p for p in [canon_path, norm_path, params_path] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    canon = pl.read_parquet(canon_path)
    norm = pl.read_parquet(norm_path)
    with open(params_path) as f:
        params = json.load(f)
    return canon, norm, params


def derived_columns_for_feature(feat: str, meta: dict) -> list[str]:
    """Which derived columns we expect for a given feature based on params metadata."""
    cols = [
        f"{feat}_z",
        f"{feat}_mad",
        f"{feat}_mm",
        f"{feat}_log",
        f"{feat}_drift",
    ]
    if meta.get("vol_ref"):
        cols.append(f"{feat}_vol")
    if meta.get("regime"):
        cols.append(f"{feat}_reg")
    return cols


def main() -> None:
    failures: list[str] = []

    # Load data
    try:
        canon, norm, params = load_files()
    except FileNotFoundError as e:
        fail(str(e), failures)
        print(f"\n{RED}SUMMARY: {len(failures)} failure(s){RESET}")
        sys.exit(1)

    # Ensure timestamps exist / normalised
    canon = ensure_open_time_ms(canon)
    norm = ensure_open_time_ms(norm)

    # === Test 1: Row count and timestamp alignment ===
    if canon.height == norm.height:
        ok(f"Row count matches: {canon.height:,}")
    else:
        fail(f"Row count mismatch canon={canon.height:,} norm={norm.height:,}", failures)

    canon_ts = canon.sort("open_time_ms")["open_time_ms"]
    norm_ts = norm.sort("open_time_ms")["open_time_ms"]
    if (canon_ts == norm_ts).all():
        ok("open_time_ms aligned exactly")
    else:
        fail("open_time_ms misalignment detected between canonical and normalised data", failures)

    # === Test 2: Raw columns preserved exactly ===
    suffixes = ("_z", "_mad", "_vol", "_reg", "_mm", "_log", "_drift")
    raw_cols = [c for c in canon.columns if not any(c.endswith(s) for s in suffixes)]

    missing_raw = [c for c in raw_cols if c not in norm.columns]
    if missing_raw:
        fail(f"Raw columns missing in normalised output: {missing_raw}", failures)
    else:
        canon_sorted = canon.sort("open_time_ms")
        norm_sorted = norm.sort("open_time_ms")

        differing: list[str] = []
        for col in raw_cols:
            s1 = canon_sorted[col]
            s2 = norm_sorted[col]
            if not (s1 == s2).all():
                differing.append(col)

        if not differing:
            ok("All raw columns preserved exactly between canonical and normalised data")
        else:
            fail(
                f"Raw columns differ between canonical and normalised: {differing[:10]} "
                f"(showing first 10)",
                failures,
            )

    # === Test 3: Derived columns presence per feature (based on params) ===
    param_features: dict = params.get("features", {})
    missing_any: list[str] = []
    for feat, meta in param_features.items():
        expected = derived_columns_for_feature(feat, meta)
        for col in expected:
            if col not in norm.columns:
                missing_any.append(col)
    if missing_any:
        fail(f"Expected derived columns missing in normalised output: {missing_any}", failures)
    else:
        ok("All expected derived columns present in normalised output")

    # Collect all derived columns for downstream tests
    derived_cols = [
        c
        for c in norm.columns
        if any(c.endswith(s) for s in ("_z", "_mad", "_vol", "_reg", "_mm", "_log", "_drift"))
    ]

    # === Test 4: No NaN or inf in derived columns ===
    if derived_cols:
        nan_counts_df = norm.select([pl.col(c).cast(pl.Float64).is_nan().sum().alias(c) for c in derived_cols])
        inf_counts_df = norm.select([pl.col(c).cast(pl.Float64).is_infinite().sum().alias(c) for c in derived_cols])

        nan_total = sum(nan_counts_df.select(pl.all()).row(0))
        inf_total = sum(inf_counts_df.select(pl.all()).row(0))

        if nan_total == 0 and inf_total == 0:
            ok("No NaN or inf in derived columns")
        else:
            detail = []
            nan_row = nan_counts_df.row(0, named=True)
            inf_row = inf_counts_df.row(0, named=True)
            for c in derived_cols:
                n = nan_row[c]
                i = inf_row[c]
                if n or i:
                    detail.append(f"{c}: NaN={n}, inf={i}")
            fail(
                f"Found NaNs or infs in derived columns "
                f"(total NaN={nan_total}, inf={inf_total}): {detail}",
                failures,
            )
    else:
        ok("No derived columns found (nothing to check for NaN/inf)")

    # === Test 5: Drift clipping check ===
    drift_cols = [c for c in derived_cols if c.endswith("_drift")]
    if drift_cols:
        drift_stats_df = norm.select(
            [pl.col(c).max().alias(f"max_{c}") for c in drift_cols]
            + [pl.col(c).min().alias(f"min_{c}") for c in drift_cols]
        )
        stats_row = drift_stats_df.row(0, named=True)
        max_vals = {k: v for k, v in stats_row.items() if k.startswith("max_")}
        min_vals = {k: v for k, v in stats_row.items() if k.startswith("min_")}

        max_ok = all(val <= CLIP_DRIFT + 1e-6 for val in max_vals.values())
        min_ok = all(val >= -CLIP_DRIFT - 1e-6 for val in min_vals.values())

        if max_ok and min_ok:
            ok("Drift columns within clipping bounds")
        else:
            fail(
                f"Drift columns exceed clipping bounds. "
                f"Max stats: {max_vals}, Min stats: {min_vals}",
                failures,
            )
    else:
        ok("No drift columns found (skipping drift clipping check)")

    # === Test 6: Derived dtypes are numeric ===
    numeric_types = (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    non_numeric = [c for c in derived_cols if norm.schema[c] not in numeric_types]
    if non_numeric:
        fail(f"Non-numeric derived columns detected: {non_numeric}", failures)
    else:
        ok("All derived columns are numeric types")

    # === Test 7: Min–max columns in [0,1] ===
    mm_cols = [c for c in derived_cols if c.endswith("_mm")]
    if mm_cols:
        min_res_df = norm.select([pl.col(c).min().alias(f"min_{c}") for c in mm_cols])
        max_res_df = norm.select([pl.col(c).max().alias(f"max_{c}") for c in mm_cols])
        min_row = min_res_df.row(0, named=True)
        max_row = max_res_df.row(0, named=True)

        bad: list[str] = []
        for c in mm_cols:
            cmin = min_row[f"min_{c}"]
            cmax = max_row[f"max_{c}"]
            if cmin < -1e-8 or cmax > 1 + 1e-8:
                bad.append(f"{c} (min={cmin:.4f}, max={cmax:.4f})")

        if not bad:
            ok("All _mm columns lie within [0,1] (within tolerance)")
        else:
            fail(f"_mm columns outside [0,1] bounds: {bad}", failures)
    else:
        ok("No _mm columns found (skipping min–max check)")

    # === Test 8: Log scaling monotonic in |x| ===
    log_cols = [c for c in derived_cols if c.endswith("_log")]
    log_violations: list[str] = []

    for log_col in log_cols:
        base_col = log_col[:-4]
        if base_col not in norm.columns:
            continue

        raw = norm[base_col].to_numpy()
        logv = norm[log_col].to_numpy()

        mask = np.isfinite(raw) & np.isfinite(logv)
        if mask.sum() < 2:
            continue

        raw_abs = np.abs(raw[mask])
        log_abs = np.abs(logv[mask])

        order = np.argsort(raw_abs)
        # Subsample to keep this cheap on very large datasets
        if len(order) > 5000:
            step = len(order) // 5000
            order = order[::step]

        raw_sorted = raw_abs[order]
        log_sorted = log_abs[order]

        tol_raw = 1e-9
        tol_log = 1e-6
        local_violations = 0

        for i in range(1, len(raw_sorted)):
            if raw_sorted[i] > raw_sorted[i - 1] + tol_raw and log_sorted[i] < log_sorted[i - 1] - tol_log:
                local_violations += 1
                if local_violations <= 3:
                    log_violations.append(
                        f"{log_col}: |x| increased from {raw_sorted[i-1]:.4f} to {raw_sorted[i]:.4f} "
                        f"but |log_x| decreased from {log_sorted[i-1]:.4f} to {log_sorted[i]:.4f}"
                    )
                else:
                    break

    if not log_violations:
        ok("Log scaling monotonicity check passed for all _log columns")
    else:
        fail(
            "Log scaling monotonicity violations detected:\n  " + "\n  ".join(log_violations),
            failures,
        )

    # === Test 9: Deterministic reapply check on a slice ===
    sample_n = min(2000, canon.height)
    if sample_n > 0:
        features = list(param_features.keys())
        regime_col = params.get("meta", {}).get("regime_col", DEFAULT_REGIME_COL)
        engine = GlobalScalingEngine(regime_col=regime_col)

        canon_slice = canon.head(sample_n)

        # Cast regime column to match params key type (JSON stores keys as strings)
        canon_slice_apply = canon_slice
        key_type = None
        for meta in param_features.values():
            reg_map = meta.get("regime") or {}
            if reg_map:
                key_type = type(next(iter(reg_map.keys())))
                break
        if regime_col in canon_slice.columns and key_type is not None:
            if key_type is str and canon_slice.schema[regime_col] != pl.Utf8:
                canon_slice_apply = canon_slice_apply.with_columns(pl.col(regime_col).cast(pl.Utf8))
            elif key_type in (int, float) and canon_slice.schema[regime_col] == pl.Utf8:
                canon_slice_apply = canon_slice_apply.with_columns(pl.col(regime_col).cast(pl.Int64))

        recomputed = engine.apply(canon_slice_apply, features, params)

        # Align on timestamp
        norm_sorted = norm.sort("open_time_ms").head(sample_n)
        recomputed_sorted = recomputed.sort("open_time_ms")

        derived_subset = [c for c in derived_cols if c in recomputed_sorted.columns]
        max_diff = 0.0

        for col in derived_subset:
            s1 = norm_sorted[col].to_numpy()
            s2 = recomputed_sorted[col].to_numpy()
            diff = np.max(np.abs(s1 - s2))
            if diff > max_diff:
                max_diff = diff

        if derived_subset and max_diff < 1e-5:
            ok(
                f"Deterministic reapply check passed on first {sample_n} rows "
                f"(max abs diff {max_diff:.2e})"
            )
        elif not derived_subset:
            ok("No overlapping derived columns for determinism check (skipping)")
        else:
            fail(
                f"Deterministic reapply check failed on first {sample_n} rows "
                f"(max abs diff {max_diff})",
                failures,
            )

    # === SUMMARY & EXIT CODE ===
    if failures:
        print(f"\n{RED}SUMMARY: {len(failures)} failure(s){RESET}")
        for msg in failures:
            print(f"- {msg}")
        sys.exit(1)
    else:
        print(f"\n{GREEN}SUMMARY: ALL TESTS PASSED{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
