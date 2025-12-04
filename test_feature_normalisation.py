import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


USE_COLOR = sys.stdout.isatty()
GREEN = "\033[92m" if USE_COLOR else ""
RED = "\033[91m" if USE_COLOR else ""
YELLOW = "\033[93m" if USE_COLOR else ""
RESET = "\033[0m" if USE_COLOR else ""


def tag(ok: bool) -> str:
    return f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"


RESULTS = []


def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{tag(ok)}] {name}")
    if detail:
        print(f"  {detail}")


def section(title: str) -> None:
    print("\n" + "─" * 10 + f" {title} " + "─" * 10)


ROOT = Path(__file__).resolve().parent
DATA_PROCESSED = ROOT / "PILLAR_FOUR_REGIME_DETECTION" / "data" / "processed"
RAW_PATHS = [
    DATA_PROCESSED / "phase0" / "canonical_raw_with_utilities.parquet",
    DATA_PROCESSED / "raw_layer" / "canonical_raw_with_utilities.parquet",
]
NORMALISED_PARQUET = DATA_PROCESSED / "normalisation" / "normalised_features.parquet"
PARAMS_JSON = DATA_PROCESSED / "normalisation" / "normalisation_params.json"
DRIFT_REPORT = DATA_PROCESSED / "normalisation" / "drift_report.txt"

sys.path.append(str(ROOT / "PHASE_0.1" / "GLOBAL_NORMALISATION"))
from normalisation_engine import NormalisationEngine  # noqa: E402


def load_raw() -> pd.DataFrame:
    for path in RAW_PATHS:
        if path.exists():
            df = pd.read_parquet(path)
            record("Load canonical_raw_with_utilities.parquet", True, f"Rows={df.shape[0]} Cols={df.shape[1]} Path={path}")
            return df
    record("Load canonical_raw_with_utilities.parquet", False, f"Not found in {RAW_PATHS}")
    return pd.DataFrame()


def load_normalised() -> pd.DataFrame:
    if NORMALISED_PARQUET.exists():
        df = pd.read_parquet(NORMALISED_PARQUET)
        record("Load normalised_features.parquet", True, f"Rows={df.shape[0]} Cols={df.shape[1]} Path={NORMALISED_PARQUET}")
        return df
    record("Load normalised_features.parquet", False, f"Missing at {NORMALISED_PARQUET}")
    return pd.DataFrame()


def load_params() -> dict:
    if PARAMS_JSON.exists():
        params = json.loads(PARAMS_JSON.read_text())
        record("Load normalisation_params.json", True, f"Keys={list(params.keys())}")
        return params
    record("Load normalisation_params.json", False, f"Missing at {PARAMS_JSON}")
    return {}


def check_shapes(raw_df: pd.DataFrame, norm_df: pd.DataFrame) -> None:
    same_len = len(raw_df) == len(norm_df)
    ts_match = "open_time_ms" in raw_df.columns and "open_time_ms" in norm_df.columns and raw_df["open_time_ms"].equals(norm_df["open_time_ms"])
    record("Shape match raw vs normalised", same_len, f"raw={raw_df.shape}, norm={norm_df.shape}")
    record("Timestamps identical", ts_match, "Checked open_time_ms column")


def safe_series(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series(dtype=float)


def log_scaling_test(raw_df: pd.DataFrame, norm_df: pd.DataFrame) -> None:
    cols = [c for c in norm_df.columns if c.endswith("_log")]
    if not cols:
        record("Log scaling presence", False, "No *_log columns")
        return
    all_ok = True
    details = []
    for col in cols:
        base = col[:-4]
        if base not in raw_df.columns:
            all_ok = False
            details.append(f"Missing base column for {col}")
            continue
        raw = raw_df[base]
        expected = np.sign(raw) * np.log1p(np.abs(raw))
        actual = norm_df[col]
        mask = ~raw.isna()
        diff = np.abs(expected[mask] - actual[mask])
        finite_ok = np.isfinite(actual).all()
        tol_ok = (diff < 1e-6).all()
        if not (finite_ok and tol_ok):
            all_ok = False
            details.append(f"{col}: finite={finite_ok} tol={tol_ok} maxdiff={diff.max() if len(diff) else 0:.2e}")
    record("Log scaling validation", all_ok, "; ".join(details) if details else "All *_log columns valid")


def sample_indices(n: int, total: int) -> np.ndarray:
    total = max(total, 1)
    if n >= total:
        return np.arange(total)
    return np.array(random.sample(range(total), n))


def zscore_test(raw_df: pd.DataFrame, norm_df: pd.DataFrame, params: dict) -> None:
    z_cols = [c for c in norm_df.columns if c.endswith("_zscore")]
    if not z_cols:
        record("Z-score presence", False, "No *_zscore columns")
        return
    window = int(params.get("meta", {}).get("config", {}).get("zscore_window", 1000))
    clip = float(params.get("meta", {}).get("config", {}).get("zscore_clip", 5.0))
    idxs = sample_indices(200, len(raw_df))
    all_ok = True
    details = []
    for col in z_cols:
        base = col[:-7]
        if base not in raw_df.columns:
            all_ok = False
            details.append(f"{col}: missing base")
            continue
        raw_series = raw_df[base]
        norm_series = norm_df[col]
        for i in idxs:
            window_slice = raw_series.iloc[max(0, i - window):i]
            if window_slice.empty:
                continue
            mean = window_slice.mean()
            std = window_slice.std(ddof=0)
            if std == 0 or np.isnan(std):
                continue
            expected = (raw_series.iloc[i] - mean) / std
            expected = max(min(expected, clip), -clip)
            actual = norm_series.iloc[i]
            if not np.isfinite(actual) or abs(actual - expected) > 1e-5:
                all_ok = False
                details.append(f"{col}[{i}]: expected={expected:.4f}, actual={actual:.4f}")
                break
    record("Z-score rolling/leak-free validation", all_ok, "; ".join(details) if details else "All sampled rows match")


def mad_test(raw_df: pd.DataFrame, norm_df: pd.DataFrame, params: dict) -> None:
    mad_cols = [c for c in norm_df.columns if c.endswith("_mad")]
    if not mad_cols:
        record("MAD presence", False, "No *_mad columns")
        return
    window = int(params.get("meta", {}).get("config", {}).get("mad_window", 1000))
    clip = float(params.get("meta", {}).get("config", {}).get("mad_clip", 6.0))
    idxs = sample_indices(200, len(raw_df))
    all_ok = True
    details = []
    for col in mad_cols:
        base = col[:-4]
        if base not in raw_df.columns:
            all_ok = False
            details.append(f"{col}: missing base")
            continue
        raw_series = raw_df[base]
        norm_series = norm_df[col]
        for i in idxs:
            window_slice = raw_series.iloc[max(0, i - window):i]
            if window_slice.empty:
                continue
            median = window_slice.median()
            mad = np.median(np.abs(window_slice - median))
            denom = mad * 1.4826
            if denom == 0 or np.isnan(denom):
                continue
            expected = (raw_series.iloc[i] - median) / denom
            expected = max(min(expected, clip), -clip)
            actual = norm_series.iloc[i]
            if not np.isfinite(actual) or abs(actual - expected) > 1e-5:
                all_ok = False
                details.append(f"{col}[{i}]: expected={expected:.4f}, actual={actual:.4f}")
                break
    record("MAD rolling/leak-free validation", all_ok, "; ".join(details) if details else "All sampled rows match")


def vol_scaling_test(raw_df: pd.DataFrame, norm_df: pd.DataFrame, params: dict) -> None:
    vol_entries = [k for k in norm_df.columns if k.endswith("_volnorm")]
    if not vol_entries:
        record("Volatility scaling presence", False, "No *_volnorm columns")
        return
    scalers = params.get("scalers", {})
    all_ok = True
    details = []
    for col in vol_entries:
        feat = col[:-8]
        key = f"{feat}|vol"
        scaler_params = scalers.get(key, {})
        denom_col = scaler_params.get("denominator_col")
        epsilon = scaler_params.get("epsilon", 1e-9)
        if feat not in raw_df.columns or denom_col not in raw_df.columns:
            all_ok = False
            details.append(f"{col}: missing base or denominator ({denom_col})")
            continue
        denom = raw_df[denom_col].replace(0, np.nan).fillna(epsilon)
        expected = raw_df[feat] / denom
        expected = expected.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(0.0)
        diff = np.abs(expected - norm_df[col])
        finite_ok = np.isfinite(norm_df[col]).all()
        tol_ok = (diff < 1e-6).all()
        if not (finite_ok and tol_ok):
            all_ok = False
            details.append(f"{col}: finite={finite_ok} tol={tol_ok} maxdiff={diff.max() if len(diff) else 0:.2e}")
    record("Volatility scaling validation", all_ok, "; ".join(details) if details else "All *_volnorm columns valid")


def regime_test(raw_df: pd.DataFrame, norm_df: pd.DataFrame, params: dict) -> None:
    regime_cols = [c for c in norm_df.columns if c.endswith("_regime_norm")]
    if not regime_cols:
        record("Regime-conditioned presence", False, "No *_regime_norm columns")
        return
    regime_col = params.get("meta", {}).get("config", {}).get("regime_col", "vol_regime")
    if regime_col not in raw_df.columns:
        record("Regime-conditioned validation", False, f"Missing regime column {regime_col}")
        return
    window = int(params.get("meta", {}).get("config", {}).get("regime_window", 500))
    clip = float(params.get("meta", {}).get("config", {}).get("regime_clip", 5.0))
    scalers = params.get("scalers", {})
    all_ok = True
    details = []
    for col in regime_cols:
        feat = col.replace("_regime_norm", "")
        key = f"{feat}|regime"
        sp = scalers.get(key, {})
        regimes = list(sp.get("scalers", {}).keys())
        series = raw_df[feat]
        regimes_series = raw_df[regime_col]
        manual = pd.Series(np.nan, index=series.index, dtype=float)
        for regime in regimes:
            mask = regimes_series == regime
            if mask.any():
                roll = series[mask].rolling(window, min_periods=1).mean().shift(1)
                std = series[mask].rolling(window, min_periods=1).std(ddof=0).shift(1).replace(0, np.nan)
                vals = ((series[mask] - roll) / std).clip(lower=-clip, upper=clip).ffill().fillna(0.0)
                manual.loc[mask] = vals
        manual = manual.ffill().fillna(0.0)
        diff = np.abs(manual - norm_df[col])
        grouped_means = norm_df[col].groupby(regimes_series).mean()
        differs = len(grouped_means) > 1 and not np.allclose(grouped_means.max(), grouped_means.min(), atol=1e-9)
        if not ((diff < 1e-5).all() and differs):
            all_ok = False
            details.append(f"{col}: diff_max={diff.max() if len(diff) else 0:.2e} regimes differ={differs}")
    record("Regime-conditioned scaling validation", all_ok, "; ".join(details) if details else "All regime-scaled columns valid")


def minmax_test(raw_df: pd.DataFrame, norm_df: pd.DataFrame, params: dict) -> None:
    mm_cols = [c for c in norm_df.columns if c.endswith("_minmax")]
    if not mm_cols:
        record("Min–max presence", False, "No *_minmax columns")
        return
    window = int(params.get("meta", {}).get("config", {}).get("minmax_window", 500))
    lo = float(params.get("meta", {}).get("config", {}).get("minmax_clip_lower", 0.01))
    hi = float(params.get("meta", {}).get("config", {}).get("minmax_clip_upper", 0.99))
    all_ok = True
    details = []
    for col in mm_cols:
        base = col.replace("_minmax", "")
        if base not in raw_df.columns:
            all_ok = False
            details.append(f"{col}: missing base")
            continue
        rmin = raw_df[base].rolling(window, min_periods=1).min().shift(1)
        rmax = raw_df[base].rolling(window, min_periods=1).max().shift(1)
        denom = (rmax - rmin).replace(0, np.nan)
        expected = ((raw_df[base] - rmin) / denom).clip(lower=lo, upper=hi).ffill().fillna(0.5)
        actual = norm_df[col]
        diff = np.abs(expected - actual)
        bounds_ok = actual.between(lo - 1e-9, hi + 1e-9).all()
        tol_ok = (diff < 1e-6).all()
        if not (bounds_ok and tol_ok):
            all_ok = False
            details.append(f"{col}: bounds={bounds_ok} tol={tol_ok} maxdiff={diff.max() if len(diff) else 0:.2e}")
    record("Min–max scaling validation", all_ok, "; ".join(details) if details else "All *_minmax columns valid")


def drift_report_test() -> None:
    if DRIFT_REPORT.exists():
        lines = DRIFT_REPORT.read_text().strip().splitlines()
        count = len([ln for ln in lines if ln.strip()])
        record("Drift detection report", True, f"Entries={count}")
    else:
        record("Drift detection report", False, f"Missing at {DRIFT_REPORT}")


def param_reload_test(raw_df: pd.DataFrame, norm_df: pd.DataFrame) -> None:
    try:
        engine = NormalisationEngine.load(PARAMS_JSON, logger=None)
        regen = engine.transform(raw_df)
        common_cols = [c for c in norm_df.columns if c in regen.columns]
        match = np.allclose(norm_df[common_cols].values, regen[common_cols].values, atol=0, rtol=0)
        record("Parameter reload reproducibility", match, f"Compared columns={len(common_cols)}")
    except Exception as exc:
        record("Parameter reload reproducibility", False, f"Load/transform failed: {exc}")


def determinism_test(raw_df: pd.DataFrame) -> None:
    try:
        engine = NormalisationEngine.load(PARAMS_JSON, logger=None)
        subset = raw_df.head(min(500, len(raw_df))).copy()
        out1 = engine.transform(subset)
        out2 = engine.transform(subset)
        identical = np.array_equal(out1.values, out2.values)
        record("Deterministic transform repeatability", identical, f"Rows={len(subset)} Cols={out1.shape[1]}")
    except Exception as exc:
        record("Deterministic transform repeatability", False, f"Failed: {exc}")


def distribution_test(norm_df: pd.DataFrame) -> None:
    checks = {
        "Log Scaling": [c for c in norm_df.columns if c.endswith("_log")],
        "Z-Score": [c for c in norm_df.columns if c.endswith("_zscore")],
        "MAD": [c for c in norm_df.columns if c.endswith("_mad")],
        "Volatility Scaling": [c for c in norm_df.columns if c.endswith("_volnorm")],
        "Min–Max": [c for c in norm_df.columns if c.endswith("_minmax")],
        "Regime-Split": [c for c in norm_df.columns if c.endswith("_regime_norm")],
    }
    for name, cols in checks.items():
        if not cols:
            record(f"{name} distribution", False, "No columns found")
            continue
        sample = norm_df[cols].select_dtypes(include=[np.number]).stack()
        if sample.empty:
            record(f"{name} distribution", False, "No numeric data")
            continue
        stats = sample.describe()
        pathologic = not np.isfinite(sample).all() or stats["std"] == 0 or np.isnan(stats["mean"])
        detail = f"count={stats['count']:.0f} mean={stats['mean']:.3f} std={stats['std']:.3f} min={stats['min']:.3f} max={stats['max']:.3f}"
        record(f"{name} distribution", not pathologic, detail)


def summary() -> None:
    total = len(RESULTS)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = total - passed
    print("\nSCALING TYPE               STATUS")
    print("--------------------------------")
    for name, ok, _ in RESULTS:
        print(f"{name:<27} {tag(ok)}")
    print("\n──────────── SUMMARY ────────────")
    print(f"Tests run: {total}")
    print(f"Passed: {GREEN if USE_COLOR else ''}{passed}{RESET if USE_COLOR else ''}")
    print(f"Failed: {RED if failed and USE_COLOR else ''}{failed}{RESET if USE_COLOR else ''}")
    overall = f"{GREEN}PASS ✓{RESET}" if failed == 0 else f"{RED}FAIL ✗{RESET}"
    print(overall)
    if failed:
        print("Failures:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"- {name}: {detail}")


def main():
    random.seed(0)
    section("LOADING")
    raw_df = load_raw()
    norm_df = load_normalised()
    params = load_params()
    if raw_df.empty or norm_df.empty or not params:
        summary()
        return

    check_shapes(raw_df, norm_df)

    section("SCALING TESTS")
    log_scaling_test(raw_df, norm_df)
    zscore_test(raw_df, norm_df, params)
    mad_test(raw_df, norm_df, params)
    vol_scaling_test(raw_df, norm_df, params)
    regime_test(raw_df, norm_df, params)
    minmax_test(raw_df, norm_df, params)

    section("DRIFT / PARAMS / DETERMINISM")
    drift_report_test()
    param_reload_test(raw_df, norm_df)
    determinism_test(raw_df)

    section("DISTRIBUTION CHECKS")
    distribution_test(norm_df)

    summary()


if __name__ == "__main__":
    main()
