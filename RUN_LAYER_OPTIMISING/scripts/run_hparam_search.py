#!/usr/bin/env python3
"""
run_hparam_search.py

PROPER PARAM-TUNING WALK-FORWARD (6mo fit, 1mo val, 1mo test, step 1mo)
========================================================================
Goal: tune ONLY decision params (thresholds/filters). No cheating on test.

Per fold:
  FIT  window: fit models/scalers on [fit_start, val_start)    (~6 months)
  VAL  window: run inference on [val_start, test_start)        (1 month)
               tune params on VAL using REALIZED backtest
  TEST window: run inference on [test_start, test_end)         (1 month)
               report REALIZED performance on TEST
  Step forward: 1 month

Key properties:
- Leak-safe: params are selected ONLY on VAL, never on TEST.
- Fold isolation: wipe RUN_LAYER/data/{raw,processed} each fold (prevents cross-fold reuse).
- Within-fold safety: snapshot FIT processed/ and restore it before VAL and before TEST.
- Realized scoring: path-based barrier backtest using 1m candles (TP/SL first touch),
  conservative same-bar rule.

Adds:
- Live per-fold summary table printed after EACH fold (and saved to CSV)
- Final edge checks:
  - median(test_total_r) > 0
  - % folds with test_total_r > 0 (target > 60%)

Outputs per fold:
  RUN_LAYER_OPTIMISING/optimization_outputs/fold_YYYYMMDD/
    best_params.json
    best_metrics.json
    candidates_topk_val.csv
    taken_trades_val.parquet
    taken_trades_test.parquet
    equity_val.png
    equity_test.png

Also writes:
  RUN_LAYER_OPTIMISING/optimization_outputs/fold_summary.csv

NOTE:
- You said leak #3 is fine -> not touched.
"""

from __future__ import annotations

import fnmatch
import inspect
import json
import logging
import math
import os
import random
import runpy
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

import numpy as np
import polars as pl
import pandas as pd
import importlib.util

# ---------------------------------------------------------------------------
# Repro / summary
# ---------------------------------------------------------------------------

SEED = 1337
PRINT_TOP_N_FOLDS_IN_TABLE = 200  # set small if you only want last N rows printed

# ---------------------------------------------------------------------------
# Walk-forward split config
# ---------------------------------------------------------------------------

FIT_DAYS = 30 * 6      # ~6 months
VAL_DAYS = 30          # 1 month
TEST_DAYS = 30         # 1 month
STEP_DAYS = 30         # step forward 1 month
MONTHS_BACK = 18       # how far back we start generating folds (increase for more folds)

# ---------------------------------------------------------------------------
# Optimiser config (tuning params)
# ---------------------------------------------------------------------------

# More samples because we're now searching for "good + frequent"
N_RANDOM_SAMPLES = 2000

# Prefilter is cheap EV-based; keep this low so we don't throw away candidates too early
MIN_TRADES_PREFILTER = 80

# Realized evaluation is what matters; this forces the optimiser away from low-activity configs
MIN_TRADES_REALIZED = 350

# Fast prefilter (EV) then realized rerank
TOPK_REALIZED = 160

# Realized backtest settings
# No overlap + high frequency => you must shorten max holds, otherwise trade count is physically capped.
# 45 minutes => if lots of timeouts, theoretical cap ~32 trades/day.
REALIZED_MAX_HOLD_MINUTES = 45
REALIZED_NO_OVERLAP = True
REALIZED_SAME_BAR_RULE = "SL_FIRST"  # conservative when TP & SL hit same candle

# Frequency target (VAL selection objective)
TARGET_TRADES_PER_DAY_LOW = 20.0
TARGET_TRADES_PER_DAY_HIGH = 40.0

# Penalty weights: punish being under 20/day hard; punish being over 40/day softly
FREQ_PENALTY_LOW = 1.25
FREQ_PENALTY_HIGH = 0.25

# Execution costs
SLIPPAGE_BPS = 3.0
ENTRY_MAKER_FEE = 0.0002
ENTRY_TAKER_FEE = 0.0004
ENTRY_MAKER_SHARE = 0.20
ENTRY_TAKER_SHARE = 0.80
EXIT_FEE = 0.0004

# ---------------------------------------------------------------------------
# Mac safety limits (disk + snapshot)
# ---------------------------------------------------------------------------

MIN_FREE_GB_REQUIRED = 4.0
SNAPSHOT_FULL_MAX_GB = 4.0
SNAPSHOT_MAX_TOTAL_GB = 2.5
SNAPSHOT_MAX_FILE_MB = 512

# ---------------------------------------------------------------------------
# Fold isolation toggles
# ---------------------------------------------------------------------------

CLEAR_PREVIOUS_FOLD_ARTIFACTS = True
CLEAR_RUN_LAYER_DATA_EACH_FOLD = True  # leak #2 fix

# ---------------------------------------------------------------------------
# Paths / sys.path wiring
# ---------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent

BASE_DIR: Path | None = None
for cand in [THIS_DIR] + list(THIS_DIR.parents):
    if (cand / "RUN_LAYER").exists() and (cand / "RUN_LAYER_OPTIMISING").exists():
        BASE_DIR = cand
        break
if BASE_DIR is None:
    BASE_DIR = THIS_DIR.parents[2]

RUN_LAYER_DIR = BASE_DIR / "RUN_LAYER"
RAW_DIR = RUN_LAYER_DIR / "data" / "raw"
PROCESSED_DIR = RUN_LAYER_DIR / "data" / "processed"

RUN_LAYER_SRC = BASE_DIR / "RUN_LAYER" / "src"
RUN_OPT_SRC = BASE_DIR / "RUN_LAYER_OPTIMISING" / "src"

for p in (RUN_OPT_SRC, RUN_LAYER_SRC):
    ps = str(p)
    if ps not in sys.path:
        sys.path.append(ps)

# Optional pillar src dirs (kept in case live modules import them)
P1_SRC = BASE_DIR / "PILLAR_ONE_ENTRY_MODEL" / "src"
P4_SRC = BASE_DIR / "PILLAR_FOUR_REGIME_DETECTION" / "src"
P2_SRC = BASE_DIR / "PILLAR_TWO_EXIT_MODEL" / "src"
for p in (P1_SRC, P4_SRC, P2_SRC, RUN_LAYER_SRC, RUN_OPT_SRC):
    ps = str(p)
    if ps not in sys.path:
        sys.path.append(ps)

OUTPUT_ROOT = BASE_DIR / "RUN_LAYER_OPTIMISING" / "optimization_outputs"
CACHE_ROOT = BASE_DIR / "RUN_LAYER_OPTIMISING" / "data" / "cache"
SUMMARY_CSV = OUTPUT_ROOT / "fold_summary.csv"
# Local paths for leak-free normalisation
PREF4_PROCESSED = PROCESSED_DIR / "PRE_PILLARFOUR"
GSE_INPUT_PATH = PREF4_PROCESSED / "canonical_with_all_engines.parquet"
GSE_OUTPUT_PATH = PREF4_PROCESSED / "normalised_features.parquet"
GSE_PARAMS_PATH = PREF4_PROCESSED / "normalisation_params.json"

# ---------------------------------------------------------------------------
# Dynamic loader helpers
# ---------------------------------------------------------------------------

def _load_offline_fetch():
    mod_path = RUN_OPT_SRC / "offline_fetch.py"
    if not mod_path.exists():
        raise FileNotFoundError(f"offline_fetch.py not found at {mod_path}")
    spec = importlib.util.spec_from_file_location("offline_fetch", mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load offline_fetch from {mod_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    sys.modules["offline_fetch"] = module
    return module

def _load_module_from_path(mod_name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Module not found: {path}")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {mod_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module

offline_fetch = _load_offline_fetch()  # type: ignore

# ---------------------------------------------------------------------------
# Live pipeline stage imports
# ---------------------------------------------------------------------------

from PRE_PILLARFOURLIVE import (  # type: ignore
    build_canonical_raw_live,
    build_canonical_with_engines_live,
    global_scaling_engine_live,
)
from PILLAR_FOUR_REGIME_DETECTION_LIVE import (  # type: ignore
    build_raw_regime_features_live,
    deterministic_regime_engines_live,
    pillar4_part3_ml_regime_classifier_live,
    pillar4_part4_fusion_layer_live,
    pillar4_part5_stability_layer_live,
)
from PILLAR_ONE_ENTRY_MODEL_LIVE import (  # type: ignore
    build_pillar1_feature_matrix_live,
    build_pillar1_mini_engines_live,
    build_pillar1_labels_live,
    build_pillar1_multi_horizon_ml_live,
    build_pillar1_meta_features_live,
    build_pillar1_master_aggregator_live,
    build_pillar1_calibration_live,
    build_pillar1_decision_layer_live,
)
from PILLAR_TWO_EXIT_MODEL_LIVE import (  # type: ignore
    build_exit_raw_features_live,
    build_exit_mini_engines_live,
    build_exit_deterministic_live,
    build_exit_ml_models_live,
    build_exit_fusion_live,
    build_exit_final_targets_live,
)

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _call_maybe_kwargs(func, name: str, **kwargs: Any) -> None:
    logging.info("Running stage: %s", name)
    try:
        sig = inspect.signature(func)
        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
        if accepted:
            func(**accepted)
        else:
            func()
    except TypeError:
        func()

def _patch_offline_paths() -> None:
    offline_fetch.RAW_DIR = RAW_DIR
    offline_fetch.TRADES_DIR = RAW_DIR / "trades"
    offline_fetch.HTF_DIR = RAW_DIR / "htf_candles"

def _patch_p1_no_bfill() -> None:
    """
    Monkeypatch Pillar 1 load_base to avoid backfilling (forward-fill only).
    This keeps features leak-free without touching the RUN_LAYER codebase.
    """
    if getattr(_patch_p1_no_bfill, "_patched", False):
        return
    import importlib
    try:
        fm = importlib.import_module("pipeline.build_pillar1_feature_matrix")
    except Exception as exc:  # noqa: BLE001
        logging.warning("Could not patch Pillar1 load_base (module import failed): %s", exc)
        _patch_p1_no_bfill._patched = True
        return

    orig_log_mem = getattr(fm, "_log_mem", None)
    orig_downcast = getattr(fm, "_downcast_numeric", None)

    def load_base_no_bfill(limit_rows=None):
        missing = [p for p in [fm.NORM_PATH, fm.DET_PATH, fm.FUSED_PATH, fm.STABLE_PATH] if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing required upstream files: {missing}")

        logging.info("[P1 patch] Loading normalised features: %s", fm.NORM_PATH)
        df = pd.read_parquet(fm.NORM_PATH)
        if "open_time_ms" not in df.columns:
            raise ValueError("normalised_features.parquet missing open_time_ms")
        df = df.sort_values("open_time_ms").reset_index(drop=True)

        if limit_rows is not None and limit_rows > 0 and len(df) > limit_rows:
            logging.info("[P1 patch] Subsetting normalised features to last %d rows (from %d)", limit_rows, len(df))
            df = df.iloc[-limit_rows:].reset_index(drop=True)
        if orig_log_mem:
            orig_log_mem(df, "[P1 patch] After normalised load (possibly limited)")

        min_ts = df["open_time_ms"].iloc[0]

        logging.info("[P1 patch] Merging deterministic regime engines: %s", fm.DET_PATH)
        det = pd.read_parquet(fm.DET_PATH)
        if "open_time_ms" in det.columns:
            det = det.sort_values("open_time_ms")
            det = det[det["open_time_ms"] >= min_ts]
            df = df.merge(det, on="open_time_ms", how="left", suffixes=("", "_det"))
        else:
            df = fm._merge_by_index(df, det, tag="det")
        if orig_log_mem:
            orig_log_mem(df, "[P1 patch] After deterministic merge")

        logging.info("[P1 patch] Merging fused regimes: %s", fm.FUSED_PATH)
        fused = pd.read_parquet(fm.FUSED_PATH)
        if "open_time_ms" in fused.columns:
            fused = fused.sort_values("open_time_ms")
            fused = fused[fused["open_time_ms"] >= min_ts]
            df = df.merge(fused, on="open_time_ms", how="left", suffixes=("", "_fused"))
        else:
            logging.warning("[P1 patch] open_time_ms missing in fused labels; aligning by index (tail-matched)")
            df = fm._merge_by_index(df, fused, tag="fused")
        if orig_log_mem:
            orig_log_mem(df, "[P1 patch] After fused merge")

        logging.info("[P1 patch] Merging stable regimes: %s", fm.STABLE_PATH)
        stable = pd.read_parquet(fm.STABLE_PATH)
        if "open_time_ms" in stable.columns:
            stable = stable.sort_values("open_time_ms")
            stable = stable[stable["open_time_ms"] >= min_ts]
            df = df.merge(stable, on="open_time_ms", how="left", suffixes=("", "_stable"))
        else:
            logging.warning("[P1 patch] open_time_ms missing in stable labels; aligning by index (tail-matched)")
            df = fm._merge_by_index(df, stable, tag="stable")
        if orig_log_mem:
            orig_log_mem(df, "[P1 patch] After stable merge")

        # Forward-fill only (no backfill) to avoid forward leakage
        df = df.ffill()

        if orig_downcast:
            df = orig_downcast(df)
            if orig_log_mem:
                orig_log_mem(df, "[P1 patch] After downcast_numeric")
        return df

    fm.load_base = load_base_no_bfill  # type: ignore[assignment]
    _patch_p1_no_bfill._patched = True
    logging.info("Patched Pillar 1 load_base to remove backfill (forward-fill only).")

def _maybe_clear_fold_artifacts(fold_id: str) -> None:
    if not CLEAR_PREVIOUS_FOLD_ARTIFACTS:
        return
    for d in (CACHE_ROOT / f"fold_{fold_id}", OUTPUT_ROOT / f"fold_{fold_id}"):
        if d.exists():
            logging.warning("Deleting %s", d)
            shutil.rmtree(d, ignore_errors=True)

def _clear_run_layer_data_dirs() -> None:
    if not CLEAR_RUN_LAYER_DATA_EACH_FOLD:
        return
    for d in (RAW_DIR, PROCESSED_DIR):
        if d.exists():
            logging.warning("Clearing RUN_LAYER data dir: %s", d)
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

def _run_python_script(script_path: Path, name: str) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"{name} script not found: {script_path}")
    logging.info("Running script: %s (%s)", name, script_path)
    runpy.run_path(str(script_path), run_name="__main__")

def _run_post_raw_build_chain() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    dam_path = RUN_OPT_SRC / "delta_and_micro.py"
    dam = _load_module_from_path("delta_and_micro_opt", dam_path)

    dam.RAW_DIR = RAW_DIR
    dam.TRADES_DIR = RAW_DIR / "trades"
    dam.TRADES_IN = dam.TRADES_DIR / "trades_BTCUSDT.csv"
    dam.CANDLES_PATH = RAW_DIR / "BTC_CANDLES.csv"
    dam.DELTA_OUT = RAW_DIR / "delta.parquet"
    dam.MICRO_OUT = RAW_DIR / "microstructure.parquet"

    if not dam.TRADES_IN.exists():
        raise FileNotFoundError(f"Expected trades CSV missing: {dam.TRADES_IN}")
    if not dam.CANDLES_PATH.exists():
        raise FileNotFoundError(f"Expected candles CSV missing: {dam.CANDLES_PATH}")

    logging.info("Running delta_and_micro.main()")
    dam.main(trades_path=dam.TRADES_IN, candles_path=dam.CANDLES_PATH)

    _run_python_script(RUN_LAYER_SRC / "rollups.py", "rollups")
    _run_python_script(RUN_LAYER_SRC / "trade_aggregator.py", "trade_aggregator")


def _run_global_scaling(mode: str) -> None:
    """
    Leak-safe normalisation:
    - FIT: fit scaling params on the current window, write params + normalised features.
    - APPLY: load saved params, apply only (no refit).
    """
    from global_scaling_engine import (  # type: ignore
        GlobalScalingEngine,
        ensure_open_time_ms,
        collect_feature_list,
        DEFAULT_REGIME_COL,
    )

    if not GSE_INPUT_PATH.exists():
        raise FileNotFoundError(f"canonical_with_all_engines missing: {GSE_INPUT_PATH}")

    df = pl.read_parquet(GSE_INPUT_PATH)
    df = ensure_open_time_ms(df)
    # Ensure regime column type is consistent (cast to string) to avoid join dtype mismatches on apply
    if DEFAULT_REGIME_COL in df.columns:
        df = df.with_columns(pl.col(DEFAULT_REGIME_COL).cast(pl.Utf8))

    promote_to_float = [
        "agg_trades",
        "qty",
        "agg_buy_qty",
        "agg_sell_qty",
        "agg_delta",
        "trades",
    ]
    for col in promote_to_float:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float32))

    exclude = ["open_time_ms", "open_time", DEFAULT_REGIME_COL, "trades"]
    feature_list = collect_feature_list(df, exclude)

    engine = GlobalScalingEngine(regime_col=DEFAULT_REGIME_COL)
    if mode == "fit":
        logging.info("[GSE] Fitting normalisation params on window")
        norm_df, params = engine.fit(df, feature_list)
        GSE_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(GSE_PARAMS_PATH, "w") as f:
            json.dump(params, f, indent=2)
    else:
        if not GSE_PARAMS_PATH.exists():
            raise FileNotFoundError(
                f"Normalisation params missing: {GSE_PARAMS_PATH}. "
                "Run FIT first or rerun the fold to regenerate params."
            )
        logging.info("[GSE] Applying saved normalisation params (no refit)")
        with open(GSE_PARAMS_PATH) as f:
            params = json.load(f)
        # Params store regime keys as strings; enforce same dtype before apply
        if DEFAULT_REGIME_COL in df.columns:
            df = df.with_columns(pl.col(DEFAULT_REGIME_COL).cast(pl.Utf8))
        norm_df = engine.apply(df, feature_list, params)

    GSE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    norm_df.write_parquet(GSE_OUTPUT_PATH)
    logging.info("[GSE] Saved normalised features -> %s", GSE_OUTPUT_PATH)

def _plot_equity(curve: pd.DataFrame, out_path: Path, y_col: str) -> None:
    if curve.empty or not HAVE_MPL:
        return
    plt.figure(figsize=(10, 4))
    times = pd.to_datetime(curve["open_time_ms"], unit="ms", utc=True)
    plt.plot(times, curve[y_col], label=y_col)
    plt.xlabel("Time")
    plt.ylabel(y_col)
    plt.title(f"Equity curve ({y_col})")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()

def _add_equity_and_dd(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a strictly-positive equity curve + sane drawdowns.
    equity_curve: compounded capital starting at 1.0
    equity_realized: PnL in "R units" (equity_curve - 1)
    drawdown_realized: in [-1, 0]
    """
    if trades.empty:
        return trades

    r = trades["realized_r"].to_numpy(np.float64)

    # safety: prevent <= -100% returns from nuking equity (shouldn't happen, but guard anyway)
    r = np.clip(r, -0.999999, None)

    equity_curve = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve / peak) - 1.0

    trades["equity_curve"] = equity_curve
    trades["equity_realized"] = equity_curve - 1.0
    trades["drawdown_realized"] = dd
    return trades



# -------------------------
# RAM-safe parquet reads
# -------------------------

def _load_parquet(path: Path, columns: List[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if columns is None:
        return pd.read_parquet(path)
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        df = pd.read_parquet(path)
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name} missing columns: {missing}")
        return df[columns].copy()

def _merge_p1_p2() -> pd.DataFrame:
    p1_path = PROCESSED_DIR / "PILLAR_ONE_ENTRY_MODEL" / "features_pillar1_final.parquet"
    p2_path = PROCESSED_DIR / "PILLAR_TWO_EXIT_MODEL" / "exit_final_targets.parquet"

    p1_cols = ["open_time_ms", "close", "entry_prob_up", "entry_prob_down", "entry_confidence"]
    p2_cols = ["open_time_ms", "optimal_tp", "optimal_sl", "tp_first_probability"]

    p1 = _load_parquet(p1_path, columns=p1_cols)
    p2 = _load_parquet(p2_path, columns=p2_cols)

    merged = p1.merge(p2, on="open_time_ms", how="inner")
    return merged.sort_values("open_time_ms").reset_index(drop=True)

def _read_candles_from_raw() -> pd.DataFrame:
    path = RAW_DIR / "BTC_CANDLES.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing candles CSV: {path}")

    c = pd.read_csv(path)
    c["open_time"] = pd.to_datetime(c["open_time"], utc=True, errors="coerce")
    c["close_time"] = pd.to_datetime(c["close_time"], utc=True, errors="coerce")
    c = c.dropna(subset=["open_time", "close_time"]).copy()

    c["open_time_ms"] = (c["open_time"].astype("int64") // 1_000_000).astype("int64")
    c["close_time_ms"] = (c["close_time"].astype("int64") // 1_000_000).astype("int64")

    for col in ["open", "high", "low", "close"]:
        c[col] = pd.to_numeric(c[col], errors="coerce")
    c = c.dropna(subset=["open_time_ms", "high", "low", "close"]).copy()

    return c.sort_values("open_time_ms").reset_index(drop=True)[
        ["open_time_ms", "close_time_ms", "open", "high", "low", "close"]
    ]

# -------------------------
# Disk-safe processed snapshot
# -------------------------

def _disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(str(path))
    return usage.free / (1024**3)

def _dir_size_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    for p in root.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except FileNotFoundError:
            pass
    return total

def _match_any(name: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)

def _snapshot_processed(dst: Path) -> None:
    free_gb = _disk_free_gb(dst.parent)
    if free_gb < MIN_FREE_GB_REQUIRED:
        raise RuntimeError(f"Not enough free disk to proceed safely: free={free_gb:.2f}GB")

    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not PROCESSED_DIR.exists():
        raise FileNotFoundError(f"processed dir missing: {PROCESSED_DIR}")

    processed_gb = _dir_size_bytes(PROCESSED_DIR) / (1024**3)

    if processed_gb <= SNAPSHOT_FULL_MAX_GB:
        shutil.copytree(PROCESSED_DIR, dst, dirs_exist_ok=True)
        return

    include_globs = [
        "*.pkl", "*.joblib", "*.json", "*.yaml", "*.yml",
        "*.npz", "*.npy", "*.pt", "*.pth", "*.onnx",
        "*.bin", "*.cbm", "*.model", "*.txt"
    ]
    exclude_globs = ["*.parquet", "*.csv", "*.feather", "*.arrow", "*.zst", "*.gz", "*.zip"]

    max_total_bytes = int(SNAPSHOT_MAX_TOTAL_GB * (1024**3))
    max_file_bytes = int(SNAPSHOT_MAX_FILE_MB * (1024**2))

    copied = 0
    for src in PROCESSED_DIR.rglob("*"):
        if not src.is_file():
            continue

        name = src.name
        if _match_any(name, exclude_globs):
            continue
        if not _match_any(name, include_globs):
            continue

        try:
            sz = src.stat().st_size
        except FileNotFoundError:
            continue

        if sz > max_file_bytes:
            continue
        if copied + sz > max_total_bytes:
            break

        rel = src.relative_to(PROCESSED_DIR)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        copied += sz

    if copied == 0:
        raise RuntimeError(
            "Snapshot copied 0 bytes. Your pipeline stores fitted artifacts in a format we didn't include.\n"
            "Add the missing artifact file patterns/extensions to include_globs."
        )

def _restore_processed(src: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"processed snapshot missing: {src}")
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR, ignore_errors=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, PROCESSED_DIR, dirs_exist_ok=True)

# ---------------------------------------------------------------------------
# Pipeline runners (fit vs inference)
# ---------------------------------------------------------------------------

def _run_pipeline_fit() -> None:
    os.environ["PIPELINE_MODE"] = "train"
    os.environ["INFERENCE_ONLY"] = "0"
    _patch_p1_no_bfill()

    _call_maybe_kwargs(build_canonical_raw_live.main, "build_canonical_raw_live", mode="train")
    _call_maybe_kwargs(build_canonical_with_engines_live.main, "build_canonical_with_engines_live", mode="train")
    _run_global_scaling(mode="fit")

    _call_maybe_kwargs(build_raw_regime_features_live.main, "build_raw_regime_features_live", mode="train")
    _call_maybe_kwargs(deterministic_regime_engines_live.main, "deterministic_regime_engines_live", mode="train")
    _call_maybe_kwargs(pillar4_part3_ml_regime_classifier_live.main, "pillar4_part3_ml_regime_classifier_live",
                      mode="train", fit=True, train=True, retrain=True)
    _call_maybe_kwargs(pillar4_part4_fusion_layer_live.main, "pillar4_part4_fusion_layer_live", mode="train")
    _call_maybe_kwargs(pillar4_part5_stability_layer_live.main, "pillar4_part5_stability_layer_live", mode="train")

    _call_maybe_kwargs(build_pillar1_feature_matrix_live.main, "build_pillar1_feature_matrix_live", mode="train")
    _call_maybe_kwargs(build_pillar1_mini_engines_live.main, "build_pillar1_mini_engines_live", mode="train")
    _call_maybe_kwargs(build_pillar1_labels_live.main, "build_pillar1_labels_live", mode="train")
    _call_maybe_kwargs(build_pillar1_multi_horizon_ml_live.main, "build_pillar1_multi_horizon_ml_live",
                      mode="train", fit=True, train=True, retrain=True)
    _call_maybe_kwargs(build_pillar1_meta_features_live.main, "build_pillar1_meta_features_live", mode="train")
    _call_maybe_kwargs(build_pillar1_master_aggregator_live.main, "build_pillar1_master_aggregator_live", mode="train")
    _call_maybe_kwargs(build_pillar1_calibration_live.main, "build_pillar1_calibration_live",
                      mode="train", fit=True, train=True, retrain=True)
    _call_maybe_kwargs(build_pillar1_decision_layer_live.main, "build_pillar1_decision_layer_live", mode="train")

    _call_maybe_kwargs(build_exit_raw_features_live.main, "build_exit_raw_features_live", mode="train")
    _call_maybe_kwargs(build_exit_mini_engines_live.main, "build_exit_mini_engines_live", mode="train")
    _call_maybe_kwargs(build_exit_deterministic_live.main, "build_exit_deterministic_live", mode="train")
    _call_maybe_kwargs(build_exit_ml_models_live.main, "build_exit_ml_models_live",
                      mode="train", fit=True, train=True, retrain=True)
    _call_maybe_kwargs(build_exit_fusion_live.main, "build_exit_fusion_live", mode="train")
    _call_maybe_kwargs(build_exit_final_targets_live.main, "build_exit_final_targets_live", mode="train")

def _run_pipeline_infer() -> None:
    os.environ["PIPELINE_MODE"] = "test"
    os.environ["INFERENCE_ONLY"] = "1"
    _patch_p1_no_bfill()

    _call_maybe_kwargs(build_canonical_raw_live.main, "build_canonical_raw_live", mode="test")
    _call_maybe_kwargs(build_canonical_with_engines_live.main, "build_canonical_with_engines_live", mode="test")
    _run_global_scaling(mode="apply")

    _call_maybe_kwargs(build_raw_regime_features_live.main, "build_raw_regime_features_live", mode="test")
    _call_maybe_kwargs(deterministic_regime_engines_live.main, "deterministic_regime_engines_live", mode="test")
    _call_maybe_kwargs(pillar4_part3_ml_regime_classifier_live.main, "pillar4_part3_ml_regime_classifier_live",
                      mode="test", fit=False, train=False, retrain=False, inference_only=True)
    _call_maybe_kwargs(pillar4_part4_fusion_layer_live.main, "pillar4_part4_fusion_layer_live", mode="test")
    _call_maybe_kwargs(pillar4_part5_stability_layer_live.main, "pillar4_part5_stability_layer_live", mode="test")

    _call_maybe_kwargs(build_pillar1_feature_matrix_live.main, "build_pillar1_feature_matrix_live", mode="test")
    _call_maybe_kwargs(build_pillar1_mini_engines_live.main, "build_pillar1_mini_engines_live", mode="test")
    _call_maybe_kwargs(build_pillar1_labels_live.main, "build_pillar1_labels_live", mode="test")
    _call_maybe_kwargs(build_pillar1_multi_horizon_ml_live.main, "build_pillar1_multi_horizon_ml_live",
                      mode="test", fit=False, train=False, retrain=False, inference_only=True)
    _call_maybe_kwargs(build_pillar1_meta_features_live.main, "build_pillar1_meta_features_live", mode="test")
    _call_maybe_kwargs(build_pillar1_master_aggregator_live.main, "build_pillar1_master_aggregator_live", mode="test")
    _call_maybe_kwargs(build_pillar1_calibration_live.main, "build_pillar1_calibration_live",
                      mode="test", fit=False, train=False, retrain=False, inference_only=True)
    _call_maybe_kwargs(build_pillar1_decision_layer_live.main, "build_pillar1_decision_layer_live", mode="test")

    _call_maybe_kwargs(build_exit_raw_features_live.main, "build_exit_raw_features_live", mode="test")
    _call_maybe_kwargs(build_exit_mini_engines_live.main, "build_exit_mini_engines_live", mode="test")
    _call_maybe_kwargs(build_exit_deterministic_live.main, "build_exit_deterministic_live", mode="test")
    _call_maybe_kwargs(build_exit_ml_models_live.main, "build_exit_ml_models_live",
                      mode="test", fit=False, train=False, retrain=False, inference_only=True)
    _call_maybe_kwargs(build_exit_fusion_live.main, "build_exit_fusion_live", mode="test")
    _call_maybe_kwargs(build_exit_final_targets_live.main, "build_exit_final_targets_live", mode="test")

# ---------------------------------------------------------------------------
# Vectorized signal cache (speeds up EV prefilter massively)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalCache:
    open_time_ms: np.ndarray
    close: np.ndarray
    prob_up: np.ndarray
    prob_dn: np.ndarray
    conf: np.ndarray
    tp: np.ndarray
    sl: np.ndarray
    win_prob: np.ndarray

@dataclass(frozen=True)
class CandleCache:
    open_time_ms: np.ndarray
    close_time_ms: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    idx_map: Dict[int, int]

def _to_signal_cache(df: pd.DataFrame) -> SignalCache:
    req = ["open_time_ms","close","entry_prob_up","entry_prob_down","entry_confidence","optimal_tp","optimal_sl","tp_first_probability"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in merged df: {missing}")
    return SignalCache(
        open_time_ms=df["open_time_ms"].to_numpy(np.int64),
        close=df["close"].to_numpy(np.float64),
        prob_up=df["entry_prob_up"].to_numpy(np.float64),
        prob_dn=df["entry_prob_down"].to_numpy(np.float64),
        conf=df["entry_confidence"].to_numpy(np.float64),
        tp=df["optimal_tp"].to_numpy(np.float64),
        sl=df["optimal_sl"].to_numpy(np.float64),
        win_prob=df["tp_first_probability"].to_numpy(np.float64),
    )

def _to_candle_cache(candles: pd.DataFrame) -> CandleCache:
    c_open = candles["open_time_ms"].to_numpy(np.int64)
    idx_map = {int(t): i for i, t in enumerate(c_open)}
    return CandleCache(
        open_time_ms=c_open,
        close_time_ms=candles["close_time_ms"].to_numpy(np.int64),
        high=candles["high"].to_numpy(np.float64),
        low=candles["low"].to_numpy(np.float64),
        close=candles["close"].to_numpy(np.float64),
        idx_map=idx_map,
    )

# ---------------------------------------------------------------------------
# Param space + cost model
# ---------------------------------------------------------------------------

def _random_params() -> Dict[str, float]:
    # Wider + lower floors => more candidate signals, then realized scoring decides what survives.
    return {
        "long_threshold": random.uniform(0.45, 0.72),
        "short_threshold": random.uniform(0.28, 0.60),
        "conf_floor": random.uniform(0.00, 0.35),
        "tp_floor_pct": random.uniform(0.0005, 0.0050),
        "sl_floor_pct": random.uniform(0.00025, 0.0025),
        "sl_cap_pct": random.uniform(0.006, 0.020),
        "rr_min": random.uniform(1.02, 2.60),
        "tp_first_prob_min": random.uniform(0.05, 0.55),
    }

def _cost_terms() -> Tuple[float, float, float]:
    entry_fee = ENTRY_MAKER_SHARE * ENTRY_MAKER_FEE + ENTRY_TAKER_SHARE * ENTRY_TAKER_FEE
    exit_fee = EXIT_FEE
    slip = SLIPPAGE_BPS / 10000.0
    return entry_fee, exit_fee, slip

def _select_signals(cache: SignalCache, params: Dict[str, float]) -> pd.DataFrame:
    """
    Vectorized selection of candidate trades.
    Produces a DataFrame with the same fields your realized backtest needs.
    """
    entry_fee, exit_fee, slip = _cost_terms()

    pu = cache.prob_up
    pdn = cache.prob_dn
    conf = cache.conf
    entry = cache.close
    tp = cache.tp
    sl = cache.sl
    win = cache.win_prob
    t = cache.open_time_ms

    # side selection
    long_mask = (pu >= params["long_threshold"]) & (pu > pdn)
    short_mask = (pdn >= params["short_threshold"]) & (pdn > pu)
    side_mask = long_mask | short_mask

    if not side_mask.any():
        return pd.DataFrame()

    # confidence filter
    side_mask &= (conf >= params["conf_floor"])
    if not side_mask.any():
        return pd.DataFrame()

    # compute tp/sl distances based on side
    is_long = long_mask & side_mask
    is_short = short_mask & side_mask

    tp_dist = np.zeros_like(entry, dtype=np.float64)
    sl_dist = np.zeros_like(entry, dtype=np.float64)

    tp_dist[is_long] = (tp[is_long] / entry[is_long]) - 1.0
    sl_dist[is_long] = 1.0 - (sl[is_long] / entry[is_long])

    tp_dist[is_short] = (entry[is_short] / tp[is_short]) - 1.0
    sl_dist[is_short] = (sl[is_short] / entry[is_short]) - 1.0

    good = side_mask & (tp_dist > 0) & (sl_dist > 0)
    if not good.any():
        return pd.DataFrame()

    rr = np.zeros_like(entry, dtype=np.float64)
    rr[good] = tp_dist[good] / sl_dist[good]

    good &= (tp_dist >= params["tp_floor_pct"])
    good &= (sl_dist >= params["sl_floor_pct"])
    good &= (sl_dist <= params["sl_cap_pct"])
    good &= (rr >= params["rr_min"])
    good &= (win >= params["tp_first_prob_min"])

    if not good.any():
        return pd.DataFrame()

    # expected R (net)
    tp_net = tp_dist - entry_fee - exit_fee - 2 * slip
    sl_net = sl_dist + entry_fee + exit_fee + 2 * slip
    exp_r = win * tp_net - (1.0 - win) * sl_net

    side = np.where(is_long, "long", "short")

    out = pd.DataFrame({
        "open_time_ms": t[good].astype(np.int64),
        "side": side[good],
        "close": entry[good].astype(np.float64),
        "entry_prob_up": pu[good].astype(np.float64),
        "entry_prob_down": pdn[good].astype(np.float64),
        "entry_confidence": conf[good].astype(np.float64),
        "optimal_tp": tp[good].astype(np.float64),
        "optimal_sl": sl[good].astype(np.float64),
        "tp_first_probability": win[good].astype(np.float64),
        "tp_net": tp_net[good].astype(np.float64),
        "sl_net": sl_net[good].astype(np.float64),
        "exp_r": exp_r[good].astype(np.float64),
    }).sort_values("open_time_ms").reset_index(drop=True)

    return out

def _score_expected_prefilter(cache: SignalCache, params: Dict[str, float]) -> float:
    sig = _select_signals(cache, params)
    n = len(sig)
    if n < MIN_TRADES_PREFILTER:
        return -1e9
    r = sig["exp_r"].to_numpy(np.float64)
    mean_r = float(r.mean())
    shrunk = mean_r * n / (n + 100.0)
    # Soft push toward more trades even in prefilter (don't overdo it here)
    # Aim roughly around the midpoint of target range for a 30-day window (~900 trades),
    # but keep it gentle because EV is noisy.
    target_mid_month = 30.0 * ((TARGET_TRADES_PER_DAY_LOW + TARGET_TRADES_PER_DAY_HIGH) / 2.0)
    shortfall = max(0.0, target_mid_month - float(n)) / max(1.0, target_mid_month)
    return float(shrunk - 0.10 * shortfall)


def _window_days_from_candles(candles: CandleCache) -> float:
    """
    Estimate window length in days from candle timestamps.
    Robust to missing candles.
    """
    if candles.open_time_ms.size == 0:
        return 30.0
    start_ms = float(candles.open_time_ms[0])
    end_ms = float(candles.close_time_ms[-1])
    days = (end_ms - start_ms) / 86_400_000.0
    return float(max(1.0, days))


def _freq_penalty(trades_per_day: float) -> float:
    """
    Penalty is 0 inside [low, high].
    Under-trading is punished hard; over-trading is punished softly.
    """
    if trades_per_day < TARGET_TRADES_PER_DAY_LOW:
        frac = (TARGET_TRADES_PER_DAY_LOW - trades_per_day) / TARGET_TRADES_PER_DAY_LOW
        return FREQ_PENALTY_LOW * float(frac)
    if trades_per_day > TARGET_TRADES_PER_DAY_HIGH:
        frac = (trades_per_day - TARGET_TRADES_PER_DAY_HIGH) / TARGET_TRADES_PER_DAY_HIGH
        return FREQ_PENALTY_HIGH * float(frac)
    return 0.0

# ---------------------------------------------------------------------------
# Realized TP/SL backtest (path-based, first touch)
# ---------------------------------------------------------------------------

def _realized_backtest(cache: SignalCache, candles: CandleCache, params: Dict[str, float]) -> pd.DataFrame:
    sig_df = _select_signals(cache, params)
    if sig_df.empty:
        return pd.DataFrame()

    # no overlap handled here
    entry_fee, exit_fee, slip = _cost_terms()
    cost = entry_fee + exit_fee + 2 * slip

    c_open = candles.open_time_ms
    c_hi = candles.high
    c_lo = candles.low
    c_close = candles.close
    c_ct = candles.close_time_ms
    idx_map = candles.idx_map
    max_i = len(c_open) - 1

    out_rows: List[Dict[str, Any]] = []
    next_free_time = -1

    for _, s in sig_df.iterrows():
        t0 = int(s["open_time_ms"])
        if REALIZED_NO_OVERLAP and t0 < next_free_time:
            continue

        i0 = idx_map.get(t0)
        if i0 is None:
            i0 = int(np.searchsorted(c_open, t0, side="left"))
            if i0 > max_i:
                continue

        entry_price = float(s["close"])
        side = str(s["side"])
        tp = float(s["optimal_tp"])
        sl = float(s["optimal_sl"])

        start_i = min(i0 + 1, max_i)
        end_i = min(i0 + REALIZED_MAX_HOLD_MINUTES, max_i)

        exit_price = entry_price
        exit_time_ms = int(c_ct[i0])
        exit_reason = "timeout"

        for j in range(start_i, end_i + 1):
            hi = float(c_hi[j])
            lo = float(c_lo[j])

            if side == "long":
                hit_tp = hi >= tp
                hit_sl = lo <= sl
            else:
                hit_tp = lo <= tp
                hit_sl = hi >= sl

            if hit_tp and hit_sl:
                if REALIZED_SAME_BAR_RULE.upper() == "SL_FIRST":
                    hit_tp = False
                else:
                    hit_sl = False

            if hit_tp:
                exit_price = tp
                exit_time_ms = int(c_ct[j])
                exit_reason = "tp"
                break
            if hit_sl:
                exit_price = sl
                exit_time_ms = int(c_ct[j])
                exit_reason = "sl"
                break

            if j == end_i:
                exit_price = float(c_close[j])
                exit_time_ms = int(c_ct[j])
                exit_reason = "timeout"

        if side == "long":
            dist = (exit_price / entry_price) - 1.0
        else:
            dist = (entry_price / exit_price) - 1.0

        realized_r = dist - cost
        hold_minutes = max(0.0, (exit_time_ms - t0) / 60000.0)

        next_free_time = exit_time_ms if REALIZED_NO_OVERLAP else -1

        row_out = dict(s)
        row_out.update({
            "exit_time_ms": int(exit_time_ms),
            "exit_price": float(exit_price),
            "exit_reason": exit_reason,
            "hold_minutes": float(hold_minutes),
            "realized_r": float(realized_r),
        })
        out_rows.append(row_out)

    if not out_rows:
        return pd.DataFrame()

    trades = pd.DataFrame(out_rows).sort_values("open_time_ms").reset_index(drop=True)
    trades = _add_equity_and_dd(trades)
    return trades

def _metrics_realized(trades: pd.DataFrame, window_days: float) -> Dict[str, Any]:
    if trades is None or trades.empty:
        return {
            "score": -1e9, "score_adj": -1e9,
            "n": 0, "trades_per_day": 0.0,
            "max_dd": 0.0, "tuw": 0.0, "mean": 0.0, "sum": 0.0, "sharpe": 0.0,
        }

    r = trades["realized_r"].to_numpy(np.float64)
    n = int(r.size)

    mean_r = float(r.mean()) if n else 0.0
    sum_r = float(r.sum()) if n else 0.0
    shrunk = mean_r * n / (n + 100.0)

    # Ensure equity/dd exist (safe if trades came from _realized_backtest, but keep robust)
    if "equity_curve" not in trades.columns or "drawdown_realized" not in trades.columns:
        trades = _add_equity_and_dd(trades)

    dd = trades["drawdown_realized"].to_numpy(np.float64)  # in [-1, 0]
    max_dd = float((-dd).max()) if n else 0.0              # in [0, 1]
    tuw = float((dd < 0).mean()) if n else 0.0

    std = float(r.std(ddof=1)) if n > 1 else 0.0
    sharpe = float((mean_r / std) * math.sqrt(n)) if std > 0 else 0.0

    score = float(shrunk - 3.0 * max_dd - 0.4 * tuw)

    days = float(max(1.0, window_days))
    tpd = float(n) / days
    pen = _freq_penalty(tpd)
    score_adj = float(score - pen)

    return {
        "score": score,
        "score_adj": score_adj,
        "n": n,
        "trades_per_day": tpd,
        "max_dd": max_dd,
        "tuw": tuw,
        "mean": mean_r,
        "sum": sum_r,
        "sharpe": sharpe,
    }

# ---------------------------------------------------------------------------
# Fold schedule + bounds
# ---------------------------------------------------------------------------

def _window_schedule() -> List[datetime]:
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = now - timedelta(days=30 * MONTHS_BACK)

    out: List[datetime] = []
    cur = start

    min_span = FIT_DAYS + VAL_DAYS + TEST_DAYS + 1
    while cur + timedelta(days=min_span) <= now:
        out.append(cur)
        cur += timedelta(days=STEP_DAYS)
    return out

def _bounds(test_start: datetime) -> Tuple[datetime, datetime, datetime, datetime]:
    test_start = test_start.replace(second=0, microsecond=0, tzinfo=timezone.utc)
    val_start = test_start - timedelta(days=VAL_DAYS)
    fit_start = val_start - timedelta(days=FIT_DAYS)
    test_end = test_start + timedelta(days=TEST_DAYS)
    return fit_start, val_start, test_start, test_end

# ---------------------------------------------------------------------------
# Build one window’s outputs
# ---------------------------------------------------------------------------

def _build_window_outputs(start_dt: datetime, end_dt: datetime, mode: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    offline_fetch.fetch_raw_window(start_ms=start_ms, end_ms=end_ms)
    _run_post_raw_build_chain()

    if mode == "fit":
        _run_pipeline_fit()
    else:
        _run_pipeline_infer()

    merged = _merge_p1_p2()
    merged = merged[(merged["open_time_ms"] >= start_ms) & (merged["open_time_ms"] < end_ms)].copy()

    candles = _read_candles_from_raw()
    candles = candles[(candles["open_time_ms"] >= start_ms) & (candles["open_time_ms"] < end_ms)].copy()

    return (
        merged.sort_values("open_time_ms").reset_index(drop=True),
        candles.sort_values("open_time_ms").reset_index(drop=True),
    )

# ---------------------------------------------------------------------------
# Build fold artifacts (FIT -> snapshot -> VAL infer -> restore snapshot -> TEST infer)
# ---------------------------------------------------------------------------

def build_fold_cache(test_start: datetime, fold_id: str) -> Tuple[Path, Path, Path, Path, Dict[str, Any]]:
    _maybe_clear_fold_artifacts(fold_id)

    cache_dir = CACHE_ROOT / f"fold_{fold_id}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    val_merged_path = cache_dir / "merged_val.parquet"
    val_candles_path = cache_dir / "candles_val.parquet"
    test_merged_path = cache_dir / "merged_test.parquet"
    test_candles_path = cache_dir / "candles_test.parquet"
    meta_path = cache_dir / "fold_meta.json"
    processed_fit_snapshot = cache_dir / "processed_fit_snapshot"

    if all(p.exists() for p in (val_merged_path, val_candles_path, test_merged_path, test_candles_path, meta_path)):
        meta = json.loads(meta_path.read_text())
        return val_merged_path, val_candles_path, test_merged_path, test_candles_path, meta

    _clear_run_layer_data_dirs()
    _patch_offline_paths()

    fit_start, val_start, test_start_dt, test_end = _bounds(test_start)

    logging.info("Fold %s FIT  : %s -> %s", fold_id, fit_start, val_start)
    _build_window_outputs(fit_start, val_start, mode="fit")
    _snapshot_processed(processed_fit_snapshot)

    logging.info("Fold %s VAL  : %s -> %s", fold_id, val_start, test_start_dt)
    _restore_processed(processed_fit_snapshot)
    df_val, c_val = _build_window_outputs(val_start, test_start_dt, mode="infer")
    df_val.to_parquet(val_merged_path, index=False, compression="zstd")
    c_val.to_parquet(val_candles_path, index=False, compression="zstd")

    logging.info("Fold %s TEST : %s -> %s", fold_id, test_start_dt, test_end)
    _restore_processed(processed_fit_snapshot)
    df_test, c_test = _build_window_outputs(test_start_dt, test_end, mode="infer")
    df_test.to_parquet(test_merged_path, index=False, compression="zstd")
    c_test.to_parquet(test_candles_path, index=False, compression="zstd")

    meta = {
        "fold_id": fold_id,
        "fit_start_utc": fit_start.isoformat(),
        "val_start_utc": val_start.isoformat(),
        "test_start_utc": test_start_dt.isoformat(),
        "test_end_utc": test_end.isoformat(),
        "rows_val": int(len(df_val)),
        "rows_test": int(len(df_test)),
        "candles_val": int(len(c_val)),
        "candles_test": int(len(c_test)),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    # prevent snapshots accumulating per fold
    shutil.rmtree(processed_fit_snapshot, ignore_errors=True)

    return val_merged_path, val_candles_path, test_merged_path, test_candles_path, meta

# ---------------------------------------------------------------------------
# Summary table helpers
# ---------------------------------------------------------------------------

SUMMARY_COLUMNS = [
    "fold_id","fit_start_utc","val_start_utc","test_start_utc","test_end_utc",
    "val_n","val_mean_r","val_total_r","val_sharpe","val_max_dd",
    "test_n","test_mean_r","test_total_r","test_sharpe","test_max_dd",
    "status",
]

def _load_existing_summary() -> List[Dict[str, Any]]:
    if not SUMMARY_CSV.exists():
        return []
    try:
        df = pd.read_csv(SUMMARY_CSV)
        rows = df.to_dict("records")
        # basic sanitize
        out = []
        for r in rows:
            if "fold_id" in r and pd.notna(r["fold_id"]):
                r["fold_id"] = str(int(r["fold_id"])) if isinstance(r["fold_id"], (int, float)) else str(r["fold_id"])
                out.append(r)
        return out
    except Exception as e:
        logging.warning("Could not load existing summary CSV (%s). Starting fresh.", e)
        return []

def _write_summary(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    # enforce columns order if present
    for c in SUMMARY_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[SUMMARY_COLUMNS].sort_values("fold_id")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(SUMMARY_CSV, index=False)

def _print_live_summary(rows: List[Dict[str, Any]], done: int, total: int) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    for c in SUMMARY_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[SUMMARY_COLUMNS].sort_values("fold_id")

    # running edge checks (only for rows with OK)
    ok = df[df["status"] == "OK"].copy()
    if not ok.empty:
        ok["test_total_r"] = pd.to_numeric(ok["test_total_r"], errors="coerce")
        med = float(ok["test_total_r"].median())
        pos = float((ok["test_total_r"] > 0).mean()) * 100.0
        logging.info("=== Running summary: %d/%d folds done | median(test_total_r)=%.6f | positive_folds=%.1f%% ===",
                     done, total, med, pos)

    # print table (optionally truncated)
    if len(df) > PRINT_TOP_N_FOLDS_IN_TABLE:
        view = df.tail(PRINT_TOP_N_FOLDS_IN_TABLE)
    else:
        view = df

    # pretty-ish formatting
    with pd.option_context("display.max_rows", 500, "display.max_columns", 200, "display.width", 220):
        print("\n----- FOLD SUMMARY (live) -----")
        print(view.to_string(index=False))
        print("----- END SUMMARY -----\n")

def _final_edge_report(rows: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        logging.warning("No folds completed.")
        return
    df = df[df["status"] == "OK"].copy()
    if df.empty:
        logging.warning("No OK folds.")
        return

    df["test_total_r"] = pd.to_numeric(df["test_total_r"], errors="coerce")
    df["test_n"] = pd.to_numeric(df["test_n"], errors="coerce")
    df["test_sharpe"] = pd.to_numeric(df["test_sharpe"], errors="coerce")
    df["test_max_dd"] = pd.to_numeric(df["test_max_dd"], errors="coerce")

    med_total = float(df["test_total_r"].median())
    mean_total = float(df["test_total_r"].mean())
    pos_frac = float((df["test_total_r"] > 0).mean())
    pos_pct = 100.0 * pos_frac

    logging.info("=== FINAL EDGE CHECKS ===")
    logging.info("Folds OK: %d", len(df))
    logging.info("Median test_total_r: %.6f", med_total)
    logging.info("Mean   test_total_r: %.6f", mean_total)
    logging.info("%% folds positive: %.1f%%", pos_pct)
    logging.info("Median test_sharpe: %.4f", float(df["test_sharpe"].median()))
    logging.info("Worst  test_max_dd: %.4f", float(df["test_max_dd"].max()))
    logging.info("Total  test_total_r (sum): %.6f", float(df["test_total_r"].sum()))
    logging.info("Total  test_n (sum): %d", int(df["test_n"].sum()))

    passed_median = med_total > 0
    passed_pos = pos_frac > 0.60

    logging.info("PASS median(test_total_r)>0  ? %s", "YES" if passed_median else "NO")
    logging.info("PASS positive_folds>60%%     ? %s", "YES" if passed_pos else "NO")

# ---------------------------------------------------------------------------
# Evaluate fold: tune on VAL, report on TEST
# ---------------------------------------------------------------------------

def evaluate_fold(test_start: datetime, fold_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    val_m_path, val_c_path, test_m_path, test_c_path, meta = build_fold_cache(test_start, fold_id)

    df_val = pd.read_parquet(val_m_path).sort_values("open_time_ms").reset_index(drop=True)
    c_val_df = pd.read_parquet(val_c_path).sort_values("open_time_ms").reset_index(drop=True)

    df_test = pd.read_parquet(test_m_path).sort_values("open_time_ms").reset_index(drop=True)
    c_test_df = pd.read_parquet(test_c_path).sort_values("open_time_ms").reset_index(drop=True)

    out_dir = OUTPUT_ROOT / f"fold_{fold_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    val_cache = _to_signal_cache(df_val)
    test_cache = _to_signal_cache(df_test)
    c_val = _to_candle_cache(c_val_df)
    c_test = _to_candle_cache(c_test_df)
    val_days = _window_days_from_candles(c_val)
    test_days = _window_days_from_candles(c_test)

    # random candidates (EV prefilter)
    candidates: List[Tuple[float, Dict[str, float]]] = []
    for _ in range(N_RANDOM_SAMPLES):
        p = _random_params()
        candidates.append((_score_expected_prefilter(val_cache, p), p))

    candidates.sort(key=lambda x: x[0], reverse=True)
    candidates_top = candidates[:TOPK_REALIZED]

    best_params: Dict[str, float] = {}
    best_val_trades = pd.DataFrame()
    best_val_metrics: Dict[str, Any] = {"score": -1e9, "score_adj": -1e9, "n": 0}

    summary_rows: List[Dict[str, Any]] = []

    for score_exp, p in candidates_top:
        trades = _realized_backtest(val_cache, c_val, p)
        m = _metrics_realized(trades, window_days=val_days)
        summary_rows.append(
            {
                "score_exp": float(score_exp),
                "score_val": float(m["score"]),
                "score_adj": float(m["score_adj"]),
                "n_val": int(m["n"]),
                "trades_per_day": float(m["trades_per_day"]),
                "max_dd_val": float(m["max_dd"]),
                "tuw_val": float(m["tuw"]),
                "mean_val": float(m["mean"]),
                "sum_val": float(m["sum"]),
                "sharpe_val": float(m["sharpe"]),
                **p,
            }
        )
        if int(m["n"]) < MIN_TRADES_REALIZED:
            continue
        # Select using frequency-adjusted score (score_adj)
        if float(m["score_adj"]) > float(best_val_metrics["score_adj"]):
            best_params = p
            best_val_trades = trades
            best_val_metrics = m

    if not best_params:
        logging.warning("Fold %s: no VAL candidate met MIN_TRADES=%d. Falling back to top EV candidate.",
                        fold_id, MIN_TRADES_REALIZED)
        best_params = candidates_top[0][1]
        best_val_trades = _realized_backtest(val_cache, c_val, best_params)
        best_val_metrics = _metrics_realized(best_val_trades, window_days=val_days)

    if summary_rows:
        pd.DataFrame(summary_rows).sort_values("score_val", ascending=False).to_csv(
            out_dir / "candidates_topk_val.csv", index=False
        )

    if not best_val_trades.empty:
        best_val_trades.to_parquet(out_dir / "taken_trades_val.parquet", index=False, compression="zstd")
        _plot_equity(best_val_trades[["open_time_ms","equity_realized"]],
                    out_dir / "equity_val.png", "equity_realized")
    else:
        pd.DataFrame().to_parquet(out_dir / "taken_trades_val.parquet", index=False)

    test_trades = _realized_backtest(test_cache, c_test, best_params)
    test_metrics = _metrics_realized(test_trades, window_days=test_days)

    if not test_trades.empty:
        test_trades.to_parquet(out_dir / "taken_trades_test.parquet", index=False, compression="zstd")
        _plot_equity(test_trades[["open_time_ms","equity_realized"]],
                    out_dir / "equity_test.png", "equity_realized")
    else:
        pd.DataFrame().to_parquet(out_dir / "taken_trades_test.parquet", index=False)

    (out_dir / "best_params.json").write_text(json.dumps(best_params, indent=2))

    best_metrics_json = {
        **meta,
        "selected_on": "VAL",
        "best_params": best_params,
        "val": {
            "score": float(best_val_metrics["score"]),
            "score_adj": float(best_val_metrics.get("score_adj", best_val_metrics["score"])),
            "n": int(best_val_metrics["n"]),
            "trades_per_day": float(best_val_metrics.get("trades_per_day", 0.0)),
            "max_dd": float(best_val_metrics["max_dd"]),
            "tuw": float(best_val_metrics["tuw"]),
            "mean_r": float(best_val_metrics["mean"]),
            "total_r": float(best_val_metrics["sum"]),
            "sharpe": float(best_val_metrics["sharpe"]),
        },
        "test": {
            "score": float(test_metrics["score"]),
            "score_adj": float(test_metrics.get("score_adj", test_metrics["score"])),
            "n": int(test_metrics["n"]),
            "trades_per_day": float(test_metrics.get("trades_per_day", 0.0)),
            "max_dd": float(test_metrics["max_dd"]),
            "tuw": float(test_metrics["tuw"]),
            "mean_r": float(test_metrics["mean"]),
            "total_r": float(test_metrics["sum"]),
            "sharpe": float(test_metrics["sharpe"]),
        },
        "frequency_target": {
            "trades_per_day_low": float(TARGET_TRADES_PER_DAY_LOW),
            "trades_per_day_high": float(TARGET_TRADES_PER_DAY_HIGH),
            "penalty_low": float(FREQ_PENALTY_LOW),
            "penalty_high": float(FREQ_PENALTY_HIGH),
        },
        "realized_settings": {
            "max_hold_minutes": int(REALIZED_MAX_HOLD_MINUTES),
            "no_overlap": bool(REALIZED_NO_OVERLAP),
            "same_bar_rule": str(REALIZED_SAME_BAR_RULE),
            "costs": {
                "slippage_bps": float(SLIPPAGE_BPS),
                "entry_fee": float(ENTRY_MAKER_SHARE * ENTRY_MAKER_FEE + ENTRY_TAKER_SHARE * ENTRY_TAKER_FEE),
                "exit_fee": float(EXIT_FEE),
            },
        },
    }
    (out_dir / "best_metrics.json").write_text(json.dumps(best_metrics_json, indent=2))

    logging.info(
        "Fold %s done | VAL score=%.4f adj=%.4f (n=%d, tpd=%.2f, dd=%.4f) | TEST score=%.4f adj=%.4f (n=%d, tpd=%.2f, dd=%.4f)",
        fold_id,
        float(best_val_metrics["score"]), float(best_val_metrics.get("score_adj", best_val_metrics["score"])),
        int(best_val_metrics["n"]), float(best_val_metrics.get("trades_per_day", 0.0)), float(best_val_metrics["max_dd"]),
        float(test_metrics["score"]), float(test_metrics.get("score_adj", test_metrics["score"])),
        int(test_metrics["n"]), float(test_metrics.get("trades_per_day", 0.0)), float(test_metrics["max_dd"]),
    )

    # This is what gets appended to fold_summary.csv
    fold_row = {
        "fold_id": meta.get("fold_id", fold_id),
        "fit_start_utc": meta.get("fit_start_utc"),
        "val_start_utc": meta.get("val_start_utc"),
        "test_start_utc": meta.get("test_start_utc"),
        "test_end_utc": meta.get("test_end_utc"),
        "val_n": int(best_val_metrics["n"]),
        "val_mean_r": float(best_val_metrics["mean"]),
        "val_total_r": float(best_val_metrics["sum"]),
        "val_sharpe": float(best_val_metrics["sharpe"]),
        "val_max_dd": float(best_val_metrics["max_dd"]),
        "test_n": int(test_metrics["n"]),
        "test_mean_r": float(test_metrics["mean"]),
        "test_total_r": float(test_metrics["sum"]),
        "test_sharpe": float(test_metrics["sharpe"]),
        "test_max_dd": float(test_metrics["max_dd"]),
        "status": "OK",
    }

    return fold_row, best_metrics_json

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)

    folds = _window_schedule()
    if not folds:
        logging.error("No folds generated; check date logic.")
        return

    # resume existing summary if present
    summary_rows: List[Dict[str, Any]] = _load_existing_summary()
    done_ids = {str(r.get("fold_id")) for r in summary_rows if r.get("status") == "OK"}

    total = len(folds)

    for idx, test_start in enumerate(folds, start=1):
        fold_id = test_start.strftime("%Y%m%d")

        if fold_id in done_ids:
            logging.info("=== Fold %d/%d (fold_id=%s) already in summary -> skipping ===", idx, total, fold_id)
            _print_live_summary(summary_rows, done=len(done_ids), total=total)
            continue

        logging.info("=== Fold %d/%d (test_start=%s, fold_id=%s) ===", idx, total, test_start, fold_id)

        try:
            fold_row, _ = evaluate_fold(test_start, fold_id)
            summary_rows.append(fold_row)
        except Exception as exc:
            logging.exception("Fold %s failed: %s", fold_id, exc)
            # still append a row so the table shows the failure
            fit_start, val_start, test_start_dt, test_end = _bounds(test_start)
            summary_rows.append({
                "fold_id": fold_id,
                "fit_start_utc": fit_start.isoformat(),
                "val_start_utc": val_start.isoformat(),
                "test_start_utc": test_start_dt.isoformat(),
                "test_end_utc": test_end.isoformat(),
                "val_n": 0, "val_mean_r": 0.0, "val_total_r": 0.0, "val_sharpe": 0.0, "val_max_dd": 0.0,
                "test_n": 0, "test_mean_r": 0.0, "test_total_r": 0.0, "test_sharpe": 0.0, "test_max_dd": 0.0,
                "status": "FAILED",
            })

        # write + print after each fold
        _write_summary(summary_rows)
        done_ok = sum(1 for r in summary_rows if r.get("status") == "OK")
        _print_live_summary(summary_rows, done=done_ok, total=total)

    _write_summary(summary_rows)
    _final_edge_report(summary_rows)
    logging.info("Done.")

if __name__ == "__main__":
    main()
