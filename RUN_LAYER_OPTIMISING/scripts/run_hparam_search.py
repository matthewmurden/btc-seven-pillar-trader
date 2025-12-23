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
- Within-fold safety: snapshot FIT processed/ and restore it before VAL and before TEST
  so VAL can’t accidentally mutate fitted artifacts used for TEST.
- Realized scoring: path-based barrier backtest using 1m candles (TP/SL first touch),
  conservative same-bar rule.

Mac safety:
- Column-pruned parquet reads for merge.
- Snapshot processed/ is MINIMAL when processed/ is large (no huge copytree).
- Snapshot is deleted after VAL/TEST caches are written.

Outputs per fold:
  RUN_LAYER_OPTIMISING/optimization_outputs/fold_YYYYMMDD/
    best_params.json
    best_metrics.json
    candidates_topk_val.csv
    taken_trades_val.parquet
    taken_trades_test.parquet
    equity_val.png
    equity_test.png

NEW: Rolling summary table (updated after every fold):
  RUN_LAYER_OPTIMISING/optimization_outputs/summary_folds.csv
  Printed to console each fold.

IMPORTANT FIX:
- Drawdown math is now stable: equity starts at 1.0 (not 0.0), so DD is meaningful.
"""

from __future__ import annotations

import argparse
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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

import numpy as np
import pandas as pd
import importlib.util

# ---------------------------------------------------------------------------
# Walk-forward split config (defaults; can be overridden via CLI)
# ---------------------------------------------------------------------------

FIT_DAYS = 30 * 6      # ~6 months
VAL_DAYS = 30          # 1 month
TEST_DAYS = 30         # 1 month
STEP_DAYS = 30         # step forward 1 month
MONTHS_BACK = 18       # how far back we start generating folds

# ---------------------------------------------------------------------------
# Optimiser config (defaults; can be overridden via CLI)
# ---------------------------------------------------------------------------

N_RANDOM_SAMPLES = 800
MIN_TRADES = 30
TOPK_REALIZED = 60

# Realized backtest settings
REALIZED_MAX_HOLD_MINUTES = 240
REALIZED_NO_OVERLAP = True
REALIZED_SAME_BAR_RULE = "SL_FIRST"  # conservative when TP & SL hit same candle

# Execution costs
SLIPPAGE_BPS = 3.0
ENTRY_MAKER_FEE = 0.0002
ENTRY_TAKER_FEE = 0.0004
ENTRY_MAKER_SHARE = 0.20
ENTRY_TAKER_SHARE = 0.80
EXIT_FEE = 0.0004

# Stable equity base for DD (CRITICAL FIX)
EQUITY_BASE = 1.0

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
CLEAR_RUN_LAYER_DATA_EACH_FOLD = True  # leak #2 fix (prevents cross-fold reuse)

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
SUMMARY_CSV = OUTPUT_ROOT / "summary_folds.csv"

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
# Helpers (pipeline + IO)
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

def _plot_equity(curve: pd.DataFrame, out_path: Path, y_col: str) -> None:
    if curve is None or curve.empty or not HAVE_MPL:
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
    """
    Memory-critical: only load the columns needed for tuning + realized backtest.
    """
    p1_path = PROCESSED_DIR / "PILLAR_ONE_ENTRY_MODEL" / "features_pillar1_final.parquet"
    p2_path = PROCESSED_DIR / "PILLAR_TWO_EXIT_MODEL" / "exit_final_targets.parquet"

    p1_cols = ["open_time_ms", "close", "entry_prob_up", "entry_prob_down", "entry_confidence"]
    p2_cols = ["open_time_ms", "optimal_tp", "optimal_sl", "tp_first_probability"]

    p1 = _load_parquet(p1_path, columns=p1_cols)
    p2 = _load_parquet(p2_path, columns=p2_cols)

    merged = p1.merge(p2, on="open_time_ms", how="inner", suffixes=("", "_p2"))
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
    """
    Mac-safe snapshot:
    - Full copy only if processed/ is small.
    - Otherwise copy only likely artifact files (models/scalers/calibration) and cap size.
    """
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
            "Run once and copy the first missing-file path; then add its extension/pattern to include_globs."
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

    _call_maybe_kwargs(build_canonical_raw_live.main, "build_canonical_raw_live", mode="train")
    _call_maybe_kwargs(build_canonical_with_engines_live.main, "build_canonical_with_engines_live", mode="train")
    _call_maybe_kwargs(global_scaling_engine_live.main, "global_scaling_engine_live", mode="train", fit=True, train=True, retrain=True)

    _call_maybe_kwargs(build_raw_regime_features_live.main, "build_raw_regime_features_live", mode="train")
    _call_maybe_kwargs(deterministic_regime_engines_live.main, "deterministic_regime_engines_live", mode="train")
    _call_maybe_kwargs(pillar4_part3_ml_regime_classifier_live.main, "pillar4_part3_ml_regime_classifier_live", mode="train", fit=True, train=True, retrain=True)
    _call_maybe_kwargs(pillar4_part4_fusion_layer_live.main, "pillar4_part4_fusion_layer_live", mode="train")
    _call_maybe_kwargs(pillar4_part5_stability_layer_live.main, "pillar4_part5_stability_layer_live", mode="train")

    _call_maybe_kwargs(build_pillar1_feature_matrix_live.main, "build_pillar1_feature_matrix_live", mode="train")
    _call_maybe_kwargs(build_pillar1_mini_engines_live.main, "build_pillar1_mini_engines_live", mode="train")
    _call_maybe_kwargs(build_pillar1_labels_live.main, "build_pillar1_labels_live", mode="train")
    _call_maybe_kwargs(build_pillar1_multi_horizon_ml_live.main, "build_pillar1_multi_horizon_ml_live", mode="train", fit=True, train=True, retrain=True)
    _call_maybe_kwargs(build_pillar1_meta_features_live.main, "build_pillar1_meta_features_live", mode="train")
    _call_maybe_kwargs(build_pillar1_master_aggregator_live.main, "build_pillar1_master_aggregator_live", mode="train")
    _call_maybe_kwargs(build_pillar1_calibration_live.main, "build_pillar1_calibration_live", mode="train", fit=True, train=True, retrain=True)
    _call_maybe_kwargs(build_pillar1_decision_layer_live.main, "build_pillar1_decision_layer_live", mode="train")

    _call_maybe_kwargs(build_exit_raw_features_live.main, "build_exit_raw_features_live", mode="train")
    _call_maybe_kwargs(build_exit_mini_engines_live.main, "build_exit_mini_engines_live", mode="train")
    _call_maybe_kwargs(build_exit_deterministic_live.main, "build_exit_deterministic_live", mode="train")
    _call_maybe_kwargs(build_exit_ml_models_live.main, "build_exit_ml_models_live", mode="train", fit=True, train=True, retrain=True)
    _call_maybe_kwargs(build_exit_fusion_live.main, "build_exit_fusion_live", mode="train")
    _call_maybe_kwargs(build_exit_final_targets_live.main, "build_exit_final_targets_live", mode="train")

def _run_pipeline_infer() -> None:
    os.environ["PIPELINE_MODE"] = "test"
    os.environ["INFERENCE_ONLY"] = "1"

    _call_maybe_kwargs(build_canonical_raw_live.main, "build_canonical_raw_live", mode="test")
    _call_maybe_kwargs(build_canonical_with_engines_live.main, "build_canonical_with_engines_live", mode="test")
    _call_maybe_kwargs(global_scaling_engine_live.main, "global_scaling_engine_live", mode="test", fit=False, train=False, retrain=False, inference_only=True)

    _call_maybe_kwargs(build_raw_regime_features_live.main, "build_raw_regime_features_live", mode="test")
    _call_maybe_kwargs(deterministic_regime_engines_live.main, "deterministic_regime_engines_live", mode="test")
    _call_maybe_kwargs(pillar4_part3_ml_regime_classifier_live.main, "pillar4_part3_ml_regime_classifier_live", mode="test", fit=False, train=False, retrain=False, inference_only=True)
    _call_maybe_kwargs(pillar4_part4_fusion_layer_live.main, "pillar4_part4_fusion_layer_live", mode="test")
    _call_maybe_kwargs(pillar4_part5_stability_layer_live.main, "pillar4_part5_stability_layer_live", mode="test")

    _call_maybe_kwargs(build_pillar1_feature_matrix_live.main, "build_pillar1_feature_matrix_live", mode="test")
    _call_maybe_kwargs(build_pillar1_mini_engines_live.main, "build_pillar1_mini_engines_live", mode="test")
    _call_maybe_kwargs(build_pillar1_labels_live.main, "build_pillar1_labels_live", mode="test")
    _call_maybe_kwargs(build_pillar1_multi_horizon_ml_live.main, "build_pillar1_multi_horizon_ml_live", mode="test", fit=False, train=False, retrain=False, inference_only=True)
    _call_maybe_kwargs(build_pillar1_meta_features_live.main, "build_pillar1_meta_features_live", mode="test")
    _call_maybe_kwargs(build_pillar1_master_aggregator_live.main, "build_pillar1_master_aggregator_live", mode="test")
    _call_maybe_kwargs(build_pillar1_calibration_live.main, "build_pillar1_calibration_live", mode="test", fit=False, train=False, retrain=False, inference_only=True)
    _call_maybe_kwargs(build_pillar1_decision_layer_live.main, "build_pillar1_decision_layer_live", mode="test")

    _call_maybe_kwargs(build_exit_raw_features_live.main, "build_exit_raw_features_live", mode="test")
    _call_maybe_kwargs(build_exit_mini_engines_live.main, "build_exit_mini_engines_live", mode="test")
    _call_maybe_kwargs(build_exit_deterministic_live.main, "build_exit_deterministic_live", mode="test")
    _call_maybe_kwargs(build_exit_ml_models_live.main, "build_exit_ml_models_live", mode="test", fit=False, train=False, retrain=False, inference_only=True)
    _call_maybe_kwargs(build_exit_fusion_live.main, "build_exit_fusion_live", mode="test")
    _call_maybe_kwargs(build_exit_final_targets_live.main, "build_exit_final_targets_live", mode="test")

# ---------------------------------------------------------------------------
# Param space + trade selection
# ---------------------------------------------------------------------------

def _random_params() -> Dict[str, float]:
    return {
        "long_threshold": random.uniform(0.52, 0.75),
        "short_threshold": random.uniform(0.25, 0.48),
        "conf_floor": random.uniform(0.05, 0.55),
        "tp_floor_pct": random.uniform(0.001, 0.006),
        "sl_floor_pct": random.uniform(0.0006, 0.003),
        "sl_cap_pct": random.uniform(0.003, 0.012),
        "rr_min": random.uniform(1.1, 3.5),
        "tp_first_prob_min": random.uniform(0.15, 0.65),
    }

def _cost_terms() -> Tuple[float, float, float]:
    entry_fee = ENTRY_MAKER_SHARE * ENTRY_MAKER_FEE + ENTRY_TAKER_SHARE * ENTRY_TAKER_FEE
    exit_fee = EXIT_FEE
    slip = SLIPPAGE_BPS / 10000.0
    return entry_fee, exit_fee, slip

def _iter_taken_signals(df: pd.DataFrame, params: Dict[str, float]) -> List[Dict[str, Any]]:
    req_cols = [
        "open_time_ms", "close", "entry_prob_up", "entry_prob_down", "entry_confidence",
        "optimal_tp", "optimal_sl", "tp_first_probability",
    ]
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in merged dataframe.")

    entry_fee, exit_fee, slip = _cost_terms()
    rows: List[Dict[str, Any]] = []

    # itertuples is faster + lower overhead than iterrows
    for row in df.itertuples(index=False):
        prob_up = float(getattr(row, "entry_prob_up"))
        prob_dn = float(getattr(row, "entry_prob_down"))
        conf = float(getattr(row, "entry_confidence", 0.0))
        entry_price = float(getattr(row, "close"))
        tp = float(getattr(row, "optimal_tp"))
        sl = float(getattr(row, "optimal_sl"))
        win_prob = float(getattr(row, "tp_first_probability"))
        t0 = int(getattr(row, "open_time_ms"))

        side: Optional[str] = None
        if prob_up >= params["long_threshold"] and prob_up > prob_dn:
            side = "long"
        elif prob_dn >= params["short_threshold"] and prob_dn > prob_up:
            side = "short"

        if side is None or conf < params["conf_floor"]:
            continue

        if side == "long":
            tp_dist = (tp / entry_price) - 1.0
            sl_dist = 1.0 - (sl / entry_price)
        else:
            tp_dist = (entry_price / tp) - 1.0
            sl_dist = (sl / entry_price) - 1.0

        if tp_dist <= 0.0 or sl_dist <= 0.0:
            continue

        rr = tp_dist / sl_dist
        if tp_dist < params["tp_floor_pct"]:
            continue
        if sl_dist < params["sl_floor_pct"]:
            continue
        if sl_dist > params["sl_cap_pct"]:
            continue
        if rr < params["rr_min"]:
            continue
        if win_prob < params["tp_first_prob_min"]:
            continue

        tp_net = tp_dist - entry_fee - exit_fee - 2.0 * slip
        sl_net = sl_dist + entry_fee + exit_fee + 2.0 * slip
        exp_r = win_prob * tp_net - (1.0 - win_prob) * sl_net

        rows.append(
            {
                "open_time_ms": t0,
                "side": side,
                "close": float(entry_price),
                "entry_prob_up": float(prob_up),
                "entry_prob_down": float(prob_dn),
                "entry_confidence": float(conf),
                "optimal_tp": float(tp),
                "optimal_sl": float(sl),
                "tp_first_probability": float(win_prob),
                "tp_net": float(tp_net),
                "sl_net": float(sl_net),
                "exp_r": float(exp_r),
            }
        )

    return rows

def _score_expected_prefilter(df: pd.DataFrame, params: Dict[str, float]) -> float:
    rows = _iter_taken_signals(df, params)
    if len(rows) < MIN_TRADES:
        return -1e9
    r = np.array([x["exp_r"] for x in rows], dtype=float)
    mean_r = float(r.mean())
    n = float(len(r))
    shrunk = mean_r * n / (n + 100.0)
    return float(shrunk)

# ---------------------------------------------------------------------------
# Realized TP/SL backtest (the one that matters)
# ---------------------------------------------------------------------------

def _realized_backtest(df_signals: pd.DataFrame, candles: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    signals = _iter_taken_signals(df_signals, params)
    if not signals:
        return pd.DataFrame()

    sig_df = pd.DataFrame(signals).sort_values("open_time_ms").reset_index(drop=True)

    # Pre-extract candle arrays for speed (avoid .iloc in inner loops)
    c_open = candles["open_time_ms"].to_numpy(dtype=np.int64)
    c_close_time = candles["close_time_ms"].to_numpy(dtype=np.int64)
    c_high = candles["high"].to_numpy(dtype=np.float64)
    c_low = candles["low"].to_numpy(dtype=np.float64)
    c_close = candles["close"].to_numpy(dtype=np.float64)

    # Map open_time_ms -> index (fast) + fallback searchsorted
    idx_map = {int(t): i for i, t in enumerate(c_open)}
    max_i = len(c_open) - 1

    entry_fee, exit_fee, slip = _cost_terms()
    cost = entry_fee + exit_fee + 2.0 * slip

    out_rows: List[Dict[str, Any]] = []
    next_free_time = -1

    for s in sig_df.itertuples(index=False):
        t0 = int(s.open_time_ms)
        if REALIZED_NO_OVERLAP and t0 < next_free_time:
            continue

        i0 = idx_map.get(t0)
        if i0 is None:
            i0 = int(np.searchsorted(c_open, t0, side="left"))
            if i0 > max_i:
                continue

        entry_price = float(s.close)
        side = str(s.side)
        tp = float(s.optimal_tp)
        sl = float(s.optimal_sl)

        start_i = min(i0 + 1, max_i)
        end_i = min(i0 + REALIZED_MAX_HOLD_MINUTES, max_i)

        exit_price = entry_price
        exit_time_ms = int(c_close_time[i0])
        exit_reason = "timeout"

        for j in range(start_i, end_i + 1):
            hi = float(c_high[j])
            lo = float(c_low[j])

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
                exit_time_ms = int(c_close_time[j])
                exit_reason = "tp"
                break
            if hit_sl:
                exit_price = sl
                exit_time_ms = int(c_close_time[j])
                exit_reason = "sl"
                break

            if j == end_i:
                exit_price = float(c_close[j])
                exit_time_ms = int(c_close_time[j])
                exit_reason = "timeout"

        if side == "long":
            dist = (exit_price / entry_price) - 1.0
        else:
            dist = (entry_price / exit_price) - 1.0

        realized_r = float(dist - cost)
        hold_minutes = float(max(0.0, (exit_time_ms - t0) / 60000.0))

        if REALIZED_NO_OVERLAP:
            next_free_time = exit_time_ms

        out_rows.append(
            {
                "open_time_ms": int(s.open_time_ms),
                "side": str(s.side),
                "close": float(s.close),
                "entry_prob_up": float(s.entry_prob_up),
                "entry_prob_down": float(s.entry_prob_down),
                "entry_confidence": float(s.entry_confidence),
                "optimal_tp": float(s.optimal_tp),
                "optimal_sl": float(s.optimal_sl),
                "tp_first_probability": float(s.tp_first_probability),
                "tp_net": float(s.tp_net),
                "sl_net": float(s.sl_net),
                "exp_r": float(s.exp_r),
                "exit_time_ms": int(exit_time_ms),
                "exit_price": float(exit_price),
                "exit_reason": exit_reason,
                "hold_minutes": hold_minutes,
                "realized_r": realized_r,
            }
        )

    if not out_rows:
        return pd.DataFrame()

    trades = pd.DataFrame(out_rows).sort_values("open_time_ms").reset_index(drop=True)

    # Stable equity & drawdown (CRITICAL FIX)
    trades["equity_realized"] = EQUITY_BASE + trades["realized_r"].cumsum()
    peak = trades["equity_realized"].cummax()
    dd = (trades["equity_realized"] - peak) / peak
    trades["drawdown_realized"] = dd.fillna(0.0)

    return trades

def _metrics_realized(trades: pd.DataFrame) -> Dict[str, Any]:
    if trades is None or trades.empty:
        return {"score": -1e9, "n": 0, "max_dd": 0.0, "tuw": 0.0, "mean": 0.0, "sum": 0.0, "sharpe": 0.0}

    n = int(len(trades))
    equity = trades["equity_realized"].astype(float)
    peak = equity.cummax()

    dd = (equity - peak) / peak
    max_dd = float(abs(dd.min(skipna=True))) if len(dd) else 0.0
    tuw = float((equity < peak).mean()) if len(peak) else 0.0

    r = trades["realized_r"].to_numpy(dtype=float)
    mean_r = float(r.mean())
    sum_r = float(r.sum())

    shrunk = mean_r * n / (n + 100.0)

    std = float(r.std(ddof=1)) if n > 1 else 0.0
    sharpe = float((mean_r / std) * math.sqrt(n)) if std > 0 else 0.0

    score = float(shrunk - 3.0 * max_dd - 0.4 * tuw)
    return {"score": score, "n": n, "max_dd": max_dd, "tuw": tuw, "mean": mean_r, "sum": sum_r, "sharpe": sharpe}

# ---------------------------------------------------------------------------
# Fold schedule + bounds
# ---------------------------------------------------------------------------

def _window_schedule(months_back: int) -> List[datetime]:
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = now - timedelta(days=30 * months_back)

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

    try:
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
        return val_merged_path, val_candles_path, test_merged_path, test_candles_path, meta

    finally:
        # Prevent snapshots accumulating per fold (even if something explodes)
        shutil.rmtree(processed_fit_snapshot, ignore_errors=True)

# ---------------------------------------------------------------------------
# Rolling summary table
# ---------------------------------------------------------------------------

def _append_and_print_summary(row: Dict[str, Any]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if SUMMARY_CSV.exists():
        try:
            df = pd.read_csv(SUMMARY_CSV)
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # Canonical sort by fold_id if possible
    if "fold_id" in df.columns:
        try:
            df = df.sort_values("fold_id").reset_index(drop=True)
        except Exception:
            pass

    df.to_csv(SUMMARY_CSV, index=False)

    # Compute the "do we have edge" quick checks on TEST
    df_ok = df[(df.get("status", "OK") == "OK")].copy() if "status" in df.columns else df.copy()
    if "test_total_r" in df_ok.columns:
        test_total = pd.to_numeric(df_ok["test_total_r"], errors="coerce")
        valid = test_total.dropna()
        if len(valid) > 0:
            pct_pos = float((valid > 0).mean() * 100.0)
            median_total = float(valid.median())
        else:
            pct_pos, median_total = float("nan"), float("nan")
    else:
        pct_pos, median_total = float("nan"), float("nan")

    # Print a compact table
    cols = [
        "fold_id",
        "val_total_r", "val_mean_r", "val_sharpe", "val_max_dd", "val_n",
        "test_total_r", "test_mean_r", "test_sharpe", "test_max_dd", "test_n",
        "status",
    ]
    show_cols = [c for c in cols if c in df.columns]

    tail = df[show_cols].tail(12).copy()

    # numeric formatting
    for c in tail.columns:
        if c in {"fold_id", "status"}:
            continue
        tail[c] = pd.to_numeric(tail[c], errors="coerce")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    logging.info("---- Rolling fold summary (last %d rows) ----", len(tail))
    logging.info("\n%s", tail.to_string(index=False))

    if not math.isnan(pct_pos) and not math.isnan(median_total):
        logging.info(
            "EDGE CHECK (TEST): median_total_r=%.6f | %%folds_positive=%.1f%% | folds_count=%d | saved=%s",
            median_total, pct_pos, int(len(pd.to_numeric(df_ok.get("test_total_r", pd.Series()), errors="coerce").dropna())),
            str(SUMMARY_CSV),
        )
    else:
        logging.info("EDGE CHECK (TEST): insufficient data yet | saved=%s", str(SUMMARY_CSV))

# ---------------------------------------------------------------------------
# Evaluate fold: tune on VAL, report on TEST
# ---------------------------------------------------------------------------

def evaluate_fold(test_start: datetime, fold_id: str) -> Dict[str, Any]:
    val_m_path, val_c_path, test_m_path, test_c_path, meta = build_fold_cache(test_start, fold_id)

    df_val = pd.read_parquet(val_m_path).sort_values("open_time_ms").reset_index(drop=True)
    c_val = pd.read_parquet(val_c_path).sort_values("open_time_ms").reset_index(drop=True)

    df_test = pd.read_parquet(test_m_path).sort_values("open_time_ms").reset_index(drop=True)
    c_test = pd.read_parquet(test_c_path).sort_values("open_time_ms").reset_index(drop=True)

    out_dir = OUTPUT_ROOT / f"fold_{fold_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates: List[Tuple[float, Dict[str, float]]] = []
    for _ in range(N_RANDOM_SAMPLES):
        p = _random_params()
        candidates.append((_score_expected_prefilter(df_val, p), p))

    candidates.sort(key=lambda x: x[0], reverse=True)
    candidates_top = candidates[:TOPK_REALIZED]

    best_params: Dict[str, float] = {}
    best_val_trades = pd.DataFrame()
    best_val_metrics: Dict[str, Any] = {"score": -1e9, "n": 0}

    summary_rows: List[Dict[str, Any]] = []

    for score_exp, p in candidates_top:
        trades = _realized_backtest(df_val, c_val, p)
        m = _metrics_realized(trades)
        summary_rows.append(
            {
                "score_exp": float(score_exp),
                "score_val": float(m["score"]),
                "n_val": int(m["n"]),
                "max_dd_val": float(m["max_dd"]),
                "tuw_val": float(m["tuw"]),
                "mean_val": float(m["mean"]),
                "sum_val": float(m["sum"]),
                "sharpe_val": float(m["sharpe"]),
                **p,
            }
        )
        if int(m["n"]) < MIN_TRADES:
            continue
        if float(m["score"]) > float(best_val_metrics["score"]):
            best_params = p
            best_val_trades = trades
            best_val_metrics = m

    if not best_params:
        logging.warning(
            "Fold %s: no VAL candidate met MIN_TRADES=%d. Falling back to top EV candidate.",
            fold_id, MIN_TRADES
        )
        best_params = candidates_top[0][1]
        best_val_trades = _realized_backtest(df_val, c_val, best_params)
        best_val_metrics = _metrics_realized(best_val_trades)

    if summary_rows:
        pd.DataFrame(summary_rows).sort_values("score_val", ascending=False).to_csv(
            out_dir / "candidates_topk_val.csv", index=False
        )

    if not best_val_trades.empty:
        best_val_trades.to_parquet(out_dir / "taken_trades_val.parquet", index=False, compression="zstd")
        _plot_equity(best_val_trades[["open_time_ms", "equity_realized"]], out_dir / "equity_val.png", "equity_realized")
    else:
        pd.DataFrame().to_parquet(out_dir / "taken_trades_val.parquet", index=False)

    test_trades = _realized_backtest(df_test, c_test, best_params)
    test_metrics = _metrics_realized(test_trades)

    if not test_trades.empty:
        test_trades.to_parquet(out_dir / "taken_trades_test.parquet", index=False, compression="zstd")
        _plot_equity(test_trades[["open_time_ms", "equity_realized"]], out_dir / "equity_test.png", "equity_realized")
    else:
        pd.DataFrame().to_parquet(out_dir / "taken_trades_test.parquet", index=False)

    (out_dir / "best_params.json").write_text(json.dumps(best_params, indent=2))

    best_metrics_json = {
        **meta,
        "selected_on": "VAL",
        "best_params": best_params,
        "equity_base": float(EQUITY_BASE),
        "drawdown_definition": "(equity - peak) / peak, with equity starting at equity_base",
        "val": {
            "score": float(best_val_metrics["score"]),
            "n": int(best_val_metrics["n"]),
            "max_dd": float(best_val_metrics["max_dd"]),
            "tuw": float(best_val_metrics["tuw"]),
            "mean_r": float(best_val_metrics["mean"]),
            "total_r": float(best_val_metrics["sum"]),
            "sharpe": float(best_val_metrics["sharpe"]),
        },
        "test": {
            "score": float(test_metrics["score"]),
            "n": int(test_metrics["n"]),
            "max_dd": float(test_metrics["max_dd"]),
            "tuw": float(test_metrics["tuw"]),
            "mean_r": float(test_metrics["mean"]),
            "total_r": float(test_metrics["sum"]),
            "sharpe": float(test_metrics["sharpe"]),
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
        "Fold %s done | VAL score=%.4f (n=%d, dd=%.4f) | TEST score=%.4f (n=%d, dd=%.4f)",
        fold_id,
        float(best_val_metrics["score"]), int(best_val_metrics["n"]), float(best_val_metrics["max_dd"]),
        float(test_metrics["score"]), int(test_metrics["n"]), float(test_metrics["max_dd"]),
    )

    # Return compact summary row for rolling table
    summary = {
        "fold_id": fold_id,
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
    return summary

# ---------------------------------------------------------------------------
# CLI + Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leak-safe walk-forward param tuning with rolling summary table.")
    p.add_argument("--months-back", type=int, default=MONTHS_BACK, help="How many months back to start folds (default: 18).")
    p.add_argument("--random-samples", type=int, default=N_RANDOM_SAMPLES, help="Random param samples per fold (default: 800).")
    p.add_argument("--topk-realized", type=int, default=TOPK_REALIZED, help="Top-K EV candidates to realize-score (default: 60).")
    p.add_argument("--min-trades", type=int, default=MIN_TRADES, help="Min trades constraint (default: 30).")
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "1337")), help="RNG seed (default: env SEED or 1337).")
    return p.parse_args()

def main() -> None:
    global MONTHS_BACK, N_RANDOM_SAMPLES, TOPK_REALIZED, MIN_TRADES

    args = _parse_args()

    MONTHS_BACK = int(args.months_back)
    N_RANDOM_SAMPLES = int(args.random_samples)
    TOPK_REALIZED = int(args.topk_realized)
    MIN_TRADES = int(args.min_trades)

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    # Start fresh summary CSV each run (optional). If you want append-across-runs, comment this out.
    if SUMMARY_CSV.exists():
        SUMMARY_CSV.unlink(missing_ok=True)  # type: ignore[arg-type]

    folds = _window_schedule(MONTHS_BACK)
    if not folds:
        logging.error("No folds generated; check date logic.")
        return

    logging.info(
        "Config | months_back=%d | random_samples=%d | topk_realized=%d | min_trades=%d | seed=%d",
        MONTHS_BACK, N_RANDOM_SAMPLES, TOPK_REALIZED, MIN_TRADES, int(args.seed)
    )

    for idx, test_start in enumerate(folds, start=1):
        fold_id = test_start.strftime("%Y%m%d")
        logging.info("=== Fold %d/%d (test_start=%s) ===", idx, len(folds), test_start)

        try:
            row = evaluate_fold(test_start, fold_id)
        except Exception as exc:
            logging.exception("Fold %s failed: %s", fold_id, exc)
            row = {
                "fold_id": fold_id,
                "fit_start_utc": None,
                "val_start_utc": None,
                "test_start_utc": None,
                "test_end_utc": None,
                "val_n": None,
                "val_mean_r": None,
                "val_total_r": None,
                "val_sharpe": None,
                "val_max_dd": None,
                "test_n": None,
                "test_mean_r": None,
                "test_total_r": None,
                "test_sharpe": None,
                "test_max_dd": None,
                "status": f"FAIL: {type(exc).__name__}",
            }

        _append_and_print_summary(row)

    logging.info("Done. Final summary saved to: %s", str(SUMMARY_CSV))

if __name__ == "__main__":
    main()