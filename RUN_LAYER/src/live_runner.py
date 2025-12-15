from __future__ import annotations

"""
Live orchestrator to run the full multi-pillar pipeline on a rolling window.

Key settings:
  LOOKBACK_MINUTES: number of latest 1m bars to keep (default: 1 day = 1440)
  MAX_RETRIES: how many times to retry a failing stage before aborting

Workflow per loop:
  1) Preload data (delta/micro, trades, candles, rollups)
  2) Pre-Pillar4 live: canonical_raw_live -> with_engines_live -> global_scaling_live
  3) Trim normalised features to LOOKBACK_MINUTES
  4) Pillar 1 live stack (feature matrix -> mini engines -> labels -> ML -> meta -> agg -> calibration -> decision)
  5) Pillar 2 live stack (raw -> labels -> mini engines -> ML -> fusion -> final targets)
  6) Pillar 3 live sizing
  7) Pillar 5 execution plans
  8) Pillar 6 safety filter
  9) Log latest rows from Pillar 5 + Pillar 6 into a combined parquet

The loop runs indefinitely, waiting for the next minute before repeating.
"""

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Any, List

import pandas as pd

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
LOOKBACK_MINUTES = 5  # easy to change (1 day default)
MAX_RETRIES = 3
SLEEP_SECONDS_BETWEEN_LOOPS = 5

# Paths
THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parents[2]
RUN_PROCESSED = BASE_DIR / "RUN_LAYER" / "data" / "processed"
RUN_RAW = BASE_DIR / "RUN_LAYER" / "data" / "raw"

LOG_PATH = RUN_PROCESSED / "live_combined_log.parquet"

# ---------------------------------------------------------------------
# Import wiring for all live modules
# ---------------------------------------------------------------------
SRC_DIRS: List[Path] = [
    THIS_DIR,
    THIS_DIR / "PRE_PILLARFOURLIVE",
    THIS_DIR / "PILLAR_FOUR_REGIME_DETECTION_LIVE",
    THIS_DIR / "PILLAR_ONE_ENTRY_MODEL_LIVE",
    THIS_DIR / "PILLAR_TWO_EXIT_MODEL_LIVE",
    THIS_DIR / "PILLAR_THREE_RISK_MODEL_LIVE",
    THIS_DIR / "PILLAR_FIVE_EXECUTION_ENGINE_LIVE",
    THIS_DIR / "PILLAR_SIX_SAFETY_LAYER_LIVE",
]

for p in SRC_DIRS:
    ps = str(p)
    if ps not in sys.path:
        sys.path.append(ps)

# Pillar wrappers
import delta_and_micro
import fetch_candles
import rollups
import trade_aggregator

from PRE_PILLARFOURLIVE import (
    build_canonical_raw_live,
    build_canonical_with_engines_live,
    global_scaling_engine_live,
)
from PILLAR_FOUR_REGIME_DETECTION_LIVE import (
    build_raw_regime_features_live,
    deterministic_regime_engines_live,
    pillar4_part3_ml_regime_classifier_live,
    pillar4_part4_fusion_layer_live,
    pillar4_part5_stability_layer_live,
    pillar_integration_live,
)
from PILLAR_ONE_ENTRY_MODEL_LIVE import (
    build_pillar1_feature_matrix_live,
    build_pillar1_mini_engines_live,
    build_pillar1_labels_live,
    build_pillar1_multi_horizon_ml_live,
    build_pillar1_meta_features_live,
    build_pillar1_master_aggregator_live,
    build_pillar1_calibration_live,
    build_pillar1_decision_layer_live,
)
from PILLAR_TWO_EXIT_MODEL_LIVE import (
    build_exit_raw_features_live,
    build_exit_labels_live,
    build_exit_mini_engines_live,
    build_exit_ml_models_live,
    build_exit_fusion_live,
    build_exit_final_targets_live,
)
from PILLAR_THREE_RISK_MODEL_LIVE import build_position_sizing_live
from PILLAR_FIVE_EXECUTION_ENGINE_LIVE import build_execution_plans_live


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _run_with_retries(func: Callable[[], Any], name: str) -> None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            func()
            return
        except Exception as exc:  # noqa: BLE001
            logging.exception("Stage %s failed (attempt %d/%d): %s", name, attempt, MAX_RETRIES, exc)
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2)


def _trim_parquet(path: Path, time_col: str, max_rows: int) -> None:
    if not path.exists():
        return
    df = pd.read_parquet(path)
    if len(df) > max_rows:
        if time_col in df.columns:
            df = df.sort_values(time_col).iloc[-max_rows:]
        else:
            df = df.iloc[-max_rows:]
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logging.info("Trimmed %s to last %d rows", path.name, max_rows)


def _keep_latest_only(path: Path, time_col: str) -> None:
    """
    Replace the parquet at `path` with only its latest row (by time_col if present).
    Intended for post-Pre-P4 so downstream pillars only process the freshest bar.
    """
    if not path.exists():
        return
    df = pd.read_parquet(path)
    if df.empty:
        return
    if time_col in df.columns:
        df = df.sort_values(time_col).iloc[[-1]]
    else:
        df = df.iloc[[-1]]
    df.to_parquet(path, index=False)
    logging.info("Reduced %s to latest row only", path.name)


def _log_latest(p5_path: Path, p6_path: Path) -> None:
    if not p5_path.exists() or not p6_path.exists():
        logging.warning("Cannot log latest outputs; missing pillar5 or pillar6 parquet.")
        return
    p5 = pd.read_parquet(p5_path).sort_values("open_time_ms") if p5_path.exists() else pd.DataFrame()
    p6 = pd.read_parquet(p6_path).sort_values("open_time_ms") if p6_path.exists() else pd.DataFrame()
    if p5.empty or p6.empty:
        logging.warning("Pillar5 or Pillar6 outputs empty; skipping log append.")
        return
    latest_p5 = p5.iloc[[-1]]
    latest_p6 = p6.iloc[[-1]]
    if "open_time_ms" in latest_p5.columns and "open_time_ms" in latest_p6.columns:
        combined = latest_p5.merge(latest_p6, on="open_time_ms", suffixes=("_p5", "_p6"))
    else:
        combined = pd.concat([latest_p5.add_suffix("_p5"), latest_p6.add_suffix("_p6")], axis=1)
    if LOG_PATH.exists():
        existing = pd.read_parquet(LOG_PATH)
        combined = pd.concat([existing, combined], ignore_index=True)
    combined = combined.iloc[-LOOKBACK_MINUTES:]
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(LOG_PATH, index=False)
    logging.info("Appended latest Pillar5/6 row to log (%s)", LOG_PATH)


def _sleep_until_next_minute(last_ts_ms: int | None) -> None:
    if last_ts_ms is None:
        time.sleep(SLEEP_SECONDS_BETWEEN_LOOPS)
        return
    last_dt = datetime.fromtimestamp(last_ts_ms / 1000, tz=timezone.utc)
    next_minute = (last_dt.replace(second=0, microsecond=0) + pd.Timedelta(minutes=1)).to_pydatetime()
    wait = max(1.0, (next_minute - datetime.now(timezone.utc)).total_seconds())
    time.sleep(wait)


# ---------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------
def run_preload() -> None:
    # Adjust lookback settings for fetchers
    delta_and_micro.LOOKBACK_DAYS = LOOKBACK_MINUTES / 1440.0
    fetch_candles.LOOKBACK_DAYS = LOOKBACK_MINUTES / 1440.0

    _run_with_retries(delta_and_micro.main, "delta_and_micro")
    _run_with_retries(trade_aggregator.main, "trade_aggregator")
    _run_with_retries(fetch_candles.main, "fetch_candles")
    _run_with_retries(rollups.main, "rollups")


def run_pre_p4() -> None:
    _run_with_retries(build_canonical_raw_live.main, "build_canonical_raw_live")
    _run_with_retries(build_canonical_with_engines_live.main, "build_canonical_with_engines_live")
    _run_with_retries(global_scaling_engine_live.main, "global_scaling_engine_live")
    norm_path = RUN_PROCESSED / "PRE_PILLARFOUR" / "normalised_features.parquet"
    # Keep full history for correctness during computation, then keep only the freshest bar for downstream pillars.
    _trim_parquet(norm_path, "open_time_ms", LOOKBACK_MINUTES)
    _keep_latest_only(norm_path, "open_time_ms")


def run_p4_live() -> None:
    _run_with_retries(build_raw_regime_features_live.main, "build_raw_regime_features_live")
    _run_with_retries(deterministic_regime_engines_live.main, "deterministic_regime_engines_live")
    _run_with_retries(pillar4_part3_ml_regime_classifier_live.main, "pillar4_part3_ml_regime_classifier_live")
    _run_with_retries(pillar4_part4_fusion_layer_live.main, "pillar4_part4_fusion_layer_live")
    _run_with_retries(pillar4_part5_stability_layer_live.main, "pillar4_part5_stability_layer_live")
    _run_with_retries(pillar_integration_live.main, "pillar_integration_live")


def run_p1_live() -> None:
    _run_with_retries(build_pillar1_feature_matrix_live.main, "build_pillar1_feature_matrix_live")
    _run_with_retries(build_pillar1_mini_engines_live.main, "build_pillar1_mini_engines_live")
    _run_with_retries(build_pillar1_labels_live.main, "build_pillar1_labels_live")
    _run_with_retries(build_pillar1_multi_horizon_ml_live.main, "build_pillar1_multi_horizon_ml_live")
    _run_with_retries(build_pillar1_meta_features_live.main, "build_pillar1_meta_features_live")
    _run_with_retries(build_pillar1_master_aggregator_live.main, "build_pillar1_master_aggregator_live")
    _run_with_retries(build_pillar1_calibration_live.main, "build_pillar1_calibration_live")
    _run_with_retries(build_pillar1_decision_layer_live.main, "build_pillar1_decision_layer_live")


def run_p2_live() -> None:
    _run_with_retries(build_exit_raw_features_live.main, "build_exit_raw_features_live")
    _run_with_retries(build_exit_labels_live.main, "build_exit_labels_live")
    _run_with_retries(build_exit_mini_engines_live.main, "build_exit_mini_engines_live")
    _run_with_retries(build_exit_ml_models_live.main, "build_exit_ml_models_live")
    _run_with_retries(build_exit_fusion_live.main, "build_exit_fusion_live")
    _run_with_retries(build_exit_final_targets_live.main, "build_exit_final_targets_live")


def run_p3_p5_live() -> None:
    _run_with_retries(build_position_sizing_live.main, "build_position_sizing_live")
    _run_with_retries(build_execution_plans_live.main, "build_execution_plans_live")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    last_ts_ms: int | None = None

    while True:
        loop_start = time.time()
        logging.info("=== Live pipeline loop start ===")

        run_preload()
        run_pre_p4()
        run_p4_live()
        run_p1_live()
        run_p2_live()
        run_p3_p5_live()

        # Trim key parquets to rolling window
        _trim_parquet(RUN_PROCESSED / "PILLAR_ONE_ENTRY_MODEL" / "features_pillar1_final.parquet", "open_time_ms", LOOKBACK_MINUTES)
        _trim_parquet(RUN_PROCESSED / "PILLAR_TWO_EXIT_MODEL" / "exit_final_targets.parquet", "open_time_ms", LOOKBACK_MINUTES)
        _trim_parquet(RUN_PROCESSED / "PILLAR_TWO_EXIT_MODEL" / "exit_with_sizing.parquet", "open_time_ms", LOOKBACK_MINUTES)
        _trim_parquet(RUN_PROCESSED / "PILLAR_FIVE_EXECUTION_ENGINE" / "execution_plans.parquet", "open_time_ms", LOOKBACK_MINUTES)
        # Log latest rows (P5 only for now)
        _log_latest(
            RUN_PROCESSED / "PILLAR_FIVE_EXECUTION_ENGINE" / "execution_plans.parquet",
            RUN_PROCESSED / "PILLAR_FIVE_EXECUTION_ENGINE" / "execution_plans.parquet",
        )

        # Update last_ts_ms
        norm_path = RUN_PROCESSED / "PRE_PILLARFOUR" / "normalised_features.parquet"
        if norm_path.exists():
            df_norm = pd.read_parquet(norm_path)
            if "open_time_ms" in df_norm.columns and len(df_norm) > 0:
                last_ts_ms = int(df_norm["open_time_ms"].max())

        loop_dur = time.time() - loop_start
        logging.info("=== Live pipeline loop complete in %.1fs ===", loop_dur)
        _sleep_until_next_minute(last_ts_ms)


if __name__ == "__main__":
    main()
