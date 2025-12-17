from __future__ import annotations

"""
Live orchestrator to run the full multi-pillar pipeline on a rolling window.

Key settings:
  LOOKBACK_MINUTES: number of latest 1m bars to keep (e.g. 1440 for 1 day)
  MAX_RETRIES: how many times to retry a failing stage before aborting

High-level behaviour:

- First loop ("bootstrap"):
    • Fetch a full LOOKBACK_MINUTES window of raw data (trades, candles, rollups).
    • Run Pre-Pillar4 live (canonical_raw -> engines -> global scaling) on the full
      window, producing normalised_features.parquet with LOOKBACK_MINUTES rows.

- Subsequent loops:
    • Determine the latest *closed* minute (now - 1m).
    • Try to "top up" the raw layer to that time:
        - If your fetcher modules expose `topup_to(end_ms, lookback_minutes)`,
          use those to fetch only missing minutes and maintain a rolling window.
        - Otherwise, fall back to the original full-window fetch mode.
    • Rebuild canonical_raw + engines + global scaling on the full local window.
      (This is cheap compared to network IO and avoids 1-row normalisation issues.)
    • Trim normalised_features.parquet to LOOKBACK_MINUTES rows.

- For each loop, after Pre-Pillar4 is up to date:
    • Run Pillar 4 live (regime detection).
    • Run Pillar 1 live (entry model).
    • Run Pillar 2 live (exit model).
    • Run Pillar 3 + Pillar 5 live on the latest row only (sizing + execution).
    • Trim key downstream parquet files to LOOKBACK_MINUTES rows.
    • Append the latest Pillar 5 row to a rolling log parquet.

The loop runs indefinitely, aligned to minute boundaries.
"""

import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, Any, List

import pandas as pd

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
# Change to 1440 for 1-day lookback in production
LOOKBACK_MINUTES = 1440

MAX_RETRIES = 3
MAX_PRE_P4_TOPUPS = 5  # safety cap when trying to catch up to latest closed minute

# Paths
THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parents[1]  # TRADINGBOTPROJECT root

RUN_PROCESSED = BASE_DIR / "RUN_LAYER" / "data" / "processed"
RUN_RAW = BASE_DIR / "RUN_LAYER" / "data" / "raw"

PRE_P4_DIR = RUN_PROCESSED / "PRE_PILLARFOUR"
NORM_PATH = PRE_P4_DIR / "normalised_features.parquet"

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
]

for p in SRC_DIRS:
    ps = str(p)
    if ps not in sys.path:
        sys.path.append(ps)

# Raw / rollups
import delta_and_micro
import fetch_candles
import rollups
import trade_aggregator

# Pre-Pillar 4 live
from PRE_PILLARFOURLIVE import (
    build_canonical_raw_live,
    build_canonical_with_engines_live,
    global_scaling_engine_live,
)

# Pillar 4 live
from PILLAR_FOUR_REGIME_DETECTION_LIVE import (
    build_raw_regime_features_live,
    deterministic_regime_engines_live,
    pillar4_part3_ml_regime_classifier_live,
    pillar4_part4_fusion_layer_live,
    pillar4_part5_stability_layer_live,
    pillar_integration_live,
)

# Pillar 1 live
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

# Pillar 2 live
from PILLAR_TWO_EXIT_MODEL_LIVE import (
    build_exit_raw_features_live,
    build_exit_labels_live,
    build_exit_mini_engines_live,
    build_exit_deterministic_live,
    build_exit_ml_models_live,
    build_exit_fusion_live,
    build_exit_final_targets_live,
)

# Pillar 3 live
from PILLAR_THREE_RISK_MODEL_LIVE import build_position_sizing_live

# Pillar 5 live
from PILLAR_FIVE_EXECUTION_ENGINE_LIVE import build_execution_plans_live

# NEW: Trade execution wiring (Binance demo futures)
from the_executor import execute_latest_plan

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _run_with_retries(func: Callable[[], Any], name: str) -> None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            func()
            return
        except Exception as exc:  # noqa: BLE001
            logging.exception(
                "Stage %s failed (attempt %d/%d): %s",
                name, attempt, MAX_RETRIES, exc,
            )
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2)


def _trim_parquet(path: Path, time_col: str, max_rows: int) -> None:
    if not path.exists():
        return
    df = pd.read_parquet(path)
    if len(df) <= max_rows:
        return
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
    Used where we intentionally want single-row outputs (e.g. P3/P5 inputs).
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


def _log_latest_p5(p5_path: Path) -> None:
    """
    Log the latest row from Pillar 5 execution_plans into a rolling log parquet.
    """
    if not p5_path.exists():
        logging.warning("Cannot log latest outputs; missing Pillar5 parquet.")
        return

    p5 = pd.read_parquet(p5_path)
    if p5.empty:
        logging.warning("Pillar5 outputs empty; skipping log append.")
        return

    p5 = p5.sort_values("open_time_ms")
    latest_p5 = p5.iloc[[-1]]

    if LOG_PATH.exists():
        existing = pd.read_parquet(LOG_PATH)
        combined = pd.concat([existing, latest_p5], ignore_index=True)
    else:
        combined = latest_p5

    combined = combined.sort_values("open_time_ms").iloc[-LOOKBACK_MINUTES:]
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(LOG_PATH, index=False)
    logging.info("Appended latest Pillar5 row to log (%s)", LOG_PATH)


def _sleep_until_next_minute() -> None:
    """
    Sleep until the next whole-minute boundary (UTC).
    """
    now = datetime.now(timezone.utc)
    next_minute = (now.replace(second=0, microsecond=0) + timedelta(minutes=1))
    wait = max(0.0, (next_minute - now).total_seconds())
    time.sleep(wait)


def _current_closed_minute_ms() -> int:
    """
    Return the start timestamp (ms) of the latest *closed* minute.
    I.e., now rounded down to the minute minus one minute.
    """
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    closed = now - pd.Timedelta(minutes=1)
    return int(closed.timestamp() * 1000)


def _load_norm_features() -> pd.DataFrame | None:
    if not NORM_PATH.exists():
        return None
    df = pd.read_parquet(NORM_PATH)
    if df.empty or "open_time_ms" not in df.columns:
        return None
    return df.sort_values("open_time_ms")


# ---------------------------------------------------------------------
# Preload / raw fetchers
# ---------------------------------------------------------------------
def run_preload_full_window(end_ms: int, lookback_minutes: int) -> None:
    """
    Run fetch_candles + rollups + delta_and_micro + trade_aggregator
    for a full `lookback_minutes` window ending at `end_ms` (exclusive).

    This is the "bootstrap" behaviour and also the fallback when no incremental
    top-up functions are available.
    """
    lookback_days = lookback_minutes / 1440.0

    # Make sure both modules agree on the same fixed window
    fetch_candles.LOOKBACK_DAYS = lookback_days
    fetch_candles.FORCE_END_MS = end_ms

    delta_and_micro.LOOKBACK_DAYS = lookback_days
    delta_and_micro.FORCE_END_MS = end_ms

    # 1) Download 1m candles for the window (fresh BTC_CANDLES.csv)
    _run_with_retries(fetch_candles.main, "fetch_candles")

    # 2) Build HTF rollups purely from the local 1m candles
    _run_with_retries(rollups.main, "rollups")

    # 3) Build trades + microstructure using the *fresh* candles file
    _run_with_retries(delta_and_micro.main, "delta_and_micro")

    # 4) Aggregate trades as needed
    _run_with_retries(trade_aggregator.main, "trade_aggregator")


def run_preload_incremental(end_ms: int, lookback_minutes: int) -> None:
    """
    Incremental preload:

    - If the underlying modules expose `topup_to(end_ms, lookback_minutes)`,
      use those to fetch ONLY the missing minutes and maintain a rolling window.
    - Otherwise, fall back to `run_preload_full_window`, which rebuilds the entire
      raw window.

    This is where you plug in your "smart incremental" behaviour without having
    to change the rest of live_runner.
    """
    used_incremental = False

    # fetch_candles: 1m candles
    if hasattr(fetch_candles, "topup_to"):
        logging.info("Using fetch_candles.topup_to for incremental candles.")
        fetch_candles.topup_to(end_ms=end_ms, lookback_minutes=lookback_minutes)
        used_incremental = True

    # delta_and_micro: trades + microstructure
    if hasattr(delta_and_micro, "topup_to"):
        logging.info("Using delta_and_micro.topup_to for incremental raw trades.")
        delta_and_micro.topup_to(end_ms=end_ms, lookback_minutes=lookback_minutes)
        used_incremental = True

    # trade_aggregator: any derived trade-based aggregates
    if hasattr(trade_aggregator, "topup_to"):
        logging.info("Using trade_aggregator.topup_to for incremental trade aggregates.")
        trade_aggregator.topup_to(end_ms=end_ms, lookback_minutes=lookback_minutes)
        used_incremental = True

    # rollups: HTF candles derived from 1m candles
    if hasattr(rollups, "topup_to"):
        logging.info("Using rollups.topup_to for incremental HTF candles.")
        rollups.topup_to(end_ms=end_ms, lookback_minutes=lookback_minutes)
        used_incremental = True
    else:
        # rollups are cheap and purely local; it's fine to recompute from local 1m candles
        logging.info("No rollups.topup_to found; recomputing HTF candles from local 1m.")
        _run_with_retries(rollups.main, "rollups")

    if not used_incremental:
        logging.info(
            "No incremental topup_to() implementations found; falling back to full-window preload."
        )
        run_preload_full_window(end_ms=end_ms, lookback_minutes=lookback_minutes)


# ---------------------------------------------------------------------
# Pre-Pillar 4
# ---------------------------------------------------------------------
def run_pre_p4_full() -> None:
    """
    Build canonical raw + engines + global scaling using whatever raw/micro/candles
    are currently on disk.

    Output: PRE_PILLARFOUR/normalised_features.parquet
    """
    _run_with_retries(build_canonical_raw_live.main, "build_canonical_raw_live")
    _run_with_retries(
        build_canonical_with_engines_live.main,
        "build_canonical_with_engines_live",
    )
    _run_with_retries(
        global_scaling_engine_live.main,
        "global_scaling_engine_live",
    )

    df_norm = _load_norm_features()
    if df_norm is None:
        raise RuntimeError(
            "global_scaling_engine_live completed but normalised_features.parquet "
            "is missing or invalid."
        )

    # Keep the normalised window bounded to LOOKBACK_MINUTES
    if len(df_norm) > LOOKBACK_MINUTES:
        df_norm = df_norm.iloc[-LOOKBACK_MINUTES:]
        PRE_P4_DIR.mkdir(parents=True, exist_ok=True)
        df_norm.to_parquet(NORM_PATH, index=False)
        logging.info(
            "Trimmed normalised_features.parquet to last %d rows", LOOKBACK_MINUTES
        )


def ensure_pre_p4_current(initial: bool) -> None:
    """
    Ensure Pre-Pillar4 is up to date with the latest *closed* minute.

    - initial=True  -> full bootstrap over LOOKBACK_MINUTES.
    - initial=False -> incremental raw top-up (if available) then full Pre-P4 run.

    We allow a few attempts in case the exchange/API lags slightly and the
    latest closed minute isn't immediately available downstream.
    """
    target_closed_ms = _current_closed_minute_ms()
    end_ms = target_closed_ms + 60_000  # end EXCLUSIVE

    logging.info(
        "ensure_pre_p4_current: target_closed_ms=%s end_ms=%s lookback=%d minutes",
        target_closed_ms,
        end_ms,
        LOOKBACK_MINUTES,
    )

    for attempt in range(1, MAX_PRE_P4_TOPUPS + 1):
        logging.info(
            "Pre-P4 ensure loop attempt %d/%d", attempt, MAX_PRE_P4_TOPUPS
        )

        if initial and attempt == 1:
            # First loop: do a full bootstrap fetch
            run_preload_full_window(end_ms=end_ms, lookback_minutes=LOOKBACK_MINUTES)
        else:
            # Subsequent loops or retries: incremental top-up if possible
            run_preload_incremental(end_ms=end_ms, lookback_minutes=LOOKBACK_MINUTES)

        # Build full canonical + engines + normalised window for the current raw state
        run_pre_p4_full()

        df_norm = _load_norm_features()
        if df_norm is None:
            logging.warning(
                "normalised_features.parquet missing/invalid after Pre-P4; retrying..."
            )
            time.sleep(2)
            continue

        latest_ms = int(df_norm["open_time_ms"].max())
        logging.info(
            "Pre-P4 latest normalised open_time_ms=%s (rows=%d)",
            latest_ms,
            len(df_norm),
        )

        if latest_ms >= target_closed_ms:
            logging.info(
                "Pre-P4 is current: latest_ms=%s >= target_closed_ms=%s",
                latest_ms,
                target_closed_ms,
            )
            return

        if attempt < MAX_PRE_P4_TOPUPS:
            logging.info(
                "Pre-P4 behind target (latest_ms=%s < target_closed_ms=%s); retrying...",
                latest_ms,
                target_closed_ms,
            )
            time.sleep(2)

    raise RuntimeError(
        f"Pre-P4 catch-up failed after {MAX_PRE_P4_TOPUPS} attempts; "
        f"latest_ms < target_closed_ms={target_closed_ms}"
    )


# ---------------------------------------------------------------------
# Pillar runners
# ---------------------------------------------------------------------
def run_p4_live() -> None:
    _run_with_retries(
        build_raw_regime_features_live.main,
        "build_raw_regime_features_live",
    )
    _run_with_retries(
        deterministic_regime_engines_live.main,
        "deterministic_regime_engines_live",
    )
    _run_with_retries(
        pillar4_part3_ml_regime_classifier_live.main,
        "pillar4_part3_ml_regime_classifier_live",
    )
    _run_with_retries(
        pillar4_part4_fusion_layer_live.main,
        "pillar4_part4_fusion_layer_live",
    )
    _run_with_retries(
        pillar4_part5_stability_layer_live.main,
        "pillar4_part5_stability_layer_live",
    )
    _run_with_retries(pillar_integration_live.main, "pillar_integration_live")


def run_p1_live() -> None:
    _run_with_retries(
        build_pillar1_feature_matrix_live.main,
        "build_pillar1_feature_matrix_live",
    )
    _run_with_retries(
        build_pillar1_mini_engines_live.main,
        "build_pillar1_mini_engines_live",
    )
    _run_with_retries(
        build_pillar1_labels_live.main,
        "build_pillar1_labels_live",
    )
    _run_with_retries(
        build_pillar1_multi_horizon_ml_live.main,
        "build_pillar1_multi_horizon_ml_live",
    )
    _run_with_retries(
        build_pillar1_meta_features_live.main,
        "build_pillar1_meta_features_live",
    )
    _run_with_retries(
        build_pillar1_master_aggregator_live.main,
        "build_pillar1_master_aggregator_live",
    )
    _run_with_retries(
        build_pillar1_calibration_live.main,
        "build_pillar1_calibration_live",
    )
    _run_with_retries(
        build_pillar1_decision_layer_live.main,
        "build_pillar1_decision_layer_live",
    )


def run_p2_live() -> None:
    _run_with_retries(build_exit_raw_features_live.main, "build_exit_raw_features_live")
    _run_with_retries(build_exit_labels_live.main, "build_exit_labels_live")
    _run_with_retries(build_exit_mini_engines_live.main, "build_exit_mini_engines_live")
    _run_with_retries(build_exit_deterministic_live.main, "build_exit_deterministic_live")
    _run_with_retries(build_exit_ml_models_live.main, "build_exit_ml_models_live")
    _run_with_retries(build_exit_fusion_live.main, "build_exit_fusion_live")
    _run_with_retries(build_exit_final_targets_live.main, "build_exit_final_targets_live")


def run_p3_p5_live() -> None:
    """
    For sizing & execution we deliberately use only the latest bar from
    Pillar 2 final targets, and propagate that through to execution_plans.
    """
    p2_final = RUN_PROCESSED / "PILLAR_TWO_EXIT_MODEL" / "exit_final_targets.parquet"
    _keep_latest_only(p2_final, "open_time_ms")

    _run_with_retries(build_position_sizing_live.main, "build_position_sizing_live")

    p2_with_sizing = RUN_PROCESSED / "PILLAR_TWO_EXIT_MODEL" / "exit_with_sizing.parquet"
    _keep_latest_only(p2_with_sizing, "open_time_ms")

    _run_with_retries(build_execution_plans_live.main, "build_execution_plans_live")

    p5_exec = RUN_PROCESSED / "PILLAR_FIVE_EXECUTION_ENGINE" / "execution_plans.parquet"
    _keep_latest_only(p5_exec, "open_time_ms")

    # NEW: actually send the trade to Binance demo futures
    try:
        execute_latest_plan(p5_exec)
    except Exception as exc:  # noqa: BLE001
        logging.exception("Live trade execution failed: %s", exc)


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    first_run = True

    while True:
        loop_start = time.time()
        print("=== Live pipeline loop start ===")

        # 1) Ensure Pre-Pillar4 has the latest closed minute (full normalised window)
        ensure_pre_p4_current(initial=first_run)
        first_run = False

        # 2) Pillar 4 (regime detection)
        run_p4_live()

        # 3) Pillar 1 (entry model)
        run_p1_live()

        # 4) Pillar 2 (exit model)
        run_p2_live()

        # 5) Pillar 3 & 5 (sizing + execution, latest row only)
        run_p3_p5_live()

        # 6) Trim key parquets to rolling window
        _trim_parquet(
            RUN_PROCESSED / "PILLAR_ONE_ENTRY_MODEL" / "features_pillar1_final.parquet",
            "open_time_ms",
            LOOKBACK_MINUTES,
        )
        _trim_parquet(
            RUN_PROCESSED / "PILLAR_TWO_EXIT_MODEL" / "exit_final_targets.parquet",
            "open_time_ms",
            LOOKBACK_MINUTES,
        )
        _trim_parquet(
            RUN_PROCESSED / "PILLAR_TWO_EXIT_MODEL" / "exit_with_sizing.parquet",
            "open_time_ms",
            LOOKBACK_MINUTES,
        )
        _trim_parquet(
            RUN_PROCESSED / "PILLAR_FIVE_EXECUTION_ENGINE" / "execution_plans.parquet",
            "open_time_ms",
            LOOKBACK_MINUTES,
        )

        # 7) Log latest P5 row
        _log_latest_p5(
            RUN_PROCESSED / "PILLAR_FIVE_EXECUTION_ENGINE" / "execution_plans.parquet"
        )

        loop_dur = time.time() - loop_start
        print(f"=== Live pipeline loop complete in {loop_dur:.1f}s ===")

        # 8) Wait for next whole-minute boundary
        _sleep_until_next_minute()


if __name__ == "__main__":
    main()