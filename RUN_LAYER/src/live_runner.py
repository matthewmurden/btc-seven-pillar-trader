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
    • Run Pillar 3 risk engine (p3_engine) on latest row, then Pillar 6 safety,
      then Pillar 5 execution engine, and log via Pillar 7 monitoring.
    • Trim key downstream parquet files to LOOKBACK_MINUTES rows.
    • Append the latest Pillar 5 row to a rolling log parquet.

The loop runs indefinitely, aligned to minute boundaries.
"""

import importlib.util
import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, Any, List, Optional, Dict

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

# Base pillar modules (non-live)
P3_ENGINE_PATH = BASE_DIR / "PILLAR_THREE_RISK_MODEL" / "src" / "pipeline" / "p3_engine.py"
P5_ENGINE_PATH = BASE_DIR / "PILLAR_FIVE_EXECUTION_ENGINE" / "src" / "execution" / "execution_engine.py"
P6_ENGINE_PATH = BASE_DIR / "PILLAR_SIX_SAFETY_LAYER" / "src" / "safety" / "safety_engine.py"
P7_MONITORING_PATH = BASE_DIR / "PILLAR_SEVEN_MONITORING" / "monitoring.py"
P7_ROOT = RUN_PROCESSED / "PILLAR_SEVEN_MONITORING"

# Downstream outputs
P1_FINAL = RUN_PROCESSED / "PILLAR_ONE_ENTRY_MODEL" / "features_pillar1_final.parquet"
P2_FINAL = RUN_PROCESSED / "PILLAR_TWO_EXIT_MODEL" / "exit_final_targets.parquet"
P4_DET = RUN_PROCESSED / "PILLAR_FOUR_REGIME_DETECTION" / "deterministic_regime_engines.parquet"
P3_OUT = RUN_PROCESSED / "PILLAR_THREE_RISK_MODEL" / "risk_intents.parquet"
P5_OUT = RUN_PROCESSED / "PILLAR_FIVE_EXECUTION_ENGINE" / "execution_plans.parquet"

# ---------------------------------------------------------------------
# Import wiring for all live modules
# ---------------------------------------------------------------------
SRC_DIRS: List[Path] = [
    THIS_DIR,
    THIS_DIR / "PRE_PILLARFOURLIVE",
    THIS_DIR / "PILLAR_FOUR_REGIME_DETECTION_LIVE",
    THIS_DIR / "PILLAR_ONE_ENTRY_MODEL_LIVE",
    THIS_DIR / "PILLAR_TWO_EXIT_MODEL_LIVE",
]

for p in SRC_DIRS:
    ps = str(p)
    if ps not in sys.path:
        sys.path.append(ps)


def _load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Module path missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    # Ensure dataclasses can resolve module references during import.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Base pillar modules (loaded by path to avoid import collisions)
p3_engine = _load_module("p3_engine", P3_ENGINE_PATH)
p5_engine_mod = _load_module("p5_engine", P5_ENGINE_PATH)
p6_engine_mod = _load_module("p6_engine", P6_ENGINE_PATH)
p7_monitoring = _load_module("p7_monitoring", P7_MONITORING_PATH)

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


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _run_with_retries(func: Callable[[], Any], name: str) -> Any:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func()
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


def _load_latest_row(path: Path, time_col: str = "open_time_ms") -> Optional[pd.Series]:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    if time_col in df.columns:
        df = df.sort_values(time_col)
    return df.iloc[-1]


def _append_or_replace_by_open_time(path: Path, row: Dict[str, Any], time_col: str = "open_time_ms") -> None:
    df_new = pd.DataFrame([row])
    if path.exists():
        df = pd.read_parquet(path)
        if time_col in df.columns and time_col in df_new.columns:
            df = df[df[time_col] != df_new.iloc[0][time_col]]
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    if time_col in df.columns:
        df = df.sort_values(time_col)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _first_present(row: Dict[str, Any] | pd.Series, keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in row and not pd.isna(row[k]):
            return row[k]
    return default


def _build_market_snapshot(p5_exec: Any, p1_row: pd.Series, spread_bps_hint: Optional[float]) -> Any:
    snap = p5_exec.market.snapshot()
    best_bid = snap.get("best_bid")
    best_ask = snap.get("best_ask")
    mid = snap.get("mid")
    spread_bps = snap.get("spread_bps")

    if (best_bid is None or best_ask is None) and hasattr(p5_exec, "client"):
        try:
            ticker = p5_exec.client.get_book_ticker(str(_first_present(p1_row, ["symbol"], "BTCUSDT")))
            best_bid = float(ticker.get("bidPrice", best_bid or 0))
            best_ask = float(ticker.get("askPrice", best_ask or 0))
        except Exception as exc:  # noqa: BLE001
            logging.warning("P5 book ticker fallback failed: %s", exc)

    if mid is None and best_bid is not None and best_ask is not None and best_bid > 0 and best_ask > 0:
        mid = (best_bid + best_ask) / 2.0
    if spread_bps is None and mid and best_bid is not None and best_ask is not None:
        spread_bps = ((best_ask - best_bid) / mid) * 1e4
    if spread_bps is None and spread_bps_hint is not None:
        spread_bps = float(spread_bps_hint)

    if mid is None:
        mid = float(_first_present(p1_row, ["close", "open"], 0.0))
    if best_bid is None:
        best_bid = float(mid)
    if best_ask is None:
        best_ask = float(mid)
    if spread_bps is None:
        spread_bps = 0.0

    close = float(_first_present(p1_row, ["close"], 0.0))
    atr = float(_first_present(p1_row, ["vol_atr_14", "atr_14"], 0.0))
    volatility_bps = (atr / close * 1e4) if close > 0 else 0.0

    return p6_engine_mod.MarketMicroSnapshot(
        ts_ms=int(time.time() * 1000),
        best_bid=float(best_bid),
        best_ask=float(best_ask),
        mid=float(mid),
        spread_bps=float(spread_bps),
        volatility_bps=float(volatility_bps),
    )


def _build_p5_intent(p3_row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "open_time_ms": int(p3_row.get("open_time_ms")),
        "symbol": str(p3_row.get("symbol", "BTCUSDT")),
        "side": str(p3_row.get("side", "")).lower(),
        "position_size": float(p3_row.get("final_position_size", 0.0)),
        "leverage_to_apply": int(round(float(p3_row.get("leverage_to_apply", 1)))),
        "tp_price": float(p3_row.get("tp_price", 0.0)),
        "stop_price": float(p3_row.get("sl_price", p3_row.get("stop_price", 0.0))),
        "max_fill_spread_bps": float(p3_row.get("max_fill_spread_bps", 0.0)),
        "max_chase_bps": float(p3_row.get("max_chase_bps", 0.0)),
        "time_budget_ms": int(p3_row.get("time_budget_ms", 0)),
        "risk_mode": str(p3_row.get("risk_mode", "normal")),
        "allowed_to_trade": bool(p3_row.get("allowed_to_trade", True)),
        "veto_reason": str(p3_row.get("veto_reason", "")),
        "execute": int(p3_row.get("execute", 1)),
    }


def _log_event_safe(logger: Optional[Any], **kwargs: Any) -> None:
    if logger is None:
        return
    try:
        logger.log_event(**kwargs)
    except Exception as exc:  # noqa: BLE001
        logging.warning("P7 log_event failed: %s", exc)


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
    _run_with_retries(build_exit_mini_engines_live.main, "build_exit_mini_engines_live")
    _run_with_retries(build_exit_deterministic_live.main, "build_exit_deterministic_live")
    _run_with_retries(build_exit_ml_models_live.main, "build_exit_ml_models_live")
    _run_with_retries(build_exit_fusion_live.main, "build_exit_fusion_live")
    _run_with_retries(build_exit_final_targets_live.main, "build_exit_final_targets_live")


def run_p3_p5_p6_p7(
    p5_exec: Any,
    p6_safety: Optional[Any],
    p7_logger: Optional[Any],
) -> None:
    """
    Pillar 3 (risk) -> Pillar 6 (safety) -> Pillar 5 (execution) -> Pillar 7 (monitoring).
    P3 selects the latest open_time_ms internally and aligns P2/P4 to it.
    """
    def _run_p3():
        return p3_engine.run_pillar3(
            p1_path=P1_FINAL,
            p2_path=P2_FINAL,
            p4_path=P4_DET,
            out_path=P3_OUT,
            live_account=True,
            live_spread=True,
        )

    p3_row = _run_with_retries(_run_p3, "p3_engine")
    if not p3_row:
        raise RuntimeError("P3 returned empty risk intent.")

    p5_intent = _build_p5_intent(p3_row)
    trade_intent_id = f"{p5_intent['symbol']}_1m_{p5_intent['open_time_ms']}_{p5_intent['side']}"
    decision_id = f"{p5_intent['symbol']}_1m_{p5_intent['open_time_ms']}"

    _log_event_safe(
        p7_logger,
        event_type="P3_RISK_INTENT",
        symbol=p5_intent["symbol"],
        trade_intent_id=trade_intent_id,
        decision_id=decision_id,
        source="P3",
        payload=p3_row,
    )

    execute = (
        p5_intent["execute"] == 1
        and p5_intent["allowed_to_trade"]
        and p5_intent["position_size"] > 0
    )
    p6_reason = ""

    if execute and p6_safety is not None:
        p1_row = _load_latest_row(P1_FINAL)
        if p1_row is None:
            logging.warning("P6 skipped: missing P1 final row.")
        else:
            market = _build_market_snapshot(
                p5_exec,
                p1_row,
                spread_bps_hint=p3_row.get("spread_bps_used"),
            )
            entry_conf = p1_row.get("entry_confidence")
            try:
                entry_conf = float(entry_conf) if entry_conf is not None else None
            except Exception:
                entry_conf = None

            entry_ref = p3_row.get("entry_ref_price")
            try:
                entry_ref = float(entry_ref) if entry_ref is not None else None
            except Exception:
                entry_ref = None

            sl_dist = p3_row.get("sl_distance_abs")
            try:
                sl_dist = float(sl_dist) if sl_dist is not None else None
            except Exception:
                sl_dist = None

            tp_dist = None
            if entry_ref is not None:
                try:
                    tp_dist = abs(float(p5_intent["tp_price"]) - float(entry_ref))
                except Exception:
                    tp_dist = None

            intent = p6_engine_mod.RiskIntent(
                side=p5_intent["side"],
                final_position_size=p5_intent["position_size"],
                leverage_to_apply=p5_intent["leverage_to_apply"],
                risk_mode=p5_intent.get("risk_mode", "normal"),
                max_fill_spread_bps=p5_intent["max_fill_spread_bps"],
                max_chase_bps=p5_intent["max_chase_bps"],
                time_budget_ms=p5_intent["time_budget_ms"],
                tp_price=p5_intent["tp_price"],
                sl_price=p5_intent["stop_price"],
                entry_ref_price=entry_ref,
                sl_distance_abs=sl_dist,
                tp_distance_abs=tp_dist,
                entry_confidence=entry_conf,
                open_time_ms=p5_intent["open_time_ms"],
            )
            decision = p6_safety.pre_trade_check(intent, market)
            if not decision.allowed:
                execute = False
                p6_reason = decision.reason
            else:
                if decision.forced_max_chase_bps is not None:
                    p5_intent["max_chase_bps"] = float(decision.forced_max_chase_bps)
                if decision.forced_time_budget_ms is not None:
                    p5_intent["time_budget_ms"] = int(decision.forced_time_budget_ms)

            _log_event_safe(
                p7_logger,
                event_type="P6_VETO" if not decision.allowed else "P6_ALLOW",
                symbol=p5_intent["symbol"],
                trade_intent_id=trade_intent_id,
                decision_id=decision_id,
                source="P6",
                payload={
                    "allowed": decision.allowed,
                    "reason": decision.reason,
                    **(decision.metadata or {}),
                },
            )

    p5_intent["execute"] = 1 if execute else 0
    if p6_reason:
        p5_intent["p6_reason"] = p6_reason

    _append_or_replace_by_open_time(P5_OUT, p5_intent)

    if not execute:
        logging.info("Execution skipped (execute=0).")
        return

    # Execute via Pillar 5 engine
    try:
        res = p5_exec.execute_intent(p5_intent)
        _log_event_safe(
            p7_logger,
            event_type="P5_EXECUTION",
            symbol=p5_intent["symbol"],
            trade_intent_id=trade_intent_id,
            decision_id=decision_id,
            source="P5",
            payload=res,
        )
    except Exception as exc:  # noqa: BLE001
        logging.exception("P5 execution failed: %s", exc)
        _log_event_safe(
            p7_logger,
            event_type="P5_EXECUTION_FAIL",
            symbol=p5_intent["symbol"],
            trade_intent_id=trade_intent_id,
            decision_id=decision_id,
            source="P5",
            payload={"error": str(exc)},
        )


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    p5_exec = p5_engine_mod.Pillar5ExecutionEngine()
    p5_exec.start(use_websocket=True)

    p6_safety = None
    try:
        p6_safety = p6_engine_mod.Pillar6Safety()
    except Exception as exc:  # noqa: BLE001
        logging.warning("P6 safety init failed; skipping safety layer. %s", exc)

    p7_logger = p7_monitoring.EventLogger(
        p7_monitoring.Pillar7Config(root_dir=P7_ROOT)
    )

    first_run = True

    try:
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

            # Short-circuit if Pillar 1 produced no actionable trade
            p1_latest = _load_latest_row(P1_FINAL)
            try:
                latest_direction = float(p1_latest.get("entry_direction", 0))
            except Exception:  # noqa: BLE001
                latest_direction = 0.0

            no_trade = (
                p1_latest is None
                or str(p1_latest.get("entry_decision", "none")).lower() == "none"
                or latest_direction == 0
            )
            if no_trade:
                logging.info("P1 produced no trade on latest bar; skipping Pillars 2/3/5/6 this loop.")
                loop_dur = time.time() - loop_start
                print(f"=== Live pipeline loop complete (no trade) in {loop_dur:.1f}s ===")
                _sleep_until_next_minute()
                continue

            # 4) Pillar 2 (exit model)
            run_p2_live()

            # 5) Pillar 3 -> Pillar 6 -> Pillar 5 -> Pillar 7
            run_p3_p5_p6_p7(
                p5_exec=p5_exec,
                p6_safety=p6_safety,
                p7_logger=p7_logger,
            )

            # 6) Trim key parquets to rolling window
            _trim_parquet(P1_FINAL, "open_time_ms", LOOKBACK_MINUTES)
            _trim_parquet(P2_FINAL, "open_time_ms", LOOKBACK_MINUTES)
            _trim_parquet(P3_OUT, "open_time_ms", LOOKBACK_MINUTES)
            _trim_parquet(P5_OUT, "open_time_ms", LOOKBACK_MINUTES)

            # 7) Log latest P5 row
            _log_latest_p5(P5_OUT)

            loop_dur = time.time() - loop_start
            print(f"=== Live pipeline loop complete in {loop_dur:.1f}s ===")

            # 8) Wait for next whole-minute boundary
            _sleep_until_next_minute()
    finally:
        p5_exec.stop()


if __name__ == "__main__":
    main()
