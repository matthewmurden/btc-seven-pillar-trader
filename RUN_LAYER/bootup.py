# RUN_LAYER/bootup.py

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from binance.client import Client

# ============================================================
# Repo roots / import wiring
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]  # TRADINGBOTPROJECT root

# Pillar roots
P1_BASE = BASE_DIR / "PILLAR_ONE_ENTRY_MODEL"
P2_BASE = BASE_DIR / "PILLAR_TWO_EXIT_MODEL"
P3_BASE = BASE_DIR / "PILLAR_THREE_RISK_MODEL"
P4_BASE = BASE_DIR / "PILLAR_FOUR_REGIME_DETECTION"
P6_BASE = BASE_DIR / "PILLAR_SIX_SAFETY_LAYER"
PRE_BASE = BASE_DIR / "PRE_PILLARFOUR"

# Make their src dirs importable
for _p in [
    P1_BASE / "src",
    P2_BASE / "src",
    P3_BASE / "src",
    P4_BASE / "src",
    P6_BASE / "src",
    PRE_BASE,
]:
    _s = str(_p)
    if _s not in sys.path:
        sys.path.append(_s)

# ------------------------------------------------------------
# Import engines / models
# (Rename these imports if your actual module names differ.)
# ------------------------------------------------------------

# PRE_PILLARFOUR global feature engines
from PRE_PILLARFOUR.global_engines.volatility_engine import (  # type: ignore
    compute as vol_compute,
)
from PRE_PILLARFOUR.global_engines.session_engine import (  # type: ignore
    compute as sess_compute,
)
from PRE_PILLARFOUR.global_scaling_engine import (  # type: ignore
    GlobalScalingEngine,
    collect_feature_list,
)

# Pillar 4 – regime detection
from regime_engine import compute_regimes  # type: ignore

# Pillar 1 – entry model
from entry_model import run_entry_model  # type: ignore

# Pillar 2 – final TP/SL targets (NO filters version)
from exit.exit_final_targets import (  # type: ignore
    build_final_exit_targets,
)

# Pillar 3 – risk & sizing
from risk.risk_surface import (  # type: ignore
    compute_risk_surface_and_sizing,
    RiskSurfaceConfig,
)

# Pillar 6 – safety / guards / filters
from safety_engine import (  # type: ignore
    SafetyEngineState,
    bootstrap_safety_from_history,
)


# ============================================================
# Config + state dataclasses
# ============================================================

@dataclass
class RunConfig:
    symbol: str = "BTCUSDT"
    interval: str = "1m"
    lookback_minutes: int = 60 * 24  # 1 day rolling window
    # "DRY_RUN" → demo env, no live orders
    # "DEMO"    → demo env, can place demo orders
    # "LIVE"    → real futures
    mode: str = "DRY_RUN"


@dataclass
class BootState:
    cfg: RunConfig
    client: Client
    history_df: pd.DataFrame          # full warmup window with all pillar outputs
    latest_row: pd.Series             # last row of history_df
    safety_state: SafetyEngineState   # Pillar 6 state (loss streak, cooldown, etc.)


# ============================================================
# Binance helpers
# ============================================================

BINANCE_FUTURES_DEMO_URL = "https://demo-fapi.binance.com"


def make_binance_client(cfg: RunConfig) -> Client:
    """
    Create a Binance client configured for:
      - Demo Futures (Binance 'Demo Trading' env) when mode != 'LIVE'
      - Live Futures when mode == 'LIVE'

    Environment variables used:
      DEMO:
        BINANCE_DEMO_API_KEY
        BINANCE_DEMO_API_SECRET
      LIVE:
        BINANCE_API_KEY
        BINANCE_API_SECRET
    """
    if cfg.mode == "LIVE":
        api_key = os.environ["BINANCE_API_KEY"]
        api_secret = os.environ["BINANCE_API_SECRET"]
        client = Client(api_key, api_secret)
        # Default API_URL already points to live binance.com
        logging.info("Bootup: using LIVE Binance futures endpoint")
    else:
        api_key = os.environ["BINANCE_DEMO_API_KEY"]
        api_secret = os.environ["BINANCE_DEMO_API_SECRET"]
        client = Client(api_key, api_secret)
        # Point REST requests (incl. futures_klines, futures orders) to demo futures
        client.API_URL = BINANCE_FUTURES_DEMO_URL
        logging.info("Bootup: using DEMO Binance futures endpoint (%s)", BINANCE_FUTURES_DEMO_URL)

    return client


def fetch_history_klines(client: Client, cfg: RunConfig) -> pd.DataFrame:
    """
    Fetch raw historical 1m futures klines for bootup.
    Returns a DataFrame whose schema matches what PRE_PILLARFOUR expects.
    """
    logging.info(
        "Bootup: fetching %s minutes of %s %s klines",
        cfg.lookback_minutes,
        cfg.symbol,
        cfg.interval,
    )

    # For 1m, lookback_minutes == number of bars we want.
    limit = min(cfg.lookback_minutes, 1500)  # Binance per-call limit is ~1500
    # For >1500 you'd loop; 1d = 1440 so we're fine.
    klines = client.futures_klines(
        symbol=cfg.symbol,
        interval=cfg.interval,
        limit=limit,
    )

    # Map Binance kline fields to your internal columns
    cols = [
        "open_time_ms",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time_ms",
        "quote_volume",
        "num_trades",
        "taker_base_volume",
        "taker_quote_volume",
        "ignore",
    ]
    df = pd.DataFrame(klines, columns=cols)

    # Convert to numeric / ms
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_base_volume",
        "taker_quote_volume",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["open_time_ms"] = df["open_time_ms"].astype("int64")
    df["close_time_ms"] = df["close_time_ms"].astype("int64")

    # Drop Binance's unused "ignore" column
    df = df.drop(columns=["ignore"])

    logging.info(
        "Bootup: fetched history df shape=%s (from Binance)",
        df.shape,
    )
    return df


# ============================================================
# Feature stack + pillar warmup
# ============================================================

def compute_pre_pillarfour_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run PRE_PILLARFOUR feature engines so that the final feature set
    matches what you used in offline training/backtests.
    """
    df = raw_df.copy()

    # 1) Global scaling / basic transforms
    glb = GlobalScalingEngine()
    df = glb.transform(df)

    # 2) Volatility engine
    df = vol_compute(df)

    # 3) Session / time-of-day engine
    df = sess_compute(df)

    # 4) Optional: restrict to the feature list actually used in models
    feat_list = collect_feature_list()
    # Keep time/price core columns too
    keep_cols = list(
        {"open_time_ms", "open", "high", "low", "close"}.union(feat_list)
    )
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    logging.info(
        "Bootup: pre-Pillar4 features computed, shape=%s, cols=%d",
        df.shape,
        df.shape[1],
    )
    return df


def run_pillars_on_history(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Pillar 4 → 1 → 2 → 3 on the historical rolling window.
    This gives us a fully-populated feature frame for the run layer.
    """
    df = features_df.copy()

    # ---- Pillar 4: Regime Detection ----
    regimes = compute_regimes(df)  # expected to return DataFrame indexed the same
    df = df.join(regimes)

    logging.info(
        "Bootup: Pillar 4 regimes joined, shape=%s, cols=%d",
        df.shape,
        df.shape[1],
    )

    # ---- Pillar 1: Entry Model ----
    df = run_entry_model(df)
    logging.info(
        "Bootup: Pillar 1 entry model outputs added, shape=%s, cols=%d",
        df.shape,
        df.shape[1],
    )

    # ---- Pillar 2: Exit Model (no internal filters) ----
    df = build_final_exit_targets(df)
    logging.info(
        "Bootup: Pillar 2 final TP/SL targets added, shape=%s, cols=%d",
        df.shape,
        df.shape[1],
    )

    # ---- Pillar 3: Risk & Sizing ----
    p3_cfg = RiskSurfaceConfig()
    df = compute_risk_surface_and_sizing(df, cfg=p3_cfg)
    logging.info(
        "Bootup: Pillar 3 sizing added, shape=%s, cols=%d",
        df.shape,
        df.shape[1],
    )

    return df


def warmup_safety_layer(full_df: pd.DataFrame) -> SafetyEngineState:
    """
    Initialise Pillar 6 safety state from the historical window.
    This is where you look at recent trades, loss streaks, volatility, etc.
    """
    safety_state = bootstrap_safety_from_history(full_df)
    logging.info("Bootup: Pillar 6 safety state initialised: %s", safety_state)
    return safety_state


# ============================================================
# Public entrypoint
# ============================================================

def run_bootup(cfg: Optional[RunConfig] = None) -> BootState:
    """
    Full bootup:
      1. Connect to Binance (demo or live, depending on cfg.mode)
      2. Fetch historical candles
      3. Run PRE_PILLARFOUR engines
      4. Run Pillars 4 → 1 → 2 → 3 on history
      5. Initialise Pillar 6 safety state
      6. Return BootState with last row + state
    """
    if cfg is None:
        cfg = RunConfig()

    logging.info("======== LIVE BOT BOOTUP START ========")

    client = make_binance_client(cfg)
    raw_df = fetch_history_klines(client, cfg)

    features_df = compute_pre_pillarfour_features(raw_df)
    full_df = run_pillars_on_history(features_df)

    # Safety warmup (Pillar 6)
    safety_state = warmup_safety_layer(full_df)

    latest_row = full_df.iloc[-1].copy()

    logging.info(
        "Bootup complete: history_len=%d, last_ts=%s",
        len(full_df),
        int(latest_row.get("open_time_ms", -1)),
    )

    return BootState(
        cfg=cfg,
        client=client,
        history_df=full_df,
        latest_row=latest_row,
        safety_state=safety_state,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    state = run_bootup()
    # Just print a tiny summary so you can see bootup works
    print("Bootup OK. History shape:", state.history_df.shape)
    print("Last row time:", state.latest_row.get("open_time_ms"))