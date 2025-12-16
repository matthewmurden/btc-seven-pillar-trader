from __future__ import annotations

"""
Live helper to generate execution plans (Pillar 5) from live Pillar 2/3 outputs.

Inputs:
  RUN_LAYER/data/processed/PILLAR_TWO_EXIT_MODEL/exit_with_sizing.parquet

Outputs:
  RUN_LAYER/data/processed/PILLAR_FIVE_EXECUTION_ENGINE/execution_plans.parquet

Notes:
  - This is a lightweight, deterministic application of the existing
    ExecutionEngine. It does NOT place orders.
  - Market snapshot fields are approximated from the bar close (best effort);
    adapt as needed once you wire real market data.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
# Repo root: TRADINGBOTPROJECT
BASE_DIR = THIS_DIR.parents[2]

RUN_PROCESSED = BASE_DIR / "RUN_LAYER" / "data" / "processed"

P2_PROCESSED = RUN_PROCESSED / "PILLAR_TWO_EXIT_MODEL"
INPUT_PATH = P2_PROCESSED / "exit_with_sizing.parquet"

P5_PROCESSED = RUN_PROCESSED / "PILLAR_FIVE_EXECUTION_ENGINE"
OUTPUT_PATH = P5_PROCESSED / "execution_plans.parquet"

# Pillar 5 src
P5_BASE = BASE_DIR / "PILLAR_FIVE_EXECUTION_ENGINE"
P5_SRC = P5_BASE / "src"
for p in (P5_BASE, P5_SRC):
    ps = str(p)
    if ps not in sys.path:
        sys.path.append(ps)

from execution.execution_engine import (  # type: ignore
    ExecutionEngine,
    ExecutionConfig,
    EntrySignal,
    ExitPlan,
    RiskParams,
    RiskMode,
    RegimeInfo,
    SafetyStatus,
    MarketSnapshot,
    OrderSide,
    VolatilityRegime,
    LiquidityRegime,
    TrendChopFlag,
)


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024**2)
    logging.info("%s | shape=%s | approx_mem=%.2f MB", msg, df.shape, mb)


# ---------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------

def _safe_side_from_decision(val: Any) -> OrderSide:
    """
    Fallback when we don't have a numeric direction.
    Treat strings starting with 's' as short, otherwise long.
    """
    s = str(val).lower()
    if s.startswith("s"):
        return OrderSide.SHORT
    return OrderSide.LONG


def _build_entry_signal(row: pd.Series) -> EntrySignal:
    # Primary: numeric direction from Pillar 1/3
    direction = float(row.get("entry_direction", row.get("direction", 0.0)))
    if direction > 0:
        side = OrderSide.LONG
    elif direction < 0:
        side = OrderSide.SHORT
    else:
        side = _safe_side_from_decision(row.get("entry_decision", "long"))

    # Prob / confidence fallbacks
    prob = float(
        row.get(
            "entry_probability",
            row.get(
                "entry_confidence",
                row.get("short_dir_confidence", 0.5),
            ),
        )
    )
    conf = float(row.get("entry_confidence", prob))

    return EntrySignal(
        side=side,
        entry_probability=max(0.0, min(1.0, prob)),
        signal_confidence=max(0.0, min(1.0, conf)),
        time_sensitivity=float(row.get("time_sensitivity", 0.5)),
        pattern_momentum_score=float(row.get("momentum_score", 0.0)),
    )


def _build_exit_plan(row: pd.Series) -> ExitPlan:
    # TP / SL levels from Pillar 2 final targets
    tp = float(
        row.get(
            "optimal_tp",
            row.get("tp_price", row.get("exec_tp", row.get("close", 0.0))),
        )
    )
    sl = float(
        row.get(
            "optimal_sl",
            row.get("stop_price", row.get("exec_sl", tp)),
        )
    )

    tp_first_prob = float(
        row.get(
            "tp_first_probability",
            row.get("predicted_TP_first_probability", 0.5),
        )
    )

    expected_exit = float(
        row.get(
            "expected_exit_time",
            row.get("predicted_exit_time", 0.0),
        )
    )

    return ExitPlan(
        tp_price=tp,
        sl_price=sl,
        tp_first_probability=max(0.0, min(1.0, tp_first_prob)),
        expected_exit_time_sec=max(0.0, expected_exit),
        exit_entropy=float(row.get("exit_entropy", 0.0)),
    )


def _build_risk_params(row: pd.Series) -> RiskParams:
    size = float(row.get("position_size", row.get("final_position_size", 0.0)))
    lev = float(row.get("position_leverage", row.get("leverage_to_apply", 1.0)))

    # volatility_adjustment_flag is a string in P3 ("cut"/"expanded"/"neutral").
    vol_adj_raw = row.get("volatility_adjustment_flag", "")
    if isinstance(vol_adj_raw, str):
        vol_adj = vol_adj_raw.lower() in ("cut", "expanded")
    else:
        vol_adj = bool(vol_adj_raw)

    raw_mode = str(row.get("risk_mode", "normal")).lower()
    if raw_mode in ("defensive", "crisis", "conservative"):
        risk_mode = RiskMode.CONSERVATIVE
    elif raw_mode == "aggressive":
        risk_mode = RiskMode.AGGRESSIVE
    else:
        risk_mode = RiskMode.NORMAL

    max_fill_spread = float(row.get("max_fill_spread_bps", 10.0))

    # Use size_volatility_multiplier as a proxy for aggressiveness, but clamp to [0,1].
    agg_raw = float(row.get("size_volatility_multiplier", 0.5))
    aggressiveness = max(0.0, min(1.0, agg_raw))

    return RiskParams(
        final_position_size=size,
        leverage_to_apply=lev,
        volatility_adjustment_flag=vol_adj,
        risk_mode=risk_mode,
        max_fill_spread_bps=max_fill_spread,
        aggressiveness=aggressiveness,
    )


def _build_regime(row: pd.Series) -> RegimeInfo:
    # Volatility regime: use volatility_score + vol expansion flag
    vol_score = float(row.get("volatility_score", 0.0))
    vol_exp_flag = bool(row.get("regime_is_vol_expansion", 0.0))

    if vol_exp_flag and vol_score > 1.5:
        vol_reg = VolatilityRegime.EXTREME
    elif vol_exp_flag or vol_score > 0.7:
        vol_reg = VolatilityRegime.HIGH
    elif vol_score < -0.7:
        vol_reg = VolatilityRegime.LOW
    else:
        vol_reg = VolatilityRegime.NORMAL

    # Liquidity regime: liq drought flag → POOR, else AVERAGE for now
    liq_drought_flag = bool(row.get("regime_is_liq_drought", 0.0))
    if liq_drought_flag:
        liq_reg = LiquidityRegime.POOR
    else:
        liq_reg = LiquidityRegime.AVERAGE

    # Trend / chop
    chop_flag = bool(row.get("regime_is_chop", row.get("regime_chop_mask", 0.0)))
    trend_flag = bool(row.get("regime_is_trend", 0.0))

    if chop_flag:
        trend_chop = TrendChopFlag.CHOP
    elif trend_flag:
        trend_chop = TrendChopFlag.TREND
    else:
        trend_chop = TrendChopFlag.UNKNOWN

    regime_final = str(row.get("regime_final", row.get("regime_label_fused", "unknown")))
    regime_conf = float(row.get("regime_confidence", row.get("regime_stability_score", 0.0)))

    return RegimeInfo(
        regime_final=regime_final,
        volatility_regime=vol_reg,
        liquidity_regime=liq_reg,
        chop_trend_flag=trend_chop,
        ignition_flag=bool(row.get("ignition_flag", False)),
        mean_reversion_flag=bool(row.get("mean_reversion_flag", False)),
        regime_confidence=max(0.0, min(1.0, regime_conf)),
    )


def _build_market_snapshot(row: pd.Series) -> MarketSnapshot:
    mid = float(row.get("close", row.get("mid_price", row.get("tp_price", 0.0))))

    spread = float(row.get("spread", 0.0))
    if (spread <= 0.0) and mid > 0:
        # Default to 1 bps if we don't know the spread
        spread = mid * 0.0001

    # Best bid/ask fallbacks if not provided
    default_bid = mid - spread / 2 if mid > 0 else 0.0
    default_ask = mid + spread / 2 if mid > 0 else 0.0
    best_bid = float(row.get("best_bid", default_bid))
    best_ask = float(row.get("best_ask", default_ask))

    # Volatility: if volatility_bps already exists, use it raw; otherwise scale rv_5m
    if "volatility_bps" in row.index:
        vol_bps = float(row.get("volatility_bps", 0.0))
    else:
        vol_raw = float(row.get("rv_5m", 0.0))
        vol_bps = vol_raw * 1e4 if mid > 0 else 0.0

    vol_bps = max(0.0, vol_bps)

    return MarketSnapshot(
        ts_ms=int(row.get("open_time_ms", 0)),
        best_bid=best_bid,
        best_ask=best_ask,
        mid_price=mid,
        spread=spread,
        spread_bps=spread / mid * 1e4 if mid > 0 else 0.0,
        bids=[],
        asks=[],
        volatility_lookback_sec=float(row.get("volatility_lookback_sec", 60.0)),
        volatility_bps=vol_bps,
        depth_imbalance=float(row.get("depth_imbalance", 0.0)),
    )


def _plan_to_row(plan, open_time_ms: Any, execute: int) -> Dict[str, Any]:
    return {
        "open_time_ms": int(open_time_ms),
        "order_type": plan.order_type.value,
        "side": plan.side.value,
        "order_price": plan.order_price,
        "allowed_slippage_bps": plan.allowed_slippage_bps,
        "position_size": plan.position_size,
        "leverage_to_apply": plan.leverage_to_apply,
        "stop_price": plan.stop_price,
        "tp_price": plan.tp_price,
        "expiration_time_ms": plan.expiration_time_ms,
        "slippage_estimate_bps": plan.slippage_estimate_bps,
        "fill_probability": plan.fill_probability,
        "fee_estimate_bps": plan.fee_estimate_bps,
        "execute": int(execute),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    # Usually configured by live_runner; uncomment if running standalone:
    # logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"exit_with_sizing parquet not found: {INPUT_PATH}")

    logging.info("Loading sized trades (live): %s", INPUT_PATH)
    trades = pd.read_parquet(INPUT_PATH)
    _log_mem(trades, "[Exec] Loaded sized trades")

    if trades.empty:
        logging.warning("[Exec] exit_with_sizing is empty; writing empty execution_plans.")
        P5_PROCESSED.mkdir(parents=True, exist_ok=True)
        trades.to_parquet(OUTPUT_PATH, index=False, compression="zstd")
        return

    # Ensure deterministic ordering
    if "open_time_ms" in trades.columns:
        trades = trades.sort_values("open_time_ms").reset_index(drop=True)

    engine = ExecutionEngine(ExecutionConfig())

    plans: List[Dict[str, Any]] = []
    for _, row in trades.iterrows():
        # Gate: mirror P3/P2 logic as best we can
        gate = True
        gate &= str(row.get("entry_decision", "none")).lower() != "none"
        gate &= float(row.get("entry_direction", row.get("direction", 0.0))) != 0.0
        gate &= float(row.get("position_size", row.get("final_position_size", 0.0))) > 0.0

        # system_filter_v1 if present
        if "system_filter_v1" in row.index:
            gate &= bool(row.get("system_filter_v1", True))

        # require TP/SL distances if available
        if "optimal_tp_distance" in row.index and "optimal_sl_distance" in row.index:
            gate &= (
                float(row.get("optimal_tp_distance", 0.0)) > 0.0
                and float(row.get("optimal_sl_distance", 0.0)) > 0.0
            )

        safety_allowed = bool(row.get("safety_allowed", True)) and gate
        safety = SafetyStatus(allowed=safety_allowed)

        entry = _build_entry_signal(row)
        exit_plan = _build_exit_plan(row)
        risk = _build_risk_params(row)
        regime = _build_regime(row)
        market = _build_market_snapshot(row)

        plan = engine.build_execution_plan(
            entry_signal=entry,
            exit_plan=exit_plan,
            risk_params=risk,
            regime=regime,
            safety=safety,
            market=market,
            now_ms=int(row.get("open_time_ms", 0)),
        )
        plans.append(_plan_to_row(plan, row.get("open_time_ms"), int(gate and safety_allowed)))

    plans_df = pd.DataFrame(plans)
    _log_mem(plans_df, "[Exec] Execution plans")

    P5_PROCESSED.mkdir(parents=True, exist_ok=True)
    plans_df.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Saved execution plans (live) -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()