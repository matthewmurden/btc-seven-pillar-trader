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


def _safe_side(val: Any) -> OrderSide:
    if str(val).lower().startswith("s"):
        return OrderSide.SHORT
    return OrderSide.LONG


def _build_entry_signal(row: pd.Series) -> EntrySignal:
    prob = float(row.get("entry_probability", row.get("entry_confidence", 0.5)))
    conf = float(row.get("entry_confidence", prob))
    return EntrySignal(
        side=_safe_side(row.get("entry_decision", "long")),
        entry_probability=prob,
        signal_confidence=conf,
        time_sensitivity=0.5,
        pattern_momentum_score=float(row.get("momentum_score", 0.0)),
    )


def _build_exit_plan(row: pd.Series) -> ExitPlan:
    tp = float(row.get("optimal_tp", row.get("tp_price", row.get("exec_tp", row.get("close", 0.0)))))
    sl = float(row.get("optimal_sl", row.get("stop_price", row.get("exec_sl", tp))))
    return ExitPlan(
        tp_price=tp,
        sl_price=sl,
        tp_first_probability=float(row.get("predicted_TP_first_probability", 0.5)),
        expected_exit_time_sec=float(row.get("predicted_exit_time", 0.0)),
        exit_entropy=float(row.get("exit_entropy", 0.0)),
    )


def _build_risk_params(row: pd.Series) -> RiskParams:
    return RiskParams(
        final_position_size=float(row.get("position_size", row.get("final_position_size", 0.0))),
        leverage_to_apply=float(row.get("position_leverage", row.get("leverage_to_apply", 1.0))),
        volatility_adjustment_flag=bool(row.get("volatility_adjustment_flag", False)),
        risk_mode=RiskMode.NORMAL if str(row.get("risk_mode", "normal")).lower() != "conservative" else RiskMode.CONSERVATIVE,
        max_fill_spread_bps=float(row.get("max_fill_spread_bps", 10.0)),
        aggressiveness=float(row.get("size_volatility_multiplier", 0.5)),
    )


def _build_regime(row: pd.Series) -> RegimeInfo:
    vol_reg = str(row.get("volatility_regime", "normal")).lower()
    liq_reg = str(row.get("liquidity_regime", "average")).lower()
    trend_flag = str(row.get("regime_chop_mask", "trend"))
    return RegimeInfo(
        regime_final=str(row.get("regime_final", row.get("regime_label_fused", "unknown"))),
        volatility_regime=VolatilityRegime.HIGH if "expansion" in vol_reg else VolatilityRegime.NORMAL,
        liquidity_regime=LiquidityRegime.POOR if "drought" in liq_reg else LiquidityRegime.AVERAGE,
        chop_trend_flag=TrendChopFlag.CHOP if "chop" in trend_flag else TrendChopFlag.TREND,
        ignition_flag=bool(row.get("ignition_flag", False)),
        mean_reversion_flag=bool(row.get("mean_reversion_flag", False)),
        regime_confidence=float(row.get("regime_confidence", 0.0)),
    )


def _build_market_snapshot(row: pd.Series) -> MarketSnapshot:
    mid = float(row.get("close", row.get("tp_price", 0.0)))
    spread = float(row.get("spread", 0.0))
    if spread <= 0 and mid > 0:
        spread = mid * 0.0001  # 1 bps default
    best_bid = mid - spread / 2
    best_ask = mid + spread / 2
    return MarketSnapshot(
        ts_ms=int(row.get("open_time_ms", 0)),
        best_bid=best_bid,
        best_ask=best_ask,
        mid_price=mid,
        spread=spread,
        spread_bps=spread / mid * 1e4 if mid else 0.0,
        bids=[],
        asks=[],
        volatility_lookback_sec=float(row.get("volatility_lookback_sec", 60)),
        volatility_bps=float(row.get("volatility_bps", row.get("rv_5m", 0.0))) * 1e4 if mid else 0.0,
        depth_imbalance=float(row.get("depth_imbalance", 0.0)),
    )


def _plan_to_row(plan, open_time_ms: Any) -> Dict[str, Any]:
    return {
        "open_time_ms": open_time_ms,
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
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"exit_with_sizing parquet not found: {INPUT_PATH}")

    logging.info("Loading sized trades (live): %s", INPUT_PATH)
    trades = pd.read_parquet(INPUT_PATH)
    _log_mem(trades, "[Exec] Loaded sized trades")

    engine = ExecutionEngine(ExecutionConfig())

    plans: List[Dict[str, Any]] = []
    for _, row in trades.iterrows():
        entry = _build_entry_signal(row)
        exit_plan = _build_exit_plan(row)
        risk = _build_risk_params(row)
        regime = _build_regime(row)
        safety = SafetyStatus(allowed=bool(row.get("safety_allowed", True)))
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
        plans.append(_plan_to_row(plan, row.get("open_time_ms")))

    plans_df = pd.DataFrame(plans)
    _log_mem(plans_df, "[Exec] Execution plans")

    P5_PROCESSED.mkdir(parents=True, exist_ok=True)
    plans_df.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Saved execution plans (live) -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
