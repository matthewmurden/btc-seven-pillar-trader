"""Unit tests for TradeSimulator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.trade_event import TradeEvent  # noqa: E402
from backtesting.trade_simulator import TradeSimulator  # noqa: E402
from backtesting.config import BacktestConfig  # noqa: E402


def prices_df():
    return pd.DataFrame(
        {
            "high": [100, 102, 101, 99],
            "low": [99, 100, 98, 97],
            "close": [100, 101, 99, 98],
            "timestamp": pd.date_range("2020-01-01", periods=4, freq="T"),
            "valid_exit_signal": [1, 1, 1, 1],
        }
    )


def test_tp_hit_first():
    cfg = BacktestConfig()
    sim = TradeSimulator(cfg)
    event = TradeEvent(entry_idx=0, entry_price=100, direction="long", tp_pct=0.015, sl_pct=0.02, expected_exit_time=3)
    res = sim.simulate(event, prices_df(), None)
    assert res.exit_reason == "TP"
    assert np.isclose(res.pnl_pct, res.tp_pct - 2 * cfg.taker_fee - cfg.slippage)
    assert res.bars_to_exit == 1


def test_sl_hit_first():
    cfg = BacktestConfig()
    sim = TradeSimulator(cfg)
    event = TradeEvent(entry_idx=0, entry_price=100, direction="long", tp_pct=0.05, sl_pct=0.01, expected_exit_time=3)
    res = sim.simulate(event, prices_df(), None)
    assert res.exit_reason == "SL"
    assert np.isclose(res.pnl_pct, -event.sl_pct - 2 * cfg.taker_fee - cfg.slippage)
    assert res.bars_to_exit == 3 or res.bars_to_exit == 2


def test_timeout():
    cfg = BacktestConfig()
    sim = TradeSimulator(cfg)
    event = TradeEvent(entry_idx=0, entry_price=100, direction="long", tp_pct=0.2, sl_pct=0.2, expected_exit_time=2)
    res = sim.simulate(event, prices_df(), None)
    assert res.exit_reason == "timeout"
    assert res.bars_to_exit == 2


def test_invalidation():
    cfg = BacktestConfig()
    sim = TradeSimulator(cfg)
    df = prices_df()
    df.loc[1, "valid_exit_signal"] = 0
    event = TradeEvent(entry_idx=0, entry_price=100, direction="long", tp_pct=0.2, sl_pct=0.2, expected_exit_time=3)
    res = sim.simulate(event, df, df["valid_exit_signal"])
    assert res.exit_reason == "invalidated"
    assert res.bars_to_exit == 1


def test_mae_mfe_tracking():
    cfg = BacktestConfig()
    sim = TradeSimulator(cfg)
    event = TradeEvent(entry_idx=0, entry_price=100, direction="long", tp_pct=0.2, sl_pct=0.2, expected_exit_time=3)
    res = sim.simulate(event, prices_df(), None)
    assert res.mfe >= 0
    assert res.mae <= 0
