"""Tests for PortfolioSimulator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.trade_event import TradeEvent  # noqa: E402
from backtesting.portfolio_simulator import PortfolioSimulator  # noqa: E402
from backtesting.config import BacktestConfig  # noqa: E402


def simple_prices():
    return pd.DataFrame(
        {
            "high": [100, 102, 104, 103, 101],
            "low": [99, 99, 100, 100, 99],
            "close": [100, 101, 103, 101, 100],
        }
    )


def test_no_overlap_behaviour():
    cfg = BacktestConfig(allow_overlapping_trades=False)
    portfolio = PortfolioSimulator(cfg)
    prices = simple_prices()
    events = [
        TradeEvent(entry_idx=0, entry_price=100, direction="long", tp_pct=0.02, sl_pct=0.01, expected_exit_time=3),
        TradeEvent(entry_idx=1, entry_price=101, direction="short", tp_pct=0.02, sl_pct=0.01, expected_exit_time=3),
    ]
    trades = portfolio.run(events, prices)
    assert len(trades) == 1


def test_stats_computation():
    cfg = BacktestConfig()
    portfolio = PortfolioSimulator(cfg)
    prices = simple_prices()
    events = [
        TradeEvent(entry_idx=0, entry_price=100, direction="long", tp_pct=0.02, sl_pct=0.01, expected_exit_time=3),
    ]
    trades = portfolio.run(events, prices)
    stats = portfolio.stats(trades)
    assert "win_rate" in stats
    assert "max_drawdown" in stats
    assert stats["max_drawdown"] >= -1

