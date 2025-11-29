"""Tests for backtesting framework."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.backtest_engine import simulate_trade, run_backtest  # noqa: E402
from backtesting.performance_metrics import compute_all_metrics  # noqa: E402


def test_simulate_trade_tp_sl_detection():
    entry_price = 100
    tp_pct = 0.02
    sl_pct = 0.01
    close_series = pd.Series([101, 102, 99, 98])
    outcome, pnl, bars = simulate_trade(entry_price, 1, tp_pct, sl_pct, close_series)
    assert outcome == "TP"
    assert np.isclose(pnl, tp_pct)
    assert bars == 2


def test_simulate_trade_sl_first():
    entry_price = 100
    tp_pct = 0.05
    sl_pct = 0.01
    close_series = pd.Series([99, 100, 101])
    outcome, pnl, bars = simulate_trade(entry_price, 1, tp_pct, sl_pct, close_series)
    assert outcome == "SL"
    assert np.isclose(pnl, -sl_pct)
    assert bars == 1


def test_run_backtest_returns_df():
    decisions = pd.DataFrame(
        {
            "entry_price": [100, 100],
            "final_direction": [1, -1],
            "final_tp_pct": [0.02, 0.02],
            "final_sl_pct": [0.01, 0.01],
        }
    )
    raw_df = pd.DataFrame({"close": [100, 102, 99, 101]})
    out = run_backtest(decisions, raw_df)
    assert isinstance(out, pd.DataFrame)
    assert {"outcome", "pnl_pct", "bars_to_exit"}.issubset(out.columns)
    assert not out.isna().any().any()


def test_metrics_keys():
    df = pd.DataFrame(
        {
            "pnl_pct": [0.01, -0.005, 0.02],
            "bars_to_exit": [5, 10, 7],
            "outcome": ["TP", "SL", "TP"],
        }
    )
    metrics = compute_all_metrics(df)
    for key in [
        "win_rate",
        "expectancy",
        "max_drawdown",
        "sharpe_ratio",
        "average_holding_time",
        "outcome_distribution",
        "regime_performance",
    ]:
        assert key in metrics


if __name__ == "__main__":
    test_simulate_trade_tp_sl_detection()
    test_simulate_trade_sl_first()
    test_run_backtest_returns_df()
    test_metrics_keys()
