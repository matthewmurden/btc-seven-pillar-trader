"""End-to-end tests for backtest runner with synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.backtest_runner import run_backtest_pipeline  # noqa: E402
from backtesting.config import BacktestConfig  # noqa: E402


def synthetic_data():
    prices = pd.DataFrame(
        {
            "high": [100, 102, 103, 101, 99],
            "low": [99, 100, 101, 98, 97],
            "close": [100, 101, 102, 99, 98],
        }
    )
    exit_df = pd.DataFrame(
        {
            "entry_price": [100, 101],
            "recommended_tp_pct": [0.02, 0.02],
            "recommended_sl_pct": [0.01, 0.01],
            "exit_quality_score": [0.6, 0.6],
            "mae_pred": [-0.01, -0.01],
            "timestamp": pd.date_range("2020-01-01", periods=2, freq="T"),
            "valid_exit_signal": [1, 1],
        }
    )
    entry_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=2, freq="T"),
            "entry_signal": [1, 1],
            "direction": [1, -1],
            "entry_price": [100, 101],
            "entry_confidence": [0.9, 0.9],
            "entry_probability": [0.8, 0.8],
        }
    )
    return prices, exit_df, entry_df


def test_backtest_runner_synthetic():
    prices, exit_df, entry_df = synthetic_data()
    cfg = BacktestConfig()
    trades, stats = run_backtest_pipeline(entry_df=entry_df, exit_df=exit_df, raw_df=prices, config=cfg)
    assert not trades.empty
    assert "win_rate" in stats
    assert stats["max_drawdown"] >= -1


if __name__ == "__main__":
    test_backtest_runner_synthetic()
