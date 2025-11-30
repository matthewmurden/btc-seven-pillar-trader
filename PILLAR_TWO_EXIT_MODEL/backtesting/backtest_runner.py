"""Backtest runner combining entry/exits and simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ml_models.model_predict_testset import generate_test_predictions
from backtesting.pipeline_test_exit import run_exit_pipeline_test
from backtesting.load_entry_signals import load_entry_signals_for_test
from backtesting.config import BacktestConfig
from backtesting.executor import ExitModelExecutor
from backtesting.portfolio_simulator import PortfolioSimulator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
EXIT_DECISIONS_PATH = DATA_DIR / "exit_decisions_test.parquet"
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "exit_raw.parquet"
EXIT_FEATURES_WITH_TS = DATA_DIR / "features_exit_with_timestamp.parquet"


def run_backtest_pipeline(
    entry_df: Optional[pd.DataFrame] = None,
    exit_df: Optional[pd.DataFrame] = None,
    raw_df: Optional[pd.DataFrame] = None,
    config: Optional[BacktestConfig] = None,
) -> tuple[pd.DataFrame, dict]:
    cfg = config or BacktestConfig()
    if EXIT_FEATURES_WITH_TS.exists():
        exit_df = pd.read_parquet(EXIT_FEATURES_WITH_TS)
    else:
        try:
            from data_pipeline.align_exit_with_pillar1 import align_exit_with_pillar1
            exit_df = pd.read_parquet(align_exit_with_pillar1())
        except Exception:
            exit_df = pd.DataFrame()
    if exit_df.empty:
        raise FileNotFoundError(f"Exit features not found at {EXIT_FEATURES_WITH_TS}")
    exit_df = exit_df.sort_values("timestamp").reset_index(drop=True)
    raw_df = raw_df if raw_df is not None else pd.read_parquet(RAW_PATH)

    if entry_df is None:
        pillar1_path = PROJECT_ROOT.parent / "PILLAR_ONE_ENTRY_MODEL" / "data" / "processed" / "pillar1_export.parquet"
        entry_df = pd.read_parquet(pillar1_path)
    entry_df = entry_df[entry_df["timestamp"].isin(exit_df["timestamp"])].copy()

    executor = ExitModelExecutor(cfg)
    events = executor.build_trade_events(entry_df, exit_df)

    # Build validity series
    valid_signal = exit_df["valid_exit_signal"] if "valid_exit_signal" in exit_df.columns else None

    prices = raw_df[["high", "low", "close"]].copy()
    if "timestamp" in raw_df.columns:
        prices["timestamp"] = pd.to_datetime(raw_df["timestamp"])
    else:
        prices["timestamp"] = exit_df["timestamp"]

    portfolio = PortfolioSimulator(cfg)
    trades = []
    for event in events:
        trade = portfolio.trade_simulator.simulate(event, prices, valid_signal)
        if trade is not None:
            trades.append(trade)
    trades_df = pd.DataFrame(trades, columns=["timestamp", "direction", "entry_price", "exit_price", "pnl_pct", "holding_bars", "outcome", "mae", "mfe"])
    stats = portfolio.stats(trades_df)

    # Save report
    report_path = PROJECT_ROOT / "reports"
    report_path.mkdir(parents=True, exist_ok=True)
    trades_df.to_parquet(report_path / "backtest_trades.parquet")
    stats_out = {k: v for k, v in stats.items() if k != "equity_curve"}
    (report_path / "backtest_report.json").write_text(pd.Series(stats_out).to_json())

    print("FINAL PERFORMANCE REPORT")
    print(f"Win Rate: {stats['win_rate']:.2%}")
    print(f"Expectancy: {stats['expectancy']:.4f}")
    print(f"Sharpe: {stats['sharpe']:.2f}")
    print(f"Max Drawdown: {stats['max_drawdown']:.2%}")
    return trades_df, stats


if __name__ == "__main__":
    run_backtest_pipeline()
