"""Backtest runner tying together prediction, pipeline, and metrics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_models.model_predict_testset import generate_test_predictions
from backtesting.pipeline_test_exit import run_exit_pipeline_test
from backtesting.load_entry_signals import load_entry_signals_for_test
from backtesting.backtest_engine import run_backtest
from backtesting.performance_metrics import compute_all_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
EXIT_DECISIONS_PATH = DATA_DIR / "exit_decisions_test.parquet"
RAW_PATH = DATA_DIR / "raw" / "exit_raw.parquet"


def run_backtest_pipeline():
    generate_test_predictions()
    run_exit_pipeline_test()
    decisions = pd.read_parquet(EXIT_DECISIONS_PATH)

    # Align entry signals (overwrite with real entries)
    entry_signals = load_entry_signals_for_test(len(decisions))
    if isinstance(entry_signals, pd.DataFrame):
        if "entry_signal" in entry_signals:
            decisions["entry_signal"] = entry_signals["entry_signal"].values
        else:
            decisions["entry_signal"] = 1
        if "entry_price" in entry_signals:
            decisions["entry_price"] = entry_signals["entry_price"].values
        if "direction" in entry_signals:
            decisions["final_direction"] = entry_signals["direction"].values
    else:
        decisions["entry_signal"] = 1

    raw_df = pd.read_parquet(RAW_PATH) if RAW_PATH.exists() else pd.DataFrame({"close": decisions["entry_price"]})
    backtest_df = run_backtest(decisions, raw_df)
    backtest_df.to_parquet(DATA_DIR / "backtest_results.parquet")

    metrics = compute_all_metrics(backtest_df)

    print("FINAL PERFORMANCE REPORT")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Expectancy: {metrics['expectancy']:.4f}")
    print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Avg Holding Time: {metrics['average_holding_time']:.2f} bars")
    print(f"Outcome Distribution: {metrics['outcome_distribution']}")
    return backtest_df, metrics


if __name__ == "__main__":
    run_backtest_pipeline()
