"""Portfolio simulator to execute sequential trades."""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd

from .trade_event import TradeEvent
from .trade_simulator import TradeSimulator
from .config import BacktestConfig


class PortfolioSimulator:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.trade_simulator = TradeSimulator(config)

    def run(self, events: List[TradeEvent], prices: pd.DataFrame, valid_signal: pd.Series | None = None) -> pd.DataFrame:
        executed: List[TradeEvent] = []
        last_exit_idx = -1
        for event in sorted(events, key=lambda e: e.entry_idx):
            if not self.config.allow_overlapping_trades and event.entry_idx <= last_exit_idx:
                continue
            sim_event = self.trade_simulator.simulate(event, prices, valid_signal)
            last_exit_idx = event.entry_idx + sim_event.bars_to_exit
            executed.append(sim_event)
        return pd.DataFrame([e.__dict__ for e in executed])

    @staticmethod
    def equity_curve(trades_df: pd.DataFrame) -> pd.Series:
        if "pnl_pct" not in trades_df.columns:
            raise ValueError("Missing pnl_pct in trades dataframe")
        equity = (1 + trades_df["pnl_pct"]).cumprod()
        return equity

    @staticmethod
    def max_drawdown(equity: pd.Series) -> float:
        peak = equity.cummax()
        dd = (equity - peak) / peak
        return dd.min() if len(dd) else 0.0

    @staticmethod
    def stats(trades_df: pd.DataFrame) -> dict:
        equity = PortfolioSimulator.equity_curve(trades_df)
        wins = (trades_df["pnl_pct"] > 0).sum()
        win_rate = wins / len(trades_df) if len(trades_df) else 0
        expectancy = trades_df["pnl_pct"].mean() if len(trades_df) else 0
        sharpe = 0.0
        if len(trades_df) > 1 and trades_df["pnl_pct"].std(ddof=1) > 0:
            sharpe = trades_df["pnl_pct"].mean() / trades_df["pnl_pct"].std(ddof=1) * np.sqrt(252)
        return {
            "win_rate": win_rate,
            "expectancy": expectancy,
            "sharpe": sharpe,
            "max_drawdown": PortfolioSimulator.max_drawdown(equity),
            "equity_curve": equity,
        }
