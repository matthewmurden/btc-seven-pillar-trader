"""Backtest configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BacktestConfig:
    max_holding_bars: int = 300
    allow_overlapping_trades: bool = False
    slippage: float = 0.0
    taker_fee: float = 0.0004
    maker_fee: float = 0.0002
    min_exit_quality_threshold: float = 0.45
    mae_pred_cutoff: float = 0.05
    scaling_mode: str = "percent_returns"
    entry_probability_threshold: float = 0.6
    invalid_regimes: tuple = (-1,)
