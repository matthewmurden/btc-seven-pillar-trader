"""Trade simulator for executing TradeEvents on OHLC data."""

from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd

from .trade_event import TradeEvent
from .config import BacktestConfig


class TradeSimulator:
    def __init__(self, config: BacktestConfig):
        self.config = config

    def _dir_mult(self, event: TradeEvent) -> int:
        return 1 if event.direction == "long" else -1

    def _apply_fees(self, pnl: float) -> float:
        return pnl - 2 * self.config.taker_fee - self.config.slippage

    def simulate(self, event: TradeEvent, prices: pd.DataFrame, valid_signal: pd.Series | None = None) -> dict:
        dir_mult = self._dir_mult(event)
        entry_idx = event.entry_idx
        future = prices.iloc[entry_idx + 1 : entry_idx + event.expected_exit_time + 1]
        if future.empty:
            exit_price = event.entry_price
            pnl_raw = 0.0
            return {
                "timestamp": event.timestamp_entry,
                "direction": event.direction,
                "entry_price": event.entry_price,
                "exit_price": exit_price,
                "pnl_pct": pnl_raw,
                "holding_bars": 0,
                "outcome": "timeout",
                "mae": 0.0,
                "mfe": 0.0,
            }

        tp_level = event.entry_price * (1 + event.tp_pct * dir_mult)
        sl_level = event.entry_price * (1 - event.sl_pct * dir_mult)

        mfe = -np.inf
        mae = np.inf

        for offset, (_, bar) in enumerate(future.iterrows(), start=1):
            high = bar["high"]
            low = bar["low"]
            # compute returns relative to entry
            ret_high = (high / event.entry_price - 1) * dir_mult
            ret_low = (low / event.entry_price - 1) * dir_mult
            mfe = max(mfe, ret_high)
            mae = min(mae, ret_low)

            # invalidation
            if valid_signal is not None:
                sig_val = valid_signal.iloc[entry_idx + offset]
                if sig_val == 0:
                    exit_ret = (bar["close"] / event.entry_price - 1) * dir_mult
                    exit_price = bar["close"]
                    pnl_raw = exit_ret
                    return {
                        "timestamp": event.timestamp_entry,
                        "direction": event.direction,
                        "entry_price": event.entry_price,
                        "exit_price": exit_price,
                        "pnl_pct": pnl_raw,
                        "holding_bars": offset,
                        "outcome": "timeout",
                        "mae": mae,
                        "mfe": mfe,
                    }

            # TP/SL check based on wick first-touch
            if dir_mult > 0:
                if high >= tp_level:
                    exit_price = tp_level
                    pnl_raw = (exit_price - event.entry_price) / event.entry_price * dir_mult
                    return {
                        "timestamp": event.timestamp_entry,
                        "direction": event.direction,
                        "entry_price": event.entry_price,
                        "exit_price": exit_price,
                        "pnl_pct": pnl_raw,
                        "holding_bars": offset,
                        "outcome": "TP",
                        "mae": mae,
                        "mfe": mfe,
                    }
                if low <= sl_level:
                    exit_price = sl_level
                    pnl_raw = (exit_price - event.entry_price) / event.entry_price * dir_mult
                    return {
                        "timestamp": event.timestamp_entry,
                        "direction": event.direction,
                        "entry_price": event.entry_price,
                        "exit_price": exit_price,
                        "pnl_pct": pnl_raw,
                        "holding_bars": offset,
                        "outcome": "SL",
                        "mae": mae,
                        "mfe": mfe,
                    }
            else:
                if low <= tp_level:
                    exit_price = tp_level
                    pnl_raw = (exit_price - event.entry_price) / event.entry_price * dir_mult
                    return {
                        "timestamp": event.timestamp_entry,
                        "direction": event.direction,
                        "entry_price": event.entry_price,
                        "exit_price": exit_price,
                        "pnl_pct": pnl_raw,
                        "holding_bars": offset,
                        "outcome": "TP",
                        "mae": mae,
                        "mfe": mfe,
                    }
                if high >= sl_level:
                    exit_price = sl_level
                    pnl_raw = (exit_price - event.entry_price) / event.entry_price * dir_mult
                    return {
                        "timestamp": event.timestamp_entry,
                        "direction": event.direction,
                        "entry_price": event.entry_price,
                        "exit_price": exit_price,
                        "pnl_pct": pnl_raw,
                        "holding_bars": offset,
                        "outcome": "SL",
                        "mae": mae,
                        "mfe": mfe,
                    }

        # timeout
        last_bar = future.iloc[-1]
        exit_price = last_bar["close"]
        pnl_raw = (exit_price / event.entry_price - 1) * dir_mult
        return {
            "timestamp": event.timestamp_entry,
            "direction": event.direction,
            "entry_price": event.entry_price,
            "exit_price": exit_price,
            "pnl_pct": pnl_raw,
            "holding_bars": len(future),
            "outcome": "timeout",
            "mae": mae,
            "mfe": mfe,
        }
