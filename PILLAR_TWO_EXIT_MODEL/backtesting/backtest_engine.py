"""Backtest engine for exit decisions."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def simulate_trade(entry_price: float, direction: int, tp_pct: float, sl_pct: float, close_series: pd.Series) -> Tuple[str, float, int]:
    """
    Simulate a single trade given future close prices.

    Returns outcome ("TP" or "SL"), pnl_pct, bars_to_exit.
    """
    if direction == 0 or tp_pct <= 0 or sl_pct <= 0:
        return "SL", 0.0, 0
    if close_series.empty:
        return "SL", 0.0, 0

    if direction > 0:
        tp_level = entry_price * (1 + tp_pct)
        sl_level = entry_price * (1 - sl_pct)
    else:
        tp_level = entry_price * (1 - tp_pct)
        sl_level = entry_price * (1 + sl_pct)

    for idx, price in enumerate(close_series, start=1):
        if direction > 0:
            if price >= tp_level:
                return "TP", tp_pct, idx
            if price <= sl_level:
                return "SL", -sl_pct, idx
        else:
            if price <= tp_level:
                return "TP", tp_pct, idx
            if price >= sl_level:
                return "SL", -sl_pct, idx
    # default to last price outcome
    final_ret = (close_series.iloc[-1] - entry_price) / entry_price * direction
    return ("TP" if final_ret > 0 else "SL"), final_ret, len(close_series)


def run_backtest(decisions_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run backtest over decisions_df using future close prices from raw_df.
    Assumes matching index alignment by position.
    """
    results = []
    closes = raw_df["close"].reset_index(drop=True)
    for i, row in decisions_df.iterrows():
        future_close = closes[i + 1 : i + 301]  # up to 300 bars ahead
        if future_close.empty:
            results.append(
                {
                    "outcome": "SL",
                    "pnl_pct": 0.0,
                    "bars_to_exit": 0,
                    "direction": row.get("final_direction", 0),
                    "entry_price": row.get("entry_price", np.nan),
                }
            )
            continue
        outcome, pnl, bars = simulate_trade(
            entry_price=row["entry_price"],
            direction=row["final_direction"],
            tp_pct=row["final_tp_pct"],
            sl_pct=row["final_sl_pct"],
            close_series=future_close,
        )
        results.append(
            {
                "outcome": outcome,
                "pnl_pct": pnl,
                "bars_to_exit": bars,
                "direction": row["final_direction"],
                "entry_price": row["entry_price"],
            }
        )

    out = pd.DataFrame(results)
    return out
