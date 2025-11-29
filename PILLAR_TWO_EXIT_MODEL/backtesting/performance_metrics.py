"""Performance metrics for backtesting results."""

from __future__ import annotations

import numpy as np
import pandas as pd


def win_rate(df: pd.DataFrame) -> float:
    wins = (df["pnl_pct"] > 0).sum()
    return wins / len(df) if len(df) > 0 else 0.0


def expectancy(df: pd.DataFrame) -> float:
    return df["pnl_pct"].mean() if len(df) > 0 else 0.0


def max_drawdown(df: pd.DataFrame) -> float:
    equity = (1 + df["pnl_pct"]).cumprod()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return drawdown.min() if len(drawdown) > 0 else 0.0


def sharpe_ratio(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 0.0
    ret = df["pnl_pct"]
    return ret.mean() / (ret.std(ddof=1) + 1e-12) * np.sqrt(252)


def average_holding_time(df: pd.DataFrame) -> float:
    return df["bars_to_exit"].mean() if len(df) > 0 else 0.0


def outcome_distribution(df: pd.DataFrame) -> dict:
    return df["outcome"].value_counts().to_dict()


def regime_performance(df: pd.DataFrame, regime_series: pd.Series | None = None) -> dict:
    if regime_series is None or len(regime_series) != len(df):
        return {}
    res = {}
    for r in regime_series.unique():
        mask = regime_series == r
        res[int(r)] = df.loc[mask, "pnl_pct"].mean()
    return res


def compute_all_metrics(df: pd.DataFrame, regime_series: pd.Series | None = None) -> dict:
    return {
        "win_rate": win_rate(df),
        "expectancy": expectancy(df),
        "max_drawdown": max_drawdown(df),
        "sharpe_ratio": sharpe_ratio(df),
        "average_holding_time": average_holding_time(df),
        "outcome_distribution": outcome_distribution(df),
        "regime_performance": regime_performance(df, regime_series),
    }
