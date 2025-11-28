"""
Deterministic, vectorized backtester for pillar 1 entry model.
Operates on DataFrames containing close prices and final_signal (1 long, -1 short, 0 flat).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    tp_pct: float = 0.003
    sl_pct: float = 0.003
    max_horizon: int = 50
    price_col: str = "close"
    signal_col: str = "final_signal"
    high_col: str = "high"
    low_col: str = "low"
    open_col: str = "open"                  # added: correct entry
    equity_start: float = 1.0
    backtest_results_path: Path = Path("data/backtests/backtest_results.parquet")
    trades_path: Path = Path("data/backtests/trades.parquet")


class Backtester:
    def __init__(self, df: pd.DataFrame, config: BacktestConfig = BacktestConfig()):
        self.df = df.reset_index(drop=True).copy()
        self.config = config
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        required = [
            self.config.price_col,
            self.config.signal_col,
            self.config.high_col,
            self.config.low_col,
        ]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        if not np.isfinite(self.df[self.config.price_col]).all():
            raise ValueError("Price column contains non-finite values")

        # Fill missing signals
        self.df[self.config.signal_col] = self.df[self.config.signal_col].fillna(0).astype(int)

        # Use close as open fallback
        if self.config.open_col not in self.df.columns:
            self.df[self.config.open_col] = self.df[self.config.price_col]

    def _simulate_trades(self) -> pd.DataFrame:
        price = self.df[self.config.price_col].to_numpy()
        openp = self.df[self.config.open_col].to_numpy()
        high = self.df[self.config.high_col].to_numpy()
        low = self.df[self.config.low_col].to_numpy()
        signal = self.df[self.config.signal_col].to_numpy()

        # Only enter when signal transitions from 0 → non-zero
        entries = np.where((signal != 0) & (np.roll(signal, 1) == 0))[0]
        entries = entries[entries > 0]

        trades = []
        n = len(self.df)
        last_exit = -1

        for idx in entries:
            if idx <= last_exit:
                continue  # skip overlapping trades

            direction = signal[idx]
            entry_idx = idx + 1
            if entry_idx >= n:
                continue

            # Entry price = next bar open (correct)
            entry_price = openp[entry_idx]

            # True directional TP/SL
            if direction > 0:
                tp = entry_price * (1 + self.config.tp_pct)
                sl = entry_price * (1 - self.config.sl_pct)
            else:
                tp = entry_price * (1 - self.config.tp_pct)
                sl = entry_price * (1 + self.config.sl_pct)

            exit_idx = None
            exit_price = None
            reason = "timeout"

            horizon_end = min(entry_idx + self.config.max_horizon, n - 1)
            seg_high = high[entry_idx + 1 : horizon_end + 1]
            seg_low = low[entry_idx + 1 : horizon_end + 1]

            if direction > 0:
                tp_hit = np.where(seg_high >= tp)[0]
                sl_hit = np.where(seg_low <= sl)[0]
            else:
                tp_hit = np.where(seg_low <= tp)[0]
                sl_hit = np.where(seg_high >= sl)[0]

            if len(tp_hit) > 0 and len(sl_hit) > 0:
                if tp_hit[0] < sl_hit[0]:
                    exit_idx = entry_idx + 1 + tp_hit[0]
                    exit_price = tp
                    reason = "tp"
                else:
                    exit_idx = entry_idx + 1 + sl_hit[0]
                    exit_price = sl
                    reason = "sl"
            elif len(tp_hit) > 0:
                exit_idx = entry_idx + 1 + tp_hit[0]
                exit_price = tp
                reason = "tp"
            elif len(sl_hit) > 0:
                exit_idx = entry_idx + 1 + sl_hit[0]
                exit_price = sl
                reason = "sl"
            else:
                exit_idx = horizon_end
                exit_price = price[exit_idx]
                reason = "timeout"

            last_exit = exit_idx

            pnl_pct = (exit_price - entry_price) / entry_price * (1 if direction > 0 else -1)

            trades.append({
                "entry_index": entry_idx,
                "exit_index": exit_idx,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "direction": direction,
                "pnl_pct": pnl_pct,
                "bars_held": exit_idx - entry_idx,
                "reason": reason,
            })

        return pd.DataFrame(trades)

    def compute_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        if trades.empty:
            return {
                "winrate": 0.0,
                "sharpe": 0.0,
                "expectancy": 0.0,
                "max_dd": 0.0,
                "total_return": 0.0,
            }

        wins = trades["pnl_pct"] > 0
        winrate = wins.mean()

        expectancy = trades["pnl_pct"].mean()
        equity = (1 + trades["pnl_pct"]).cumprod()
        returns = trades["pnl_pct"]

        sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min()
        total_return = equity.iloc[-1] - 1

        return {
            "winrate": float(winrate),
            "sharpe": float(sharpe),
            "expectancy": float(expectancy),
            "max_dd": float(max_dd),
            "total_return": float(total_return),
        }

    def run(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        trades = self._simulate_trades()

        if trades.empty:
            equity_curve = pd.Series([self.config.equity_start])
            dd_curve = pd.Series([0.0])
        else:
            equity_curve = self.config.equity_start * (1 + trades["pnl_pct"]).cumprod()
            peak = equity_curve.cummax()
            dd_curve = (equity_curve - peak) / peak

        return trades, equity_curve, dd_curve


def main():
    data_path = Path("data/processed/features_with_final_signal.parquet")
    if not data_path.exists():
        raise FileNotFoundError("Run final decision layer before backtesting.")

    df = pd.read_parquet(data_path).ffill().bfill()

    if "final_signal" not in df.columns:
        raise ValueError("final_signal missing; ensure final decision layer has been applied.")

    bt = Backtester(df)
    trades, eq, dd = bt.run()
    metrics = bt.compute_metrics(trades)

    trades.to_parquet(bt.config.trades_path, index=False)
    pd.DataFrame({"equity_curve": eq, "drawdown": dd}).to_parquet(
        bt.config.backtest_results_path, index=False
    )

    print("\nBacktest complete.")
    print("Total trades:", len(trades))
    print("Winrate:", f"{metrics['winrate']:.2%}")
    print("Expectancy:", f"{metrics['expectancy']:.4f}")
    print("Sharpe:", f"{metrics['sharpe']:.3f}")
    print("Max DD:", f"{metrics['max_dd']:.2%}")
    print("Total return:", f"{metrics['total_return']:.2%}")
    print("\nSample trades:")
    print(trades.head())


if __name__ == "__main__":
    main()
