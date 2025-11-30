"""Exit model executor: merges entry signals with exit predictions to create trade events."""

from __future__ import annotations

from typing import List
import pandas as pd

from .trade_event import TradeEvent
from .config import BacktestConfig


class ExitModelExecutor:
    """Combine Pillar 1 entries with Pillar 2 exits to emit TradeEvents."""

    def __init__(self, config: BacktestConfig):
        self.config = config

    def _merge(self, entry_df: pd.DataFrame, exit_df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" not in entry_df.columns or "timestamp" not in exit_df.columns:
            raise ValueError("Both entry_df and exit_df must contain 'timestamp' column")

        entry_df = entry_df.copy().drop_duplicates(subset=["timestamp"], keep="last")
        exit_df = exit_df.copy().drop_duplicates(subset=["timestamp"], keep="last")
        entry_df["timestamp"] = pd.to_datetime(entry_df["timestamp"])
        exit_df["timestamp"] = pd.to_datetime(exit_df["timestamp"])

        merged = pd.merge_asof(
            entry_df.sort_values("timestamp"),
            exit_df.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("1min"),
        )

        merged = merged.dropna(subset=["timestamp"]).reset_index(drop=True)
        if "entry_price_x" in merged.columns:
            merged = merged.rename(columns={"entry_price_x": "entry_price"})
        if "entry_price_y" in merged.columns:
            merged = merged.drop(columns=["entry_price_y"])
        return merged

    def build_trade_events(self, entry_df: pd.DataFrame, exit_df: pd.DataFrame) -> List[TradeEvent]:
        merged = self._merge(entry_df, exit_df)
        # standardise entry price column
        if "entry_price_x" in merged.columns:
            merged = merged.rename(columns={"entry_price_x": "entry_price"})
        if "entry_price_y" in merged.columns:
            merged = merged.drop(columns=["entry_price_y"])
        events: List[TradeEvent] = []
        for idx, row in merged.iterrows():
            if row.get("exit_quality_score", row.get("exit_quality_calibrated", 0)) < self.config.min_exit_quality_threshold:
                continue
            if abs(row.get("mae_pred", 0)) > self.config.mae_pred_cutoff:
                continue
            if row.get("entry_signal", 0) != 1:
                continue
            if row.get("entry_probability", 0) < self.config.entry_probability_threshold:
                continue
            if "regime_state" in row and row["regime_state"] in self.config.invalid_regimes:
                continue
            if row.get("valid_exit_signal", 1) == 0:
                continue
            tp_pct = row.get("recommended_tp_pct", row.get("optimal_tp_pred", 0))
            sl_pct = row.get("recommended_sl_pct", row.get("optimal_sl_pred", 0))
            if tp_pct is None or tp_pct <= 0.001:
                continue
            if sl_pct is None or sl_pct <= 0.001:
                continue
            direction_val = row.get("direction", row.get("pillar_1_direction", 1))
            direction = "long" if direction_val in [1, "long", True] else "short"
            expected_exit = int(row.get("expected_exit_time", row.get("exit_time_pred", self.config.max_holding_bars)))
            expected_exit = min(expected_exit, self.config.max_holding_bars)
            events.append(
                TradeEvent(
                    entry_idx=idx,
                    entry_price=float(
                        row.get("entry_price", row.get("entry_price_x", row.get("entry_price_y")))
                    ),
                    direction=direction,
                    tp_pct=float(tp_pct),
                    sl_pct=float(sl_pct),
                    expected_exit_time=expected_exit,
                    timestamp_entry=row["timestamp"] if "timestamp" in row else None,
                )
            )
        return events
