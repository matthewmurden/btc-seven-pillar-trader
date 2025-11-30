"""Trade event dataclass capturing entry and exit details."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class TradeEvent:
    entry_idx: int
    entry_price: float
    direction: str  # "long" or "short"
    tp_pct: float
    sl_pct: float
    expected_exit_time: int
    timestamp_entry: Optional[pd.Timestamp] = None
    timestamp_exit: Optional[pd.Timestamp] = None
    exit_reason: Optional[str] = None
    mae: float = 0.0
    mfe: float = 0.0
    pnl_pct: float = 0.0
    bars_to_exit: int = 0

