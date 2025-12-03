import json
from typing import Any, Dict

import numpy as np
import pandas as pd


class MinMaxScaler:
    def __init__(self, window: int = 500, clip_lower: float = 0.01, clip_upper: float = 0.99):
        self.window = window
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper
        self.last_min: float = 0.0
        self.last_max: float = 1.0

    def fit(self, series: pd.Series) -> "MinMaxScaler":
        rolling_min = series.rolling(self.window, min_periods=1).min().shift(1)
        rolling_max = series.rolling(self.window, min_periods=1).max().shift(1)
        if not rolling_min.dropna().empty:
            self.last_min = float(rolling_min.dropna().iloc[-1])
        if not rolling_max.dropna().empty:
            self.last_max = float(rolling_max.dropna().iloc[-1])
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        rolling_min = series.rolling(self.window, min_periods=1).min().shift(1)
        rolling_max = series.rolling(self.window, min_periods=1).max().shift(1)
        denom = (rolling_max - rolling_min).replace(0, np.nan)
        scaled = (series - rolling_min) / denom
        scaled = scaled.clip(lower=self.clip_lower, upper=self.clip_upper)
        scaled = scaled.ffill().fillna(0.5)
        return scaled

    def fit_transform(self, series: pd.Series) -> pd.Series:
        self.fit(series)
        return self.transform(series)

    def to_params(self) -> Dict[str, Any]:
        return {
            "type": "minmax",
            "window": self.window,
            "clip_lower": self.clip_lower,
            "clip_upper": self.clip_upper,
            "last_min": self.last_min,
            "last_max": self.last_max,
        }

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "MinMaxScaler":
        scaler = cls(
            window=params.get("window", 500),
            clip_lower=params.get("clip_lower", 0.01),
            clip_upper=params.get("clip_upper", 0.99),
        )
        scaler.last_min = params.get("last_min", 0.0)
        scaler.last_max = params.get("last_max", 1.0)
        return scaler

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_params(), f)


__all__ = ["MinMaxScaler"]
