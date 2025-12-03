import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class ZScoreScaler:
    def __init__(self, window: int = 1000, clip: float = 5.0):
        self.window = window
        self.clip = clip
        self.last_mean: Optional[float] = None
        self.last_std: Optional[float] = None

    def fit(self, series: pd.Series) -> "ZScoreScaler":
        mean = series.rolling(self.window, min_periods=1).mean().shift(1)
        std = series.rolling(self.window, min_periods=1).std(ddof=0).shift(1)
        self.last_mean = float(mean.dropna().iloc[-1]) if not mean.dropna().empty else None
        self.last_std = float(std.dropna().iloc[-1]) if not std.dropna().empty else None
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        mean = series.rolling(self.window, min_periods=1).mean().shift(1)
        std = series.rolling(self.window, min_periods=1).std(ddof=0).shift(1)
        std = std.replace(0, np.nan)
        scaled = (series - mean) / std
        scaled = scaled.clip(lower=-self.clip, upper=self.clip)
        scaled = scaled.ffill().fillna(0.0)
        return scaled

    def fit_transform(self, series: pd.Series) -> pd.Series:
        self.fit(series)
        return self.transform(series)

    def to_params(self) -> Dict[str, Any]:
        return {
            "type": "zscore",
            "window": self.window,
            "clip": self.clip,
            "last_mean": self.last_mean,
            "last_std": self.last_std,
        }

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "ZScoreScaler":
        scaler = cls(window=params.get("window", 1000), clip=params.get("clip", 5.0))
        scaler.last_mean = params.get("last_mean")
        scaler.last_std = params.get("last_std")
        return scaler

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_params(), f)


__all__ = ["ZScoreScaler"]
