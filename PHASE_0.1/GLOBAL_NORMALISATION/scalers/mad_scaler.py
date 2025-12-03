import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class MADScaler:
    def __init__(self, window: int = 1000, clip: float = 6.0):
        self.window = window
        self.clip = clip
        self.last_median: Optional[float] = None
        self.last_mad: Optional[float] = None

    @staticmethod
    def _rolling_mad(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=1).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True
        )

    def fit(self, series: pd.Series) -> "MADScaler":
        median = series.rolling(self.window, min_periods=1).median().shift(1)
        mad = self._rolling_mad(series, self.window).shift(1)
        self.last_median = float(median.dropna().iloc[-1]) if not median.dropna().empty else None
        self.last_mad = float(mad.dropna().iloc[-1]) if not mad.dropna().empty else None
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        median = series.rolling(self.window, min_periods=1).median().shift(1)
        mad = self._rolling_mad(series, self.window).shift(1)
        denom = mad * 1.4826
        denom = denom.replace(0, np.nan)
        scaled = (series - median) / denom
        scaled = scaled.clip(lower=-self.clip, upper=self.clip)
        scaled = scaled.ffill().fillna(0.0)
        return scaled

    def fit_transform(self, series: pd.Series) -> pd.Series:
        self.fit(series)
        return self.transform(series)

    def to_params(self) -> Dict[str, Any]:
        return {
            "type": "mad",
            "window": self.window,
            "clip": self.clip,
            "last_median": self.last_median,
            "last_mad": self.last_mad,
        }

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "MADScaler":
        scaler = cls(window=params.get("window", 1000), clip=params.get("clip", 6.0))
        scaler.last_median = params.get("last_median")
        scaler.last_mad = params.get("last_mad")
        return scaler

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_params(), f)


__all__ = ["MADScaler"]
