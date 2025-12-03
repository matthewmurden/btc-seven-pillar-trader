import json
from typing import Any, Dict, Optional, Sequence, Type

import numpy as np
import pandas as pd

from scalers.zscore_scaler import ZScoreScaler


class RegimeSplitScaler:
    def __init__(
        self,
        base_scaler_cls: Type = ZScoreScaler,
        regimes: Optional[Sequence[str]] = None,
        window: int = 500,
        clip: float = 5.0,
    ):
        self.base_scaler_cls = base_scaler_cls
        self.regimes = list(regimes) if regimes is not None else ["compression", "normal", "expansion"]
        self.window = window
        self.clip = clip
        self.scalers: Dict[str, Any] = {}

    def fit(self, series: pd.Series, regime_series: pd.Series) -> "RegimeSplitScaler":
        self.scalers = {}
        for regime in self.regimes:
            mask = regime_series == regime
            scaler = self.base_scaler_cls(window=self.window, clip=self.clip)
            scaler.fit(series[mask]) if mask.any() else scaler.fit(series.iloc[0:0])
            self.scalers[regime] = scaler
        return self

    def transform(self, series: pd.Series, regime_series: pd.Series) -> pd.Series:
        result = pd.Series(np.nan, index=series.index, dtype=float)
        default_scaler = self.scalers.get("normal") or (self.scalers.values().__iter__().__next__() if self.scalers else None)
        for regime, scaler in self.scalers.items():
            mask = regime_series == regime
            if mask.any():
                result.loc[mask] = scaler.transform(series[mask])
        remaining = result.isna()
        if remaining.any() and default_scaler is not None:
            result.loc[remaining] = default_scaler.transform(series.loc[remaining])
        result = result.ffill().fillna(0.0)
        return result

    def fit_transform(self, series: pd.Series, regime_series: pd.Series) -> pd.Series:
        self.fit(series, regime_series)
        return self.transform(series, regime_series)

    def to_params(self) -> Dict[str, Any]:
        return {
            "type": "regime_split",
            "regimes": self.regimes,
            "base_scaler": self.base_scaler_cls.__name__,
            "window": self.window,
            "clip": self.clip,
            "scalers": {reg: scaler.to_params() for reg, scaler in self.scalers.items()},
        }

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "RegimeSplitScaler":
        scaler = cls(
            base_scaler_cls=ZScoreScaler,
            regimes=params.get("regimes", ["compression", "normal", "expansion"]),
            window=params.get("window", 500),
            clip=params.get("clip", 5.0),
        )
        scaler.scalers = {reg: ZScoreScaler.from_params(s_params) for reg, s_params in params.get("scalers", {}).items()}
        return scaler

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_params(), f)


__all__ = ["RegimeSplitScaler"]
