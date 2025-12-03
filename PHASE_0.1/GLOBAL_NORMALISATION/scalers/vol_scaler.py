import json
from typing import Any, Dict

import numpy as np
import pandas as pd


class VolScaler:
    def __init__(self, denominator_col: str, epsilon: float = 1e-9):
        self.denominator_col = denominator_col
        self.epsilon = epsilon
        self.last_denominator: float = 1.0

    def fit(self, series: pd.Series, denom: pd.Series) -> "VolScaler":
        denom = denom.replace(0, np.nan).fillna(self.epsilon)
        self.last_denominator = float(denom.iloc[-1])
        return self

    def transform(self, series: pd.Series, denom: pd.Series) -> pd.Series:
        denom = denom.replace(0, np.nan).fillna(self.epsilon)
        scaled = series / denom
        scaled = scaled.replace([np.inf, -np.inf], np.nan)
        scaled = scaled.ffill().fillna(0.0)
        return scaled

    def fit_transform(self, series: pd.Series, denom: pd.Series) -> pd.Series:
        self.fit(series, denom)
        return self.transform(series, denom)

    def to_params(self) -> Dict[str, Any]:
        return {
            "type": "vol",
            "denominator_col": self.denominator_col,
            "epsilon": self.epsilon,
            "last_denominator": self.last_denominator,
        }

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "VolScaler":
        scaler = cls(
            denominator_col=params.get("denominator_col"),
            epsilon=params.get("epsilon", 1e-9),
        )
        scaler.last_denominator = params.get("last_denominator", 1.0)
        return scaler

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_params(), f)


__all__ = ["VolScaler"]
