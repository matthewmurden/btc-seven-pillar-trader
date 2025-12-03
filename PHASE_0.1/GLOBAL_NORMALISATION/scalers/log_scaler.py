import json
from typing import Any, Dict

import numpy as np
import pandas as pd


class LogScaler:
    def __init__(self, signed: bool = True):
        self.signed = signed

    def fit(self, series: pd.Series) -> "LogScaler":
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        if self.signed:
            scaled = np.sign(series) * np.log1p(np.abs(series))
        else:
            scaled = np.log1p(series.clip(lower=0))
        scaled = pd.Series(scaled, index=series.index)
        scaled = scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return scaled

    def fit_transform(self, series: pd.Series) -> pd.Series:
        return self.transform(series)

    def to_params(self) -> Dict[str, Any]:
        return {"type": "log", "signed": self.signed}

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "LogScaler":
        return cls(signed=params.get("signed", True))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_params(), f)


__all__ = ["LogScaler"]
