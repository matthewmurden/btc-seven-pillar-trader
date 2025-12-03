from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class MinMaxScaler:
    params: Optional[Dict[str, Dict[str, float]]] = None

    def fit(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        self.params = {}
        for col in df.columns:
            mn, mx = df[col].min(), df[col].max()
            self.params[col] = {"min": float(mn), "max": float(mx)}
        return self.params

    def transform(self, df: pd.DataFrame, params: Optional[Dict[str, Dict[str, float]]] = None) -> pd.DataFrame:
        params = params or self.params
        if params is None:
            raise ValueError("Scaler parameters not set.")
        out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
        for col in df.columns:
            p = params[col]
            denom = (p["max"] - p["min"]) if (p["max"] - p["min"]) != 0 else 1e-9
            scaled = (df[col] - p["min"]) / denom
            scaled = scaled.clip(0.0, 1.0).fillna(0.0).replace([np.inf, -np.inf], 0.0)
            out[col] = scaled
        return out
