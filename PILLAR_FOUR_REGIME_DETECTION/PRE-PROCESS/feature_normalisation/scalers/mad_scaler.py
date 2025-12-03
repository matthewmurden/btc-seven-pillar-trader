from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class MADScaler:
    clip: float = 5.0
    params: Optional[Dict[str, Dict[str, float]]] = None

    def fit(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        self.params = {}
        for col in df.columns:
            med = df[col].median()
            mad = (df[col] - med).abs().median()
            self.params[col] = {"median": float(med), "mad": float(mad if mad != 0 else 1e-9), "clip": self.clip}
        return self.params

    def transform(self, df: pd.DataFrame, params: Optional[Dict[str, Dict[str, float]]] = None) -> pd.DataFrame:
        params = params or self.params
        if params is None:
            raise ValueError("Scaler parameters not set.")
        out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
        for col in df.columns:
            p = params[col]
            med = p["median"]
            mad = p["mad"]
            clip = p.get("clip", self.clip)
            scaled = (df[col] - med) / (1.4826 * mad)
            scaled = scaled.clip(-clip, clip).fillna(0.0).replace([np.inf, -np.inf], 0.0)
            out[col] = scaled
        return out
