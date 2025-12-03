from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class ZScoreScaler:
    window: int = 1000
    clip: float = 5.0
    params: Optional[Dict[str, Dict[str, float]]] = None

    def fit(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        roll = df.rolling(self.window, min_periods=self.window)
        means = roll.mean()
        stds = roll.std(ddof=0)
        self.params = {}
        for col in df.columns:
            self.params[col] = {
                "window": self.window,
                "clip": self.clip,
                "last_mean": float(means[col].iloc[-1]) if len(means[col]) else 0.0,
                "last_std": float(stds[col].iloc[-1]) if len(stds[col]) else 1e-9,
            }
        return self.params

    def transform(self, df: pd.DataFrame, params: Optional[Dict[str, Dict[str, float]]] = None) -> pd.DataFrame:
        params = params or self.params
        if params is None:
            raise ValueError("Scaler parameters not set.")
        out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
        for col in df.columns:
            p = params[col]
            win = int(p.get("window", self.window))
            clip = float(p.get("clip", self.clip))
            roll = df[col].rolling(win, min_periods=win)
            z = (df[col] - roll.mean()) / roll.std(ddof=0)
            z = z.clip(-clip, clip).fillna(0.0).replace([np.inf, -np.inf], 0.0)
            out[col] = z
        return out
