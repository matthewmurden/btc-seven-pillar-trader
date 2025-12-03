from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class VolatilityScaler:
    epsilon: float = 1e-9

    def transform(self, df: pd.DataFrame, vol_series: pd.Series) -> pd.DataFrame:
        vol = vol_series.replace(0, np.nan).fillna(method="ffill").fillna(vol_series.mean() + self.epsilon)
        out = df.div(vol, axis=0)
        return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
