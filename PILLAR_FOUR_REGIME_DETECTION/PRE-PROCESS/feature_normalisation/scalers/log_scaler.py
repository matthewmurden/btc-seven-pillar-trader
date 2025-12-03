import numpy as np
import pandas as pd


class LogScaler:
    @staticmethod
    def transform(df: pd.DataFrame) -> pd.DataFrame:
        arr = df.to_numpy(dtype=float)
        transformed = np.sign(arr) * np.log1p(np.abs(arr))
        out = pd.DataFrame(transformed, index=df.index, columns=df.columns)
        return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
