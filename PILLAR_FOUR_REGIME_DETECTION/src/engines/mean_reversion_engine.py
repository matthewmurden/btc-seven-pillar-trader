import numpy as np
import pandas as pd

REV_THRESHOLD = 0.6


def _minmax(series: pd.Series) -> pd.Series:
    smin, smax = series.min(), series.max()
    if smax - smin == 0:
        return pd.Series(0.0, index=series.index)
    return (series - smin) / (smax - smin)


def compute_mean_reversion_engine(df: pd.DataFrame) -> pd.DataFrame:
    fvd = df.get("fair_value_divergence", pd.Series(0, index=df.index))
    reversion_score = _minmax(fvd.abs()).fillna(0)
    mean_reversion_flag = (reversion_score > REV_THRESHOLD).astype(int)

    return pd.DataFrame(
        {
            "mean_reversion_flag": mean_reversion_flag,
            "reversion_score": reversion_score,
            "fair_value_distance": fvd.fillna(0),
        },
        index=df.index,
    )
