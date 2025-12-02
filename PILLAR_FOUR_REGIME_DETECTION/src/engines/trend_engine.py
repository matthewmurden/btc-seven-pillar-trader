import numpy as np
import pandas as pd

TREND_STRENGTH_THRESHOLD = 0.4  # can be tuned; keep deterministic


def _minmax(series: pd.Series) -> pd.Series:
    smin, smax = series.min(), series.max()
    if smax - smin == 0:
        return pd.Series(0.0, index=series.index)
    return (series - smin) / (smax - smin)


def compute_trend_engine(df: pd.DataFrame) -> pd.DataFrame:
    rolling_slope = df.get("rolling_slope", pd.Series(0, index=df.index))
    trend_strength_score = _minmax(rolling_slope.abs()).fillna(0)
    trend_flag = (trend_strength_score > TREND_STRENGTH_THRESHOLD).astype(int)
    trend_direction = np.where(trend_flag == 1, np.sign(rolling_slope), 0).astype(int)

    return pd.DataFrame(
        {
            "trend_flag": trend_flag,
            "trend_strength_score": trend_strength_score,
            "trend_direction": trend_direction,
        },
        index=df.index,
    )
