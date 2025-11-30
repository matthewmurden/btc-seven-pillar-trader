import numpy as np
import pandas as pd


def zscore(series: pd.Series, window: int = 100) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / (std + 1e-9)


def smooth(series: pd.Series, alpha: float = 0.2) -> pd.Series:
    return series.ewm(alpha=alpha).mean()