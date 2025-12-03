import logging
from typing import Optional

import numpy as np
import pandas as pd


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    sum_x = x.sum()
    sum_x2 = (x * x).sum()
    denom = window * sum_x2 - sum_x**2

    def _slope(arr: np.ndarray) -> float:
        y = np.asarray(arr, dtype=float)
        sum_y = y.sum()
        sum_xy = (x * y).sum()
        return (window * sum_xy - sum_x * sum_y) / denom

    return series.rolling(window, min_periods=window).apply(_slope, raw=True)


class SessionEngine:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("phase0.session")

    @staticmethod
    def _safe_div(num: pd.Series, denom: pd.Series) -> pd.Series:
        denom = denom.replace(0, np.nan)
        return num / denom

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Running SessionEngine on shape %s", df.shape)
        out = df.copy()
        timestamps = pd.to_datetime(out["open_time_ms"], unit="ms", utc=True)
        hours = timestamps.dt.hour

        out["session_asia"] = ((hours >= 0) & (hours < 8)).astype(int)
        out["session_london"] = ((hours >= 8) & (hours < 16)).astype(int)
        out["session_ny"] = ((hours >= 13) & (hours < 21)).astype(int)

        log_ret = np.log(out["close"]).diff().fillna(0)
        out["session_vol_bias"] = log_ret.rolling(60, min_periods=10).std()

        rolling_24h = log_ret.rolling(1440, min_periods=120).std()
        out["session_liq_compression"] = self._safe_div(out["session_vol_bias"], rolling_24h)

        out["session_vol_slope"] = _rolling_slope(out["session_vol_bias"], window=300)
        self.logger.info("SessionEngine completed")
        return out


__all__ = ["SessionEngine"]
