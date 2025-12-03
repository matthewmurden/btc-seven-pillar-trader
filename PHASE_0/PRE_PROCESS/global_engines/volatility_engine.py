import logging
from typing import Optional

import numpy as np
import pandas as pd


class VolatilityEngine:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("phase0.volatility")

    @staticmethod
    def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        prev_close = close.shift(1)
        ranges = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        )
        return ranges.max(axis=1)

    @staticmethod
    def _safe_div(num: pd.Series, denom: pd.Series) -> pd.Series:
        denom = denom.replace(0, np.nan)
        return num / denom

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Running VolatilityEngine on shape %s", df.shape)
        out = df.copy()

        log_close = np.log(out["close"])
        log_ret = log_close.diff()
        ret_abs = log_ret.abs()

        tr = self._true_range(out["high"], out["low"], out["close"])
        out["atr14"] = tr.rolling(14, min_periods=1).mean()
        out["atr50"] = tr.rolling(50, min_periods=1).mean()
        out["atr200"] = tr.rolling(200, min_periods=1).mean()

        out["rv_5m"] = log_ret.rolling(5, min_periods=2).std()
        out["rv_15m"] = log_ret.rolling(15, min_periods=3).std()
        out["rv_1h"] = log_ret.rolling(60, min_periods=10).std()

        atr_daily_mean = out["atr14"].rolling(1440, min_periods=60).mean()
        atr_daily_std = out["atr14"].rolling(1440, min_periods=60).std()
        out["vol_cycle_phase"] = self._safe_div(out["atr14"] - atr_daily_mean, atr_daily_std)

        thresh_low = ret_abs.rolling(1000, min_periods=200).quantile(0.2)
        thresh_high = ret_abs.rolling(1000, min_periods=200).quantile(0.8)
        out["compression_flag"] = (ret_abs < thresh_low).astype(int)
        out["expansion_flag"] = (ret_abs > thresh_high).astype(int)

        out["vol_regime"] = np.select(
            [
                out["expansion_flag"] == 1,
                out["compression_flag"] == 1,
            ],
            [
                "expansion",
                "compression",
            ],
            default="normal",
        )
        self.logger.info("VolatilityEngine completed")
        return out


__all__ = ["VolatilityEngine"]
