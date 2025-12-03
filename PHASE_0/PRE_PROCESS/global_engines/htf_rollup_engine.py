import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from PHASE_0.PRE_PROCESS.raw_data_layer import HTF_TIMEFRAMES


class HTFRollupEngine:
    def __init__(
        self,
        timeframes: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.timeframes = timeframes or HTF_TIMEFRAMES
        self.logger = logger or logging.getLogger("phase0.htf")

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False, min_periods=span).mean()

    def _compute_vol_slope(self, vol_df: pd.DataFrame) -> pd.Series:
        tf_map = ["5m", "15m", "1h", "4h"]
        avail = [tf for tf in tf_map if f"{tf}_vol" in vol_df.columns]
        if len(avail) < 2:
            return pd.Series(np.nan, index=vol_df.index)

        x_lookup = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}
        x = np.array([x_lookup[tf] for tf in avail], dtype=float)
        x_centered = x - x.mean()
        var_x = (x_centered ** 2).sum()

        vol_matrix = vol_df[[f"{tf}_vol" for tf in avail]].to_numpy()
        slope = (vol_matrix * x_centered).sum(axis=1) / var_x
        return pd.Series(slope, index=vol_df.index)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Running HTFRollupEngine on shape %s", df.shape)
        out = df.copy()
        out["ret"] = out["close"].pct_change()

        vol_df = pd.DataFrame(index=out.index)
        for tf in ["5m", "15m", "1h", "4h"]:
            close_col = f"{tf}_close"
            if close_col not in out.columns:
                continue
            ret_tf = out[close_col].pct_change()
            vol_df[f"{tf}_vol"] = ret_tf.rolling(50, min_periods=5).std()

        out["htf_vol_slope"] = self._compute_vol_slope(vol_df)

        for tf in self.timeframes:
            close_col = f"{tf}_close"
            if close_col not in out.columns:
                continue
            ema20 = self._ema(out[close_col], span=20)
            ema50 = self._ema(out[close_col], span=50)
            out[f"htf_trend_{tf}"] = np.sign(ema20 - ema50)

        self.logger.info("HTFRollupEngine completed")
        return out


__all__ = ["HTFRollupEngine"]
