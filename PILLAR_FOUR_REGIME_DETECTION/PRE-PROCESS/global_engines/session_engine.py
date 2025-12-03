import numpy as np
import pandas as pd


class SessionEngine:
    """Session context utilities aligned on open_time_ms."""

    @staticmethod
    def session_flags(timestamp_ms: pd.Series) -> pd.DataFrame:
        ts = pd.to_datetime(timestamp_ms, unit="ms", utc=True)
        hours = ts.dt.hour
        asia = hours.between(0, 7).astype(int)
        london = hours.between(7, 13).astype(int)
        ny = hours.between(13, 20).astype(int)
        return pd.DataFrame(
            {
                "session_asia": asia,
                "session_london": london,
                "session_ny": ny,
            },
            index=timestamp_ms.index if hasattr(timestamp_ms, "index") else None,
        )

    @staticmethod
    def session_volatility_bias(close: pd.Series) -> pd.Series:
        rets = close.pct_change().fillna(0)
        rv = rets.rolling(60, min_periods=60).std()
        # weight by active session (assumes flags align with index)
        return rv.rename("session_vol_bias").fillna(0)

    @staticmethod
    def session_liquidity_compression(close: pd.Series, session_vol_bias: pd.Series) -> pd.Series:
        rets = close.pct_change().fillna(0)
        rv24 = rets.rolling(1440, min_periods=60).std()
        liq_comp = (session_vol_bias / rv24.replace(0, np.nan)).replace([np.inf, -np.inf], 0).fillna(0)
        return liq_comp.rename("session_liq_compression")

    @staticmethod
    def session_volatility_slopes(rv_5m: pd.Series, rv_15m: pd.Series, rv_1h: pd.Series) -> pd.Series:
        x = np.array([5, 15, 60], dtype=float)
        # stack and compute slope row-wise
        arr = np.vstack([rv_5m, rv_15m, rv_1h]).T
        slopes = []
        for row in arr:
            if np.isnan(row).all():
                slopes.append(0.0)
                continue
            coef = np.polyfit(x[~np.isnan(row)], row[~np.isnan(row)], 1)
            slopes.append(coef[0])
        return pd.Series(slopes, index=rv_5m.index, name="session_vol_slope")
