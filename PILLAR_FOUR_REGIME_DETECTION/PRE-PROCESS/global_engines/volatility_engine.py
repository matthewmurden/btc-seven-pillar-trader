import numpy as np
import pandas as pd


class VolatilityEngine:
    """Deterministic volatility utilities."""

    @staticmethod
    def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr

    def compute_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
        tr = self._true_range(high, low, close)

        def _wilder(tr_series: pd.Series, period: int) -> pd.Series:
            atr = tr_series.ewm(alpha=1 / period, adjust=False).mean()
            return atr

        atr14 = _wilder(tr, 14)
        atr50 = _wilder(tr, 50)
        atr200 = _wilder(tr, 200)
        return pd.DataFrame({"atr14": atr14, "atr50": atr50, "atr200": atr200})

    def realised_volatility(self, close: pd.Series, windows=None) -> pd.DataFrame:
        if windows is None:
            windows = [5, 15, 60]
        rets = close.pct_change().fillna(0)
        out = {}
        for w in windows:
            out[f"rv_{w}m" if w < 60 else "rv_1h"] = rets.rolling(w, min_periods=w).std().fillna(0)
        return pd.DataFrame(out)

    def volatility_cycles(self, close: pd.Series) -> pd.Series:
        rets = close.pct_change().fillna(0)
        rv60 = rets.rolling(60, min_periods=60).std()
        mean = rv60.rolling(200, min_periods=60).mean()
        std = rv60.rolling(200, min_periods=60).std()
        z = (rv60 - mean) / std.replace(0, np.nan)
        z = z.fillna(0)
        vol_cycle_phase = z.ewm(span=30, adjust=False).mean().fillna(0)
        return vol_cycle_phase.rename("vol_cycle_phase")

    def compression_expansion(self, close: pd.Series) -> pd.DataFrame:
        rets = close.pct_change().fillna(0)
        rv = rets.rolling(60, min_periods=60).std()
        q20 = rv.rolling(240, min_periods=60).quantile(0.2)
        q80 = rv.rolling(240, min_periods=60).quantile(0.8)
        compression_flag = (rv < q20).astype(int)
        expansion_flag = (rv > q80).astype(int)
        vol_regime = pd.Series("normal", index=close.index)
        vol_regime = vol_regime.mask(expansion_flag == 1, "expansion")
        vol_regime = vol_regime.mask(compression_flag == 1, "compression")
        return pd.DataFrame(
            {
                "compression_flag": compression_flag.fillna(0).astype(int),
                "expansion_flag": expansion_flag.fillna(0).astype(int),
                "vol_regime": vol_regime,
            }
        )
