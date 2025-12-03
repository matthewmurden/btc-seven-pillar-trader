import numpy as np
import pandas as pd
from typing import Dict


class HTFRollupEngine:
    """High-timeframe rollup utilities."""

    @staticmethod
    def merge_htf_candles(df_1m: pd.DataFrame, htf_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        merged = df_1m.copy()
        for tf, df in htf_dict.items():
            df_sorted = df.sort_values("open_time_ms").drop_duplicates("open_time_ms")
            suffix = f"_{tf}"
            merged = pd.merge_asof(
                merged.sort_values("open_time_ms"),
                df_sorted.add_suffix(suffix),
                left_on="open_time_ms",
                right_on=f"open_time_ms{suffix}",
                direction="backward",
            )
            merged = merged.drop(columns=[f"open_time_ms{suffix}"])
            merged = merged.fillna(method="ffill")
        return merged

    @staticmethod
    def compute_htf_volatility_slope(htf_dict: Dict[str, pd.DataFrame]) -> pd.Series:
        time_map = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}
        vols = {}
        for tf, df in htf_dict.items():
            if tf not in time_map:
                continue
            rets = df["close"].pct_change().fillna(0)
            vols[tf] = rets.rolling(5, min_periods=5).std()
        # align by asof to 5m timestamps base
        base = htf_dict["5m"][["open_time_ms"]].copy()
        for tf, series in vols.items():
            df_tf = pd.DataFrame({"open_time_ms": htf_dict[tf]["open_time_ms"], f"rv_{tf}": series})
            base = pd.merge_asof(
                base.sort_values("open_time_ms"),
                df_tf.sort_values("open_time_ms"),
                on="open_time_ms",
                direction="backward",
            )
        slopes = []
        xs = []
        cols = []
        for tf, minutes in time_map.items():
            if f"rv_{tf}" in base.columns:
                xs.append(minutes)
                cols.append(f"rv_{tf}")
        x_arr = np.array(xs, dtype=float)
        for _, row in base[cols].iterrows():
            y = row.to_numpy()
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                slopes.append(0.0)
            else:
                coef = np.polyfit(x_arr[mask], y[mask], 1)
                slopes.append(coef[0])
        base["htf_vol_slope"] = slopes
        return base["htf_vol_slope"].fillna(0)

    @staticmethod
    def compute_htf_trend_context(htf_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        base = htf_dict["5m"][["open_time_ms"]].sort_values("open_time_ms").reset_index(drop=True)
        for tf, df in htf_dict.items():
            ema_fast = df["close"].ewm(span=20, adjust=False).mean()
            ema_slow = df["close"].ewm(span=50, adjust=False).mean()
            trend = ema_fast - ema_slow
            aligned = pd.merge_asof(
                base,
                pd.DataFrame({"open_time_ms": df["open_time_ms"], f"trend_{tf}": trend}),
                on="open_time_ms",
                direction="backward",
            )
            base[f"htf_trend_{tf}"] = aligned[f"trend_{tf}"].ffill().fillna(0)
        return base
