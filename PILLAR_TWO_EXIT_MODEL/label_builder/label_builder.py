"""Label builder for Pillar Two (enhanced, leakage-free)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


class LabelBuilder:
    """
    Construct forward-looking exit labels with strict leakage controls.

    All calculations use forward windows via reversed rolling operations to avoid
    peeking beyond the intended horizon.
    """

    def __init__(self, horizon: int = 300, tp_sl_grid: List[float] | None = None, noise_floor_pct: float = 0.0002):
        self.horizon = horizon
        self.tp_sl_grid = tp_sl_grid or list(np.linspace(0.001, 0.03, 10))
        self.noise_floor_pct = noise_floor_pct

    @staticmethod
    def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        return 100 - (100 / (1 + rs))

    def _forward_max(self, series: pd.Series) -> pd.Series:
        return series.iloc[::-1].rolling(self.horizon, min_periods=1).max().iloc[::-1]

    def _forward_min(self, series: pd.Series) -> pd.Series:
        return series.iloc[::-1].rolling(self.horizon, min_periods=1).min().iloc[::-1]

    def _future_vol(self, returns: pd.Series, window: int) -> pd.Series:
        return returns.iloc[::-1].rolling(window, min_periods=1).std(ddof=0).iloc[::-1]

    def compute_excursions(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]

        entry_price = close
        noise_floor = entry_price * self.noise_floor_pct

        max_future_high = self._forward_max(high)
        min_future_low = self._forward_min(low)

        mfe_raw = max_future_high - entry_price
        mae_raw = min_future_low - entry_price

        df["MFE_raw"] = mfe_raw - noise_floor
        df["MAE_raw"] = mae_raw + noise_floor
        df["MFE_pct"] = df["MFE_raw"] / entry_price
        df["MAE_pct"] = df["MAE_raw"] / entry_price
        return df

    def compute_path_series(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["close"].values
        n = len(df)
        max_h = min(self.horizon, n - 1)
        returns_matrix = np.full((n, 300), np.nan)
        for i in range(1, min(301, n)):
            returns_matrix[:, i - 1] = (np.roll(close, -i) - close) / close
        returns_matrix[np.arange(n)[:, None] + np.arange(1, 301) > n] = np.nan
        df["next_1_to_300_returns"] = list(returns_matrix)

        returns = pd.Series(close).pct_change()
        horizons = [1, 5, 15, 30, 60]
        for h in horizons:
            df[f"volatility_curve_{h}"] = self._future_vol(returns.shift(-1), h)

        drift = returns.shift(-1).iloc[::-1].rolling(50, min_periods=1).apply(
            lambda x: np.sign(x).mean(), raw=True
        ).iloc[::-1]
        df["drift_direction"] = drift
        return df

    def compute_first_touch(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        returns_arr = np.vstack(df["next_1_to_300_returns"].to_numpy())
        tp_first = np.zeros(len(df), dtype=int)
        sl_first = np.zeros(len(df), dtype=int)
        exit_bar = np.full(len(df), np.nan)
        tp_dist = np.full(len(df), np.nan)
        sl_dist = np.full(len(df), np.nan)

        tp_level = min(self.tp_sl_grid)
        sl_level = min(self.tp_sl_grid)

        tp_hits = returns_arr >= tp_level
        sl_hits = returns_arr <= -sl_level

        # find first occurrence per row
        tp_idx = np.where(tp_hits.any(axis=1), tp_hits.argmax(axis=1) + 1, np.nan)
        sl_idx = np.where(sl_hits.any(axis=1), sl_hits.argmax(axis=1) + 1, np.nan)

        for i in range(len(df)):
            t = tp_idx[i]
            s = sl_idx[i]
            if np.isnan(t) and np.isnan(s):
                continue
            if np.isnan(s) or (not np.isnan(t) and t <= s):
                tp_first[i] = 1
                exit_bar[i] = t
                tp_dist[i] = returns_arr[i, int(t) - 1] if not np.isnan(t) else np.nan
            else:
                sl_first[i] = 1
                exit_bar[i] = s
                sl_dist[i] = returns_arr[i, int(s) - 1] if not np.isnan(s) else np.nan

        df["tp_first"] = tp_first
        df["sl_first"] = sl_first
        df["exit_bar"] = exit_bar
        df["first_touch_tp_distance"] = tp_dist
        df["first_touch_sl_distance"] = sl_dist
        return df

    def compute_volatility_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        returns = df["close"].pct_change()
        horizons = [1, 5, 15, 30, 60]
        for h in horizons:
            df[f"post_realised_vol_{h}"] = self._future_vol(returns.shift(-1), h)
        return df

    def compute_reversion_label(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "feature_reversion_rsi" in df.columns:
            rsi = df["feature_reversion_rsi"]
            if rsi.isna().all():
                rsi = self._compute_rsi(df["close"])
        else:
            rsi = self._compute_rsi(df["close"])

        rolling = df["close"].rolling(20, min_periods=20)
        zdev = (df["close"] - rolling.mean()) / (rolling.std(ddof=0) + 1e-9)

        low_thr = rsi.rolling(200, min_periods=50).quantile(0.1)
        high_thr = rsi.rolling(200, min_periods=50).quantile(0.9)
        rsi_signal = (rsi <= low_thr) | (rsi >= high_thr)
        zdev_signal = zdev.abs() > 1.25
        df["reversion_label"] = (rsi_signal | zdev_signal).fillna(0).astype(int)
        return df

    def compute_breakout_label(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        rolling_high = df["high"].rolling(20, min_periods=1).max().shift(1)
        df["breakout_label"] = (df["high"] > rolling_high * 1.001).astype(int)
        return df

    def compute_forward_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["forward_returns_1m"] = df["close"].shift(-1) / df["close"] - 1
        df["forward_returns_5m"] = df["close"].shift(-5) / df["close"] - 1
        df["forward_returns_15m"] = df["close"].shift(-15) / df["close"] - 1
        return df

    def run_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.compute_excursions(df)
        df = self.compute_path_series(df)
        df = self.compute_first_touch(df)
        df = self.compute_volatility_labels(df)
        df = self.compute_forward_returns(df)
        df = self.compute_breakout_label(df)
        df = self.compute_reversion_label(df)
        return df


def build_and_save_labels(
    raw_df: pd.DataFrame,
    output_path: Path | None = None,
    builder: LabelBuilder | None = None,
) -> Path:
    """
    Build labels and persist them to parquet for downstream usage.
    """
    output_path = output_path or Path(__file__).resolve().parent.parent / "data" / "processed" / "labels.parquet"
    builder = builder or LabelBuilder()
    labels_df = builder.run_all(raw_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_cols = [
        "MFE_raw",
        "MAE_raw",
        "MFE_pct",
        "MAE_pct",
        "tp_first",
        "sl_first",
        "exit_bar",
        "first_touch_tp_distance",
        "first_touch_sl_distance",
        "forward_returns_1m",
        "forward_returns_5m",
        "forward_returns_15m",
        "breakout_label",
        "reversion_label",
        "drift_direction",
        "post_realised_vol_1",
        "post_realised_vol_5",
        "post_realised_vol_15",
        "post_realised_vol_30",
        "post_realised_vol_60",
    ]
    # Ensure volatility curve columns included
    label_cols += [c for c in labels_df.columns if c.startswith("volatility_curve_")]
    # Path series (object arrays) retained for testing
    label_cols.append("next_1_to_300_returns")
    label_cols = ["timestamp"] + label_cols
    labels_df[label_cols].to_parquet(output_path)
    return output_path
