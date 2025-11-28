"""Label builder for Pillar Two."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class LabelBuilder:
    """Construct forward return labels and simple breakout/reversion labels."""

    def __init__(self, horizon_1m: int = 1, horizon_5m: int = 5, horizon_15m: int = 15):
        self.horizon_1m = horizon_1m
        self.horizon_5m = horizon_5m
        self.horizon_15m = horizon_15m

    @staticmethod
    def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        return 100 - (100 / (1 + rs))

    def compute_forward_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["forward_returns_1m"] = df["close"].shift(-self.horizon_1m) / df["close"] - 1
        df["forward_returns_5m"] = df["close"].shift(-self.horizon_5m) / df["close"] - 1
        df["forward_returns_15m"] = df["close"].shift(-self.horizon_15m) / df["close"] - 1
        return df

    def compute_breakout_label(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        rolling_high = df["high"].rolling(20, min_periods=1).max().shift(1)
        df["breakout_label"] = (
            (df["high"] > rolling_high * 1.001)
        ).astype(int)
        return df

    def compute_reversion_label(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Prefer existing RSI feature; if absent or all NaN, compute a local RSI.
        if "feature_reversion_rsi" in df.columns:
            rsi = df["feature_reversion_rsi"]
            if rsi.isna().all():
                rsi = self._compute_rsi(df["close"])
        else:
            rsi = self._compute_rsi(df["close"])

        # Use z-score deviation as an additional proxy if available.
        if "feature_reversion_zscore_dev" in df.columns:
            zdev = df["feature_reversion_zscore_dev"]
        else:
            rolling = df["close"].rolling(20, min_periods=20)
            zdev = (df["close"] - rolling.mean()) / (rolling.std(ddof=0) + 1e-9)

        # Dynamic RSI thresholds to avoid flat-zero labels.
        low_thr = rsi.rolling(200, min_periods=50).quantile(0.1)
        high_thr = rsi.rolling(200, min_periods=50).quantile(0.9)
        rsi_signal = (rsi <= low_thr) | (rsi >= high_thr)

        zdev_signal = zdev.abs() > 1.25

        reversion_signal = (rsi_signal | zdev_signal).astype(int)
        df["reversion_label"] = reversion_signal.fillna(0).astype(int)
        return df

    def run_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.compute_forward_returns(df)
        df = self.compute_breakout_label(df)
        df = self.compute_reversion_label(df)
        return df


def build_and_save_labels(
    raw_df: pd.DataFrame,
    output_path: Path | None = None,
    builder: LabelBuilder | None = None,
) -> Path:
    """Convenience helper to build labels and persist them."""
    output_path = output_path or Path(__file__).resolve().parent.parent / "data" / "processed" / "labels.parquet"
    builder = builder or LabelBuilder()
    labels_df = builder.run_all(raw_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_df[
        [
            "forward_returns_1m",
            "forward_returns_5m",
            "forward_returns_15m",
            "breakout_label",
            "reversion_label",
        ]
    ].to_parquet(output_path)
    return output_path
