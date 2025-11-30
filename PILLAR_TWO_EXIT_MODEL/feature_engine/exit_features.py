"""Exit-specific feature engineering for Pillar Two."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


class ExitFeatureEngine:
    """
    Compute exit-focused features with leakage safeguards.
    All forward-looking quantities remain in labels; features rely on current/past bars only.
    """

    def __init__(self, atr_window: int = 14):
        self.atr_window = atr_window
        self.entry_features_path_candidates = [
            Path(__file__).resolve().parent.parent.parent
            / "PILLAR_ONE_ENTRY_MODEL"
            / "data"
            / "processed"
            / "features_with_entry_probability.parquet",
            Path(__file__).resolve().parent.parent.parent
            / "PILLAR_ONE_ENTRY_MODEL"
            / "data"
            / "processed"
            / "features_with_final_signal.parquet",
        ]

    @staticmethod
    def _true_range(df: pd.DataFrame) -> pd.Series:
        prev_close = df["close"].shift(1)
        ranges = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        )
        return ranges.max(axis=1)

    def _atr(self, df: pd.DataFrame) -> pd.Series:
        tr = self._true_range(df)
        return tr.rolling(self.atr_window, min_periods=self.atr_window).mean()

    def _volatility_percentile(self, vol: pd.Series, window: int = 200) -> pd.Series:
        def pct_rank(x: np.ndarray) -> float:
            valid = x[~np.isnan(x)]
            if len(valid) == 0:
                return np.nan
            last = valid[-1]
            return (np.sum(valid <= last) / len(valid)) * 100.0

        return vol.rolling(window, min_periods=window).apply(pct_rank, raw=True)

    def _entropy(self, series: pd.Series, window: int = 20) -> pd.Series:
        def ent(x: np.ndarray) -> float:
            x = x[~np.isnan(x)]
            if len(x) == 0:
                return np.nan
            vals, counts = np.unique(x, return_counts=True)
            probs = counts / counts.sum()
            return -(probs * np.log(probs + 1e-12)).sum()

        return series.rolling(window, min_periods=window).apply(ent, raw=True)

    def _attach_entry_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for path in self.entry_features_path_candidates:
            if path.exists():
                entry_df = pd.read_parquet(path)
                if "timestamp" in entry_df.columns:
                    entry_df["timestamp"] = pd.to_datetime(entry_df["timestamp"])
                    entry_df = entry_df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
                    df = df.merge(entry_df, on="timestamp", how="left", suffixes=("", "_p1"))
                # align by index position as fallback
                cols = [
                    "momentum_score",
                    "trend_score",
                    "volatility_score",
                    "structure_score",
                    "orderflow_score",
                    "pattern_score",
                    "regime_state",
                    "entry_probability",
                    "raw_entry_confidence",
                ]
                for c in cols:
                    if c in df.columns and c + "_p1" in df.columns:
                        df[c] = df[c].fillna(df[c + "_p1"])
                    elif c + "_p1" in df.columns:
                        df[c] = df[c + "_p1"]
                    elif c in entry_df.columns and c not in df.columns:
                        df[c] = entry_df[c].values[: len(df)]
                drop_cols = [c for c in df.columns if c.endswith("_p1")]
                if drop_cols:
                    df = df.drop(columns=drop_cols)
                break
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "timestamp" in out.columns:
            out["timestamp"] = pd.to_datetime(out["timestamp"])

        # Price & Market State
        out["feature_exit_hl2"] = (out["high"] + out["low"]) / 2
        out["feature_exit_hlc3"] = (out["high"] + out["low"] + out["close"]) / 3
        body = (out["close"] - out["open"]).abs()
        range_ = (out["high"] - out["low"]).replace(0, np.nan)
        out["feature_exit_body_pct"] = body / range_
        out["feature_exit_wick_top_pct"] = out["wick_top"] / range_
        out["feature_exit_wick_bottom_pct"] = out["wick_bottom"] / range_
        out["feature_exit_range_pct"] = range_ / out["close"]
        out["feature_exit_spread"] = (out["high"] - out["low"])

        atr = self._atr(out)
        vol_percentile = self._volatility_percentile(out["volatility"])
        out["feature_exit_vol_percentile"] = vol_percentile

        # Excursion Predictors
        # distance_to_liquidity: distance to last swing highs/lows
        swing_high = out["high"].rolling(20, min_periods=1).max()
        swing_low = out["low"].rolling(20, min_periods=1).min()
        out["feature_exit_distance_to_liquidity"] = (
            np.minimum((swing_high - out["close"]).abs(), (out["close"] - swing_low).abs()) / (out["close"] + 1e-9)
        )
        # VWAP deviation
        vwap_cum = (out["close"] * out["volume"]).cumsum() / (out["volume"].cumsum() + 1e-9)
        out["feature_exit_distance_to_vwap"] = (out["close"] - vwap_cum) / (vwap_cum + 1e-9)
        # trend stretch
        slope = out["close"].diff(5)
        out["feature_exit_trend_stretch"] = slope / (atr + 1e-9)
        # volatility compression
        vol_rolling = out["volatility"].rolling(50, min_periods=20)
        out["feature_exit_vol_compression"] = (
            (out["volatility"] <= vol_rolling.quantile(0.2)).astype(int)
        )
        # volume expansion
        vol_mean = out["volume"].rolling(30, min_periods=5).mean()
        out["feature_exit_volume_expansion"] = out["volume"] / (vol_mean + 1e-9)

        # Path Predictors
        returns = out["close"].pct_change()
        out["feature_exit_exhaustion_score"] = (-out["feature_exit_wick_top_pct"] + -out["feature_exit_wick_bottom_pct"]).fillna(0)
        out["feature_exit_bar_clustering"] = (
            (returns > 0).rolling(10, min_periods=5).mean()
        )
        out["feature_exit_micro_entropy"] = self._entropy(np.sign(returns), window=20)

        # Orderflow Predictors
        out["feature_exit_delta_into_entry"] = out["delta"]
        norm_imb = out.get("feature_orderflow_norm_imbalance")
        if norm_imb is None:
            norm_imb = out["delta"] / (out["volume"].replace(0, np.nan))
        imbalance_change = norm_imb.diff()
        out["feature_exit_absorption_score"] = (-imbalance_change).fillna(0)
        out["feature_exit_imbalance_persistence"] = norm_imb.rolling(
            20, min_periods=5
        ).mean()

        # Structure Predictors
        htf_high = out["high"].rolling(50, min_periods=10).max()
        htf_low = out["low"].rolling(50, min_periods=10).min()
        out["feature_exit_distance_htf_swing"] = (
            np.minimum((htf_high - out["close"]).abs(), (out["close"] - htf_low).abs()) / (out["close"] + 1e-9)
        )
        bos_up = out.get("feature_structure_bos_up", pd.Series(0, index=out.index))
        bos_down = out.get("feature_structure_bos_down", pd.Series(0, index=out.index))
        out["feature_exit_bos_distance"] = (
            (bos_up - bos_down)
            .rolling(20, min_periods=5)
            .sum()
        )
        out["feature_exit_liquidity_alignment"] = np.minimum(
            (out["high"] - htf_high).abs(), (out["low"] - htf_low).abs()
        ) / (out["close"] + 1e-9)

        # Pattern Predictors
        out["feature_exit_wick_imbalance"] = (
            (out["wick_top"] - out["wick_bottom"]).abs() / (range_ + 1e-9)
        )
        out["feature_exit_climax_volume"] = out["volume"] / (out["volume"].rolling(50, min_periods=5).max() + 1e-9)

        # Volatility Forecast
        out["feature_exit_atr_slope"] = atr.diff()
        for h in [5, 15, 30, 60]:
            out[f"feature_exit_realised_vol_past_{h}"] = (
                returns.rolling(h, min_periods=5).std(ddof=0)
            )
        # simple GARCH-style EWMA
        out["feature_exit_garch_var"] = returns.pow(2).ewm(alpha=0.1, adjust=False).mean()

        # Carry-over from Pillar 1
        out = self._attach_entry_features(out)

        # Fill scalar NaNs (excluding object/array columns)
        for col in out.columns:
            if col == "next_1_to_300_returns":
                continue
            if out[col].dtype.kind in ("O", "S", "U"):
                continue
            out[col] = out[col].ffill().bfill()

        return out
