"""
Pillar 1 — Label Construction

Builds the label set used by the multi-horizon ML predictors (Section 3.4):

Labels produced:
  - y_dir_1_2      : multi-class direction label for 1–2 bar horizon (-1, 0, 1)
  - y_dir_5_10     : multi-class direction label for 5–10 bar horizon (-1, 0, 1)
  - y_dir_15_30    : multi-class direction label for 15–30 bar horizon (-1, 0, 1)
  - y_breakout     : binary breakout label (0/1)
  - y_mean_reversion: binary mean-reversion label (0/1)

All labels are:
  - strictly forward-looking (no leakage)
  - aligned on open_time_ms
  - designed to be volatility-aware and regime-agnostic (regime is used in Pillar 4).

Intended input:
  - df with columns: open_time_ms, close, high, low, vol_atr_14, rev_zscore_50
    (e.g. features_pillar1_with_engines.parquet)
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class Pillar1LabelConfig:
    # Direction horizons: (min_h, max_h) in bars, using forward extremes.
    dir_horizons: Dict[str, Tuple[int, int]] = None
    # Multipliers applied to volatility scale to get per-bar move thresholds.
    dir_threshold_mult: Dict[str, float] = None

    # Breakout label settings
    breakout_lookback: int = 50
    breakout_horizon: int = 30
    breakout_range_mult: float = 0.75  # fraction of recent range to qualify as breakout

    # Mean reversion label settings
    mr_horizon: int = 10
    mr_z_threshold: float = 1.5       # |z| >= this to be considered an "extreme"
    mr_return_mult: float = 0.5       # fraction of vol_scale to require as snap-back move

    def __post_init__(self):
        if self.dir_horizons is None:
            self.dir_horizons = {
                "y_dir_1_2": (1, 2),
                "y_dir_5_10": (5, 10),
                "y_dir_15_30": (15, 30),
            }
        if self.dir_threshold_mult is None:
            self.dir_threshold_mult = {
                "y_dir_1_2": 0.6,
                "y_dir_5_10": 0.8,
                "y_dir_15_30": 1.0,
            }


# ----------------------------------------------------------------------
# Core helpers
# ----------------------------------------------------------------------
def _ensure_required_columns(df: pd.DataFrame) -> None:
    required = ["open_time_ms", "close", "high", "low"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input df for labels is missing required columns: {missing}")

    if "vol_atr_14" not in df.columns:
        raise ValueError(
            "vol_atr_14 is required for volatility-aware thresholds. "
            "Ensure Pillar 1 primitive + volatility primitives were built."
        )

    if "rev_zscore_50" not in df.columns:
        raise ValueError(
            "rev_zscore_50 is required for mean-reversion labels. "
            "Ensure reversion primitives were built."
        )


def _compute_vol_scale(df: pd.DataFrame) -> pd.Series:
    """
    Volatility scale used for thresholds.

    vol_scale ~ typical 1-bar move in return space:
        vol_scale = ATR / close

    Clipped gently to avoid pathological spikes.
    """
    vol = pd.to_numeric(df["vol_atr_14"], errors="coerce").abs()
    close = pd.to_numeric(df["close"], errors="coerce").abs()
    scale = (vol / (close + 1e-9)).replace([np.inf, -np.inf], np.nan)
    scale = scale.clip(lower=1e-5, upper=0.5)  # avoid insane extremes
    return scale.fillna(scale.median())


def _compute_directional_labels(
    close: pd.Series,
    vol_scale: pd.Series,
    horizons: Dict[str, Tuple[int, int]],
    mults: Dict[str, float],
) -> Dict[str, np.ndarray]:
    """
    For each named label (e.g. y_dir_1_2), compute a multi-class direction label:

      - Horizon window [min_h, max_h] forward in time
      - Use max and min forward returns within that window
      - Let thr_i = mult * vol_scale[i]
      - If max_up >= thr_i and max_up >= |min_dn| -> label = +1
      - Else if |min_dn| >= thr_i and |min_dn| > max_up -> label = -1
      - Else label = 0

    Tail bars that don't have enough future data get NaN.
    """
    close_arr = np.asarray(close, float)
    vol_arr = np.asarray(vol_scale, float)
    n = len(close_arr)

    out: Dict[str, np.ndarray] = {}

    for name, (min_h, max_h) in horizons.items():
        labels = np.full(n, np.nan, dtype=float)
        thr_mult = mults.get(name, 1.0)

        for i in range(n):
            start = i + min_h
            end = min(n, i + max_h + 1)
            if start >= end:
                # Not enough future bars to label
                continue

            base = close_arr[i]
            future_slice = close_arr[start:end]
            rets = future_slice / base - 1.0

            max_up = np.max(rets)
            min_dn = np.min(rets)

            thr = thr_mult * vol_arr[i]
            if np.isnan(thr) or thr <= 0:
                continue

            if max_up >= thr and max_up >= abs(min_dn):
                labels[i] = 1.0
            elif abs(min_dn) >= thr and abs(min_dn) > max_up:
                labels[i] = -1.0
            else:
                labels[i] = 0.0

        out[name] = labels

    return out


def _compute_breakout_label(
    high: pd.Series,
    low: pd.Series,
    lookback: int,
    fwd_horizon: int,
    range_mult: float,
) -> np.ndarray:
    """
    Breakout label (binary):

      - Compute recent range over [t-lookback, t) using high/low.
      - Look forward fwd_horizon bars and take future high/low.
      - If future_high >= prev_high + range_mult * range
        OR future_low <= prev_low - range_mult * range
        → breakout (1)
      - Else → no breakout (0)
    """
    high_arr = np.asarray(high, float)
    low_arr = np.asarray(low, float)
    n = len(high_arr)
    labels = np.full(n, np.nan, dtype=float)

    for i in range(n):
        if i < lookback:
            continue  # insufficient history

        prev_high = np.max(high_arr[i - lookback : i])
        prev_low = np.min(low_arr[i - lookback : i])
        range_ = prev_high - prev_low
        if range_ <= 0:
            continue

        end = min(n, i + fwd_horizon + 1)
        if i + 1 >= end:
            continue  # insufficient future

        fut_high = np.max(high_arr[i + 1 : end])
        fut_low = np.min(low_arr[i + 1 : end])

        up_break = fut_high >= prev_high + range_mult * range_
        dn_break = fut_low <= prev_low - range_mult * range_

        labels[i] = 1.0 if (up_break or dn_break) else 0.0

    return labels


def _compute_mean_reversion_label(
    close: pd.Series,
    zscore: pd.Series,
    vol_scale: pd.Series,
    horizon: int,
    z_threshold: float,
    ret_mult: float,
) -> np.ndarray:
    """
    Mean reversion label (binary):

    Idea:
      - At time t, we are at an extreme if |zscore_t| >= z_threshold.
      - Look horizon bars forward to close_{t+h}.
      - If z_t >= z_threshold (overbought) and forward_return <= -ret_mult * vol_scale_t → 1
      - If z_t <= -z_threshold (oversold) and forward_return >=  ret_mult * vol_scale_t → 1
      - Otherwise → 0
    """
    close_arr = np.asarray(close, float)
    z_arr = np.asarray(zscore, float)
    vol_arr = np.asarray(vol_scale, float)

    n = len(close_arr)
    labels = np.full(n, np.nan, dtype=float)

    for i in range(n):
        j = i + horizon
        if j >= n:
            continue

        fr = close_arr[j] / close_arr[i] - 1.0
        z = z_arr[i]
        thr = ret_mult * vol_arr[i]

        if np.isnan(z) or np.isnan(thr) or thr <= 0:
            continue

        if z >= z_threshold and fr <= -thr:
            labels[i] = 1.0
        elif z <= -z_threshold and fr >= thr:
            labels[i] = 1.0
        else:
            labels[i] = 0.0

    return labels


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def build_pillar1_labels(
    df: pd.DataFrame,
    cfg: Pillar1LabelConfig,
) -> pd.DataFrame:
    """
    Given a feature-level DataFrame (e.g. features_pillar1_with_engines),
    compute all Pillar 1 labels and return a compact labels DataFrame with:

      ['open_time_ms',
       'y_dir_1_2', 'y_dir_5_10', 'y_dir_15_30',
       'y_breakout', 'y_mean_reversion']

    Tail rows without full future horizon are NaN and will be dropped
    by the training pipeline.
    """
    _ensure_required_columns(df)

    df = df.sort_values("open_time_ms").reset_index(drop=True)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol_scale = _compute_vol_scale(df)
    zscore = pd.to_numeric(df["rev_zscore_50"], errors="coerce")

    # Direction labels
    dir_labels = _compute_directional_labels(
        close=close,
        vol_scale=vol_scale,
        horizons=cfg.dir_horizons,
        mults=cfg.dir_threshold_mult,
    )

    # Breakout label
    breakout = _compute_breakout_label(
        high=high,
        low=low,
        lookback=cfg.breakout_lookback,
        fwd_horizon=cfg.breakout_horizon,
        range_mult=cfg.breakout_range_mult,
    )

    # Mean reversion label
    mean_rev = _compute_mean_reversion_label(
        close=close,
        zscore=zscore,
        vol_scale=vol_scale,
        horizon=cfg.mr_horizon,
        z_threshold=cfg.mr_z_threshold,
        ret_mult=cfg.mr_return_mult,
    )

    labels_df = pd.DataFrame(
        {
            "open_time_ms": df["open_time_ms"].values,
            "y_dir_1_2": dir_labels["y_dir_1_2"],
            "y_dir_5_10": dir_labels["y_dir_5_10"],
            "y_dir_15_30": dir_labels["y_dir_15_30"],
            "y_breakout": breakout,
            "y_mean_reversion": mean_rev,
        }
    )

    return labels_df