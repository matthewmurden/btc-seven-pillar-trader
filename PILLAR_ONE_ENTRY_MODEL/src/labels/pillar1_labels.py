"""
Pillar 1 — Label Construction

Builds the label set used by the multi-horizon ML predictors (Section 3.4):

Labels produced:
  - y_dir_1_2        : multi-class direction label for 1–2 bar horizon (-1, 0, 1)
  - y_dir_5_10       : multi-class direction label for 5–10 bar horizon (-1, 0, 1)
  - y_dir_15_30      : multi-class direction label for 15–30 bar horizon (-1, 0, 1)
  - y_breakout       : binary breakout label (0/1)
  - y_mean_reversion : binary mean-reversion label (0/1)

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

    # Mean reversion label settings (LOOSENED)
    #
    # - mr_horizon: how far ahead we look for a snap-back (in bars)
    # - mr_z_threshold: how "extreme" rev_zscore_50 must be to count as a candidate
    # - mr_return_mult: how large the snap-back move must be, as a fraction of vol_scale
    #
    # New defaults are intentionally more permissive so we actually get positive labels.
    mr_horizon: int = 15
    mr_z_threshold: float = 1.2       # was 1.5
    mr_return_mult: float = 0.35      # was 0.5

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
    Mean reversion label (binary, LOOSENED DEFINITION):

    Idea:
      - At time t, we are a *candidate* for reversion if |zscore_t| >= z_threshold.
      - For such extremes, look forward up to `horizon` bars.
      - Let vol_scale_t be our volatility scale; threshold_thr = ret_mult * vol_scale_t.
      - Overbought (z_t >= z_threshold):
          If min forward return over [t+1, t+horizon] <= -threshold_thr → label 1
        (i.e., price snaps back down by at least that much at some point).
      - Oversold (z_t <= -z_threshold):
          If max forward return over [t+1, t+horizon] >=  threshold_thr → label 1
        (i.e., price snaps back up by that much at some point).
      - Otherwise (extreme but no decent snap-back) → label 0.
      - Non-extreme bars (|z_t| < z_threshold) → NaN (excluded from training).

    This keeps the label focused on "does an extreme actually mean-revert soon?" rather than
    polluting it with non-extreme bars.
    """
    close_arr = np.asarray(close, float)
    z_arr = np.asarray(zscore, float)
    vol_arr = np.asarray(vol_scale, float)

    n = len(close_arr)
    labels = np.full(n, np.nan, dtype=float)

    for i in range(n):
        z = z_arr[i]
        if np.isnan(z):
            continue

        # Only label genuine extremes; others stay NaN.
        if abs(z) < z_threshold:
            continue

        j_end = min(n, i + horizon + 1)
        if i + 1 >= j_end:
            continue  # insufficient future

        base = close_arr[i]
        fut_slice = close_arr[i + 1 : j_end]
        # forward returns to all future points within horizon
        rets = fut_slice / base - 1.0

        thr = ret_mult * vol_arr[i]
        if np.isnan(thr) or thr <= 0:
            continue

        max_up = np.max(rets)
        min_dn = np.min(rets)

        if z >= z_threshold:
            # Overbought: we want a decent move DOWN at SOME point
            if min_dn <= -thr:
                labels[i] = 1.0
            else:
                labels[i] = 0.0
        elif z <= -z_threshold:
            # Oversold: we want a decent move UP at SOME point
            if max_up >= thr:
                labels[i] = 1.0
            else:
                labels[i] = 0.0
        # (If z was between thresholds we already continued above.)

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
    by the training pipeline. For y_mean_reversion, only "extreme" bars are
    labelled (1 or 0); non-extreme bars remain NaN.
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

    # Mean reversion label (using loosened settings + new logic)
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