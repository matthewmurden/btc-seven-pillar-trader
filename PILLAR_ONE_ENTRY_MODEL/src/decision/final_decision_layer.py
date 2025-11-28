import numpy as np
import pandas as pd
from typing import List

from .final_decision_config import DEFAULT_FINAL_DECISION_CONFIG, FinalDecisionConfig


REQUIRED_COLS: List[str] = ["entry_probability_final", "regime_state"]


def _validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"FinalDecisionLayer missing required columns: {', '.join(missing)}")


def compute_final_signal(df: pd.DataFrame, config: FinalDecisionConfig = DEFAULT_FINAL_DECISION_CONFIG) -> pd.DataFrame:
    """
    Deterministic, regime-aware decision layer.
    Produces {-1, 0, 1}.
    Neutral zone applied BEFORE long/short rules.
    """
    _validate(df)
    out = df.copy()

    prob = out["entry_probability_final"].astype(float).clip(0.0, 1.0)
    regime = out["regime_state"].round().clip(0, 2).astype(int)
    vol = out.get("volatility_score", pd.Series(0.0, index=out.index)).astype(float)
    vol_pct = out.get(
        "volatility_percentile",
        out.get("vol_percentile", pd.Series(1.0, index=out.index)),
    ).astype(float)
    agreement = out.get("agreement_score", pd.Series(1.0, index=out.index)).astype(float)

    # Base thresholds
    long_th = np.full(len(out), config.long_threshold)
    short_th = np.full(len(out), config.short_threshold)

    # Regime adjustments
    long_th = np.where(regime == 2, config.breakout_long_threshold, long_th)  # breakout
    long_th = np.where(regime == 0, config.range_long_threshold, long_th)     # range
    long_th = np.where(regime == 1, config.trend_long_threshold, long_th)     # trend

    short_bias = np.zeros(len(out))
    short_bias = np.where(regime == 2, config.breakout_short_bias, short_bias)
    short_bias = np.where(regime == 0, config.range_short_bias, short_bias)
    short_th = (short_th + short_bias).clip(0.0, 1.0)

    # Balanced regime-aware threshold shifts (Upgrade #5)
    long_shift = np.zeros(len(out))
    long_shift = np.where(regime == 0, config.regime_range_long_shift, long_shift)
    long_shift = np.where(regime == 1, config.regime_trend_long_shift, long_shift)
    long_shift = np.where(regime == 2, config.regime_breakout_long_shift, long_shift)

    short_shift = np.zeros(len(out))
    short_shift = np.where(regime == 0, config.regime_range_short_shift, short_shift)
    short_shift = np.where(regime == 1, config.regime_trend_short_shift, short_shift)
    short_shift = np.where(regime == 2, config.regime_breakout_short_shift, short_shift)

    long_th = long_th + long_shift
    short_th = short_th + short_shift

    # Enforce minimum confidence gap
    gap = long_th - short_th
    too_tight = gap < config.min_confidence_gap
    adjust = (config.min_confidence_gap - gap) / 2
    long_th = np.where(too_tight, long_th + adjust, long_th)
    short_th = np.where(too_tight, short_th - adjust, short_th)

    # Neutral zone
    neutral_low = short_th - config.neutral_zone_width
    neutral_high = long_th + config.neutral_zone_width

    # Volatility suppression adjustment
    vol_suppressed = vol < config.vol_suppression_threshold
    long_th = np.where(vol_suppressed, long_th + config.min_confidence_gap, long_th)
    short_th = np.where(vol_suppressed, short_th - config.min_confidence_gap, short_th)

    # ---------------------------
    # Correct evaluation order
    # ---------------------------
    signals = np.zeros(len(out), dtype=int)

    # 1. Neutral zone first
    neutral_mask = (prob >= neutral_low) & (prob <= neutral_high)
    signals[neutral_mask] = 0

    # 2. Long signal
    long_mask = prob > long_th
    signals[long_mask] = 1

    # 3. Short signal
    short_mask = prob < short_th
    signals[short_mask] = -1

    # 4. Safety filters — override to flat
    range_low_vol = (regime == 0) & (vol_pct < 0.25)
    conflict_mask = agreement < 0
    ultra_low_vol = vol < 0.1
    filters = range_low_vol | conflict_mask | ultra_low_vol
    signals[filters] = 0

    out["final_signal"] = signals.astype(int)
    return out
