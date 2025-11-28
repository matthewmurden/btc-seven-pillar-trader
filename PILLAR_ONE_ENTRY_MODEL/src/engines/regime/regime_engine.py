import numpy as np
import pandas as pd

from .regime_engine_config import DEFAULT_REGIME_CONFIG, RegimeConfig


def _robust_z(series: pd.Series, clip: float) -> pd.Series:
    s = series.astype(float)
    med = s.median()
    iqr = s.quantile(0.75) - s.quantile(0.25)
    scale = iqr * 0.7413
    if not np.isfinite(scale) or scale < 1e-9:
        scale = s.std()
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    z = (s - med) / scale
    return z.clip(-clip, clip).fillna(0.0)


def compute_regime_engine_state(
    df: pd.DataFrame, config: RegimeConfig = DEFAULT_REGIME_CONFIG
) -> pd.DataFrame:
    """
    Deterministic regime classifier:
    0 = range, 1 = trend, 2 = breakout.
    """
    out = df.copy()

    vol_pct = out.get("vol_percentile", pd.Series(0.0, index=out.index)).astype(float).clip(0.0, 1.0)
    vol_z = _robust_z(out.get("volatility", pd.Series(0.0, index=out.index)), config.clip_value)
    trend_strength = (out.get("trend_strength", pd.Series(0.0, index=out.index)).astype(float) / 100.0).clip(0.0, 1.0)
    bb_width = _robust_z(out.get("bb_width", pd.Series(0.0, index=out.index)), config.clip_value)
    atrx = out.get("atr_expansion", pd.Series(0.0, index=out.index)).astype(float)

    # Scores
    trend_score = (
        config.trend_weight * trend_strength +
        config.vol_weight * vol_pct +
        0.1 * vol_z
    )

    breakout_score = (
        config.vol_weight * vol_pct +
        config.bb_weight * bb_width +
        config.atrx_weight * atrx +
        0.1 * vol_z
    )

    range_score = (
        config.vol_weight * (1.0 - vol_pct) +
        config.trend_weight * (1.0 - trend_strength) -
        0.05 * bb_width
    )

    stacked = pd.DataFrame(
        {"range": range_score, "trend": trend_score, "breakout": breakout_score},
        index=out.index,
    )
    # Softmax-like confidence without overflow
    max_score = stacked.max(axis=1)
    exp_scores = np.exp(stacked.subtract(max_score, axis=0))
    probs = exp_scores.div(exp_scores.sum(axis=1), axis=0)

    regime_state = stacked.idxmax(axis=1).map({"range": 0, "trend": 1, "breakout": 2}).astype(int)
    regime_confidence = probs.max(axis=1)

    out["regime_state"] = pd.Series(regime_state).ewm(alpha=config.smoothing_alpha, adjust=False).mean().round().astype(int)
    out["regime_confidence"] = regime_confidence
    return out
