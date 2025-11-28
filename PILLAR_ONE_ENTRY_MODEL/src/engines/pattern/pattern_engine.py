import numpy as np
import pandas as pd

from .pattern_engine_config import DEFAULT_PATTERN_CONFIG, PatternConfig


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


def compute_pattern_engine_score(
    df: pd.DataFrame, config: PatternConfig = DEFAULT_PATTERN_CONFIG
) -> pd.DataFrame:
    """
    Fuse candlestick/micro-pattern primitives with volatility bias into pattern_score.
    """
    out = df.copy()

    base = out.get("pattern_score", pd.Series(0.0, index=out.index)).astype(float)
    pin = out.get("pin_bar", pd.Series(0.0, index=out.index)).astype(float)
    momentum_bar = out.get("momentum_bar", pd.Series(0.0, index=out.index)).astype(float)
    engulfing = out.get("engulfing", pd.Series(0.0, index=out.index)).astype(float)
    imbalance_bar = out.get("imbalance_bar", pd.Series(0.0, index=out.index)).astype(float)

    base_z = _robust_z(base, config.clip_value)
    pin_z = _robust_z(pin, config.clip_value)
    mom_bar_z = _robust_z(momentum_bar, config.clip_value)
    engulf_z = _robust_z(engulfing, config.clip_value)
    imb_bar_z = _robust_z(imbalance_bar, config.clip_value)

    vol_pct = out.get("vol_percentile", pd.Series(0.5, index=out.index)).astype(float).clip(0.0, 1.0)
    vol_bias = (vol_pct * 2.0 - 1.0)  # bias toward patterns in high vol

    weights = config
    raw_score = (
        weights.base_weight * base_z +
        weights.pin_weight * pin_z +
        weights.momentum_bar_weight * mom_bar_z +
        weights.engulfing_weight * engulf_z +
        weights.imbalance_bar_weight * imb_bar_z +
        weights.vol_bias_weight * vol_bias
    )

    out["pattern_score"] = pd.Series(np.tanh(raw_score)).ewm(
        alpha=config.smoothing_alpha, adjust=False
    ).mean()
    return out
