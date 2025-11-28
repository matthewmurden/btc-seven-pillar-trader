import numpy as np
import pandas as pd

from .volatility_engine_config import DEFAULT_VOLATILITY_CONFIG, VolatilityConfig


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


def compute_volatility_engine_score(
    df: pd.DataFrame, config: VolatilityConfig = DEFAULT_VOLATILITY_CONFIG
) -> pd.DataFrame:
    """
    Build a deterministic volatility_score using ATR, expansion, realized vol,
    percentile rank, and volatility dynamics. Vectorized and robust-normalized.
    """
    out = df.copy()

    # ATR component (robust z)
    atr_z = _robust_z(out["atr"], config.clip_value)

    # ATR expansion component (directional change signal)
    atrx = out["atr_expansion"].astype(float).clip(-config.clip_value, config.clip_value).fillna(0.0)

    # Realized volatility component
    realized_z = _robust_z(out["volatility"], config.clip_value)

    # Percentile rank (already 0-1); optionally smooth lightly
    vol_pct = out["vol_percentile"].astype(float).clip(0.0, 1.0).fillna(0.5)
    vol_pct_sm = vol_pct.ewm(alpha=0.05, adjust=False).mean()

    # Volatility dynamics: EMA of first difference, normalized
    vol_change = out["volatility"].astype(float).diff().fillna(0.0)
    vol_change_sm = vol_change.ewm(span=24, adjust=False).mean()
    dynamics = np.tanh(_robust_z(vol_change_sm, config.clip_value))

    # Weighted fusion
    weights = config
    raw_score = (
        weights.atr_weight * atr_z +
        weights.atrx_weight * atrx +
        weights.realized_weight * realized_z +
        weights.percentile_weight * (vol_pct_sm * 2.0 - 1.0) +  # shift to roughly centered
        weights.dynamics_weight * dynamics
    )

    out["volatility_score"] = pd.Series(np.tanh(raw_score)).ewm(
        alpha=config.smoothing_alpha, adjust=False
    ).mean()
    return out
