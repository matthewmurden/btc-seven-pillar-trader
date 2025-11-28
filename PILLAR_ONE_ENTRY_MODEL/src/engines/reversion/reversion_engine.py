import numpy as np
import pandas as pd

from .reversion_engine_config import DEFAULT_REVERSION_CONFIG, ReversionConfig


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


def compute_reversion_engine_score(
    df: pd.DataFrame, config: ReversionConfig = DEFAULT_REVERSION_CONFIG
) -> pd.DataFrame:
    """
    Build a deterministic mean-reversion score from RSI tension, overshoot,
    z-score pressure, reversal impulse, and acceleration, with volatility scaling.
    """
    out = df.copy()

    rsi = out["rsi"].astype(float)
    rsi_dev = (50.0 - rsi) / 50.0  # positive when mean-reversion to midpoint likely
    rsi_z = _robust_z(rsi_dev, config.clip_value)

    overshoot = out.get("rsi_overshoot", pd.Series(0.0, index=out.index)).astype(float)
    overshoot_z = _robust_z(overshoot, config.clip_value)

    zscore = out["zscore"].astype(float)
    zscore_rev = -_robust_z(zscore, config.clip_value)

    if "reversion_signal" in out.columns:
        rev_signal = out["reversion_signal"].astype(float)
    else:
        # Fallback: impulse from price deviation to vwap or mean close
        if "vwap_dev" in out.columns:
            dev = out["vwap_dev"].astype(float)
        else:
            dev = (out["close"] - out["close"].rolling(48, min_periods=24).mean()).fillna(0.0)
        rev_signal = -_robust_z(dev, config.clip_value)
    rev_signal_z = _robust_z(rev_signal, config.clip_value)

    rsi_change = rsi.diff().fillna(0.0)
    accel = -_robust_z(rsi_change, config.clip_value)  # negative change after high RSI is mean-reverting

    vol = out.get("volatility", out.get("atr", pd.Series(0.0, index=out.index))).astype(float)
    vol_floor = max(config.vol_floor, vol.median() * 0.25)
    vol_norm = (1.0 / (vol + vol_floor)).clip(upper=10.0)

    weights = config
    raw_score = (
        weights.rsi_weight * rsi_z +
        weights.overshoot_weight * overshoot_z +
        weights.zscore_weight * zscore_rev +
        weights.signal_weight * rev_signal_z +
        weights.accel_weight * accel
    )

    raw_score = raw_score * vol_norm

    out["reversion_score"] = pd.Series(np.tanh(raw_score)).ewm(
        alpha=config.smoothing_alpha, adjust=False
    ).mean()
    return out
