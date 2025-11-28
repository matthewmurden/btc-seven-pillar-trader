import numpy as np
import pandas as pd

from .trend_engine_config import TrendConfig


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


def _rolling_slope(close: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling slope via covariance/variance trick (vectorized).
    Returns slope per period (normalized by index spacing).
    """
    x = pd.Series(np.arange(len(close), dtype=float), index=close.index)
    y = close.astype(float)

    roll_x = x.rolling(window, min_periods=window)
    roll_y = y.rolling(window, min_periods=window)

    sum_x = roll_x.sum()
    sum_y = roll_y.sum()
    sum_xy = (x * y).rolling(window, min_periods=window).sum()
    sum_x2 = (x * x).rolling(window, min_periods=window).sum()

    denom = (sum_x2 - (sum_x * sum_x) / window).replace(0, np.nan)
    numer = (sum_xy - (sum_x * sum_y) / window)
    slope = numer / denom
    return slope.fillna(0.0)


def compute_trend_engine_score(df: pd.DataFrame, config: TrendConfig) -> pd.DataFrame:
    """
    Build a robust trend score from slope, HTF alignment, strength, efficiency,
    noise ratio, and MA distance. Vectorized and volatility-aware.
    """
    out = df.copy()

    close = out["close"].astype(float)
    volatility = out["volatility"].astype(float)
    vol_floor = max(config.vol_floor, volatility.median() * 0.25)
    vol_norm = (1.0 / (volatility + vol_floor)).clip(upper=10.0)

    # Rolling slope normalized by volatility
    slope_raw = _rolling_slope(close, config.slope_window)
    slope_vol_norm = slope_raw * vol_norm
    slope_z = _robust_z(slope_vol_norm, config.z_clip).clip(-config.slope_clip, config.slope_clip)

    # Trend strength: normalized R² proxy (trend_strength given) with stability floor
    strength = out["trend_strength"].astype(float) / 100.0
    strength = strength.clip(lower=config.r2_floor)
    strength_z = _robust_z(strength, config.z_clip)

    # Efficiency ratio (clipped 0-1) minus noise ratio (ATR / range proxy)
    efficiency = out.get("trend_efficiency", pd.Series(0.0, index=out.index)).astype(float).clip(0.0, 1.0)
    atr = out.get("atr", pd.Series(0.0, index=out.index)).astype(float)
    candle_range = out.get("candle_range", pd.Series(1.0, index=out.index)).astype(float)
    noise_ratio = (atr / (candle_range + config.noise_floor)).clip(0.0, 10.0)
    noise_z = _robust_z(noise_ratio, config.z_clip)
    efficiency_z = _robust_z(efficiency, config.z_clip)

    # MA distance normalized
    ma = close.rolling(config.ma_window, min_periods=config.ma_window).mean()
    ma_dist = (close - ma) / (ma.abs() + config.noise_floor)
    ma_dist_z = _robust_z(ma_dist, config.ma_dist_clip)

    # HTF alignment
    htf = out.get("htf_alignment", pd.Series(0.0, index=out.index)).astype(float)
    htf_z = _robust_z(htf, config.z_clip)

    # Dynamic weighting: tilt more to slope/efficiency in high volatility
    vol_pct = out.get("vol_percentile", pd.Series(0.5, index=out.index)).astype(float).clip(0.0, 1.0)
    slope_w = config.slope_weight * (1.0 + 0.4 * (vol_pct - 0.5))
    eff_w = config.efficiency_weight * (1.0 + 0.3 * (vol_pct - 0.5))

    weights = {
        "slope": slope_w,
        "strength": config.strength_weight,
        "efficiency": eff_w,
        "noise": config.noise_weight,
        "ma": config.ma_dist_weight,
        "htf": config.htf_weight,
    }

    trend_raw = (
        weights["slope"] * slope_z +
        weights["strength"] * strength_z +
        weights["efficiency"] * efficiency_z -
        weights["noise"] * noise_z +
        weights["ma"] * ma_dist_z +
        weights["htf"] * htf_z
    )

    out["trend_score"] = pd.Series(np.tanh(trend_raw)).ewm(alpha=config.smooth_alpha, adjust=False).mean()
    out["trend_score"] = out["trend_score"].ffill().bfill()
    return out
