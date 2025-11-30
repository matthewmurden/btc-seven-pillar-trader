import numpy as np
import pandas as pd


def _sanitize(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def compute_regime_exit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify exit environment into range (0), trend (1), or breakout (2).

    Inputs:
        volatility_curve_5/15/30, drift_direction, volume_expansion_5, wick_ratio, spread.
    Rules:
        - Breakout: strong drift + high short-term volatility + elevated volume.
        - Trend: moderate drift + stable volatility + balanced wick_ratio.
        - Range: default fallback when neither breakout nor trend triggers.
    Output:
        Adds column `exit_regime` with integer classes {0,1,2}, NaN-safe.
    """
    if "exit_regime" in df.columns:
        raise ValueError("exit_regime already exists; refusing to overwrite.")

    working = df.copy()

    drift_strength = _sanitize(working.get("drift_direction", pd.Series(index=working.index))).abs().fillna(0)
    vol5 = _sanitize(working.get("volatility_curve_5", pd.Series(index=working.index))).fillna(0)
    vol15 = _sanitize(working.get("volatility_curve_15", pd.Series(index=working.index))).fillna(0)
    vol30 = _sanitize(working.get("volatility_curve_30", pd.Series(index=working.index))).fillna(0)
    volume_exp = _sanitize(working.get("volume_expansion_5", pd.Series(index=working.index))).fillna(0)
    wick_ratio = _sanitize(working.get("wick_ratio", pd.Series(index=working.index))).fillna(0.5)
    spread = _sanitize(working.get("spread", pd.Series(index=working.index))).fillna(0)

    drift_high = drift_strength.quantile(0.75)
    drift_mid = drift_strength.quantile(0.5)
    vol5_high = vol5.quantile(0.75)
    vol15_mid = vol15.quantile(0.5)
    vol_exp_thresh = volume_exp.quantile(0.6)
    spread_low = spread.quantile(0.4)

    breakout = (drift_strength >= drift_high) & (vol5 >= vol5_high) & (volume_exp >= vol_exp_thresh)
    trend = (
        ~breakout
        & (drift_strength >= drift_mid)
        & (vol15 >= vol15_mid)
        & wick_ratio.between(0.3, 0.7)
    )
    range_mask = ~(breakout | trend) | ((spread <= spread_low) & (volume_exp < vol_exp_thresh))

    exit_regime = pd.Series(0, index=working.index, dtype=int)
    exit_regime = exit_regime.where(~breakout, 2)
    exit_regime = exit_regime.where(~trend, 1)
    exit_regime = exit_regime.where(~range_mask, 0)

    working["exit_regime"] = exit_regime.fillna(0).astype(int)
    return working
