import numpy as np
import pandas as pd


def _sanitize(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _min_max_norm(series: pd.Series) -> pd.Series:
    finite = series.replace([np.inf, -np.inf], np.nan)
    min_val = finite.min(skipna=True)
    max_val = finite.max(skipna=True)
    span = max_val - min_val
    if span is None or pd.isna(span) or span == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (finite - min_val) / span


def compute_volatility_excursion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate favourable move potential (MFE) using only current-bar volatility context.

    Inputs:
        ATR_14, ATR_slope, realized_vol_5/15/30, volatility_compression_score,
        spread, wick_ratio, volume_expansion_5 (optional).
    Logic:
        - Higher ATR_14 and positive ATR_slope increase score.
        - Lower compression (vol20 / vol100) increases score.
        - Higher realized volatility suggests room for extension.
        - Positive curvature of realized vols (discrete second derivative) increases score.
        - Higher volume expansion increases score.
        - Moderate wick_ratio is preferred; extreme wicks reduce score.
        - Spread contributes additively as an additional activity proxy.
    Output:
        Adds column `mfe_potential_score` in [0, 1], NaN-safe and clipped.
    """
    if "mfe_potential_score" in df.columns:
        raise ValueError("mfe_potential_score already exists; refusing to overwrite.")

    working = df.copy()

    atr_norm = _min_max_norm(_sanitize(working.get("ATR_14", pd.Series(index=working.index))))
    slope = _sanitize(working.get("ATR_slope", pd.Series(index=working.index)))
    slope_norm = 0.5 * (np.tanh(slope) + 1)

    comp = _min_max_norm(
        _sanitize(working.get("volatility_compression_score", pd.Series(index=working.index)))
    )
    comp_inverted = 1 - comp

    vol5 = _sanitize(working.get("realized_vol_5", pd.Series(index=working.index)))
    vol15 = _sanitize(working.get("realized_vol_15", pd.Series(index=working.index)))
    vol30 = _sanitize(working.get("realized_vol_30", pd.Series(index=working.index)))
    vol_mix = (vol5 + vol15 + vol30) / 3
    vol_norm = _min_max_norm(vol_mix)

    # Discrete curvature approximation: second derivative of realized vols.
    curvature = vol5 - 2 * vol15 + vol30
    curvature_norm = _min_max_norm(curvature)

    spread_norm = _min_max_norm(_sanitize(working.get("spread", pd.Series(index=working.index))))

    volume_exp_norm = _min_max_norm(
        _sanitize(working.get("volume_expansion_5", pd.Series(index=working.index)))
    )

    wick = _sanitize(working.get("wick_ratio", pd.Series(index=working.index)))
    wick_penalty = 1 - (np.abs(wick - 0.5) / 0.5)
    wick_penalty = wick_penalty.clip(0, 1).fillna(0.5)

    score = (
        0.25 * atr_norm
        + 0.15 * slope_norm
        + 0.15 * comp_inverted
        + 0.10 * vol_norm
        + 0.10 * spread_norm
        + 0.10 * wick_penalty
        + 0.10 * volume_exp_norm
        + 0.05 * curvature_norm
    )

    working["mfe_potential_score"] = score.clip(0, 1).fillna(0.0)
    return working
