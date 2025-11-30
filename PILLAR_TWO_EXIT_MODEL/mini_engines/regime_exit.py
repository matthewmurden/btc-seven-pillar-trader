import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------

def _sanitize(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _robust_norm(series: pd.Series) -> pd.Series:
    """Median-IQR logistic scaling."""
    s = series.replace([np.inf, -np.inf], np.nan)
    med = s.median(skipna=True)
    iqr = s.quantile(0.75) - s.quantile(0.25)
    if iqr is None or pd.isna(iqr) or iqr == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    z = (s - med) / (iqr * 1.349)
    return 1 / (1 + np.exp(-z))


def _softmax(a: np.ndarray, axis: int = -1) -> np.ndarray:
    exp_a = np.exp(a - np.max(a, axis=axis, keepdims=True))
    return exp_a / exp_a.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------
# Regime Exit Engine — Highest Standard
# ---------------------------------------------------------

def compute_regime_exit(df: pd.DataFrame, hysteresis: float = 0.25) -> pd.DataFrame:
    """
    Elite quant regime classifier for exits:
        0 = range
        1 = trend
        2 = breakout

    Inputs:
        - volatility_curve_5/15/30
        - drift_direction
        - volume_expansion_5
        - spread, wick_ratio
        - distance_to_high / low (implicitly via wick_ratio & drift)

    Features Used:
        - Volatility slope (breakout signature)
        - Volume–volatility synergy
        - Drift stability (trend signature)
        - Spread expansion (breakout) / compression (range)
        - Wick balance (trend health)

    Output:
        - exit_regime : {0,1,2}
        - exit_regime_confidence : [0,1]
    """

    if "exit_regime" in df.columns:
        raise ValueError("exit_regime already exists; refusing to overwrite.")

    w = df.copy()

    # ---- Raw Features -------------------------------------------------------
    drift = _sanitize(w.get("drift_direction", pd.Series(index=w.index))).fillna(0)
    vol5 = _sanitize(w.get("volatility_curve_5", pd.Series(index=w.index))).fillna(0)
    vol15 = _sanitize(w.get("volatility_curve_15", pd.Series(index=w.index))).fillna(0)
    vol30 = _sanitize(w.get("volatility_curve_30", pd.Series(index=w.index))).fillna(0)

    volume_exp = _sanitize(w.get("volume_expansion_5", pd.Series(index=w.index))).fillna(0)
    wick_ratio = _sanitize(w.get("wick_ratio", pd.Series(index=w.index))).fillna(0.5)
    spread = _sanitize(w.get("spread", pd.Series(index=w.index))).fillna(0)

    # ---- Derived Features ----------------------------------------------------
    vol_slope_short = vol5 - vol15     # breakout signature
    vol_slope_long = vol15 - vol30     # regime-shift signal
    vol_slope = vol_slope_short + 0.5 * vol_slope_long

    drift_strength = drift.abs()
    drift_stability = drift_strength / (1 + vol15)  # consistent drift beats noise

    # volume-volatility synergy = core breakout detector
    vov_synergy = _robust_norm(volume_exp * vol5)

    spread_norm = _robust_norm(spread)
    wick_balance = 1 - np.abs(wick_ratio - 0.5) * 2  # centered around 0.5

    # ---- Normalised Inputs ---------------------------------------------------
    norm_drift = _robust_norm(drift_strength)
    norm_vol_slope = _robust_norm(vol_slope)
    norm_vov = _robust_norm(vov_synergy)
    norm_spread = _robust_norm(spread)
    norm_wick = wick_balance.clip(0, 1)

    # ---- Scoring Logic -------------------------------------------------------
    # 3-class softmax logits (un-normalized)
    logits = np.vstack([
        # RANGE SCORE (class 0)
        -(norm_vol_slope + norm_drift)       # low drift, low slope
        + 0.7 * (1 - norm_spread)            # compressed spread
        + 0.5 * norm_wick,                   # neutral wick structure

        # TREND SCORE (class 1)
        1.2 * norm_drift                     # stable direction
        + 0.6 * norm_wick                    # wick symmetry
        + 0.4 * (1 - norm_vol_slope)         # stable vol

        ,

        # BREAKOUT SCORE (class 2)
        1.4 * norm_vol_slope                 # rising vol slope
        + 1.3 * norm_vov                     # volume-vol synergy
        + 0.6 * norm_spread                  # spread expansion
    ]).T

    # ---- Softmax for probabilities ------------------------------------------
    probs = _softmax(logits, axis=1)
    regime_class = probs.argmax(axis=1)
    regime_conf = probs.max(axis=1)

    # ---- Hysteresis / smoothing ---------------------------------------------
    # Avoid rapid flip-flops: only switch regime if confidence high
    smoothed = regime_class.copy()
    for i in range(1, len(smoothed)):
        if regime_conf[i] < hysteresis:
            smoothed[i] = smoothed[i - 1]

    w["exit_regime"] = smoothed.astype(int)
    w["exit_regime_confidence"] = regime_conf

    return w