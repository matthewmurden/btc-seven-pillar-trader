import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------

def _sanitize(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _robust_norm(series: pd.Series) -> pd.Series:
    """Median/IQR logistic scaling → stable across regimes."""
    s = series.replace([np.inf, -np.inf], np.nan)
    med = s.median(skipna=True)
    iqr = s.quantile(0.75) - s.quantile(0.25)
    if iqr is None or pd.isna(iqr) or iqr == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    z = (s - med) / (iqr * 1.349)
    return 1 / (1 + np.exp(-z))


def _softmax(a, axis=-1):
    exp_a = np.exp(a - np.max(a, axis=axis, keepdims=True))
    return exp_a / exp_a.sum(axis=axis, keepdims=True)


# ------------------------------------------------------------
# Elite Volatility Excursion Engine (MFE Potential)
# ------------------------------------------------------------

def compute_volatility_excursion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elite MFE extension estimator.

    Uses:
        - multi-horizon realized volatility shape & curvature
        - ATR level + ATR acceleration
        - volatility compression acceleration
        - volume–volatility synergy (breakout predictor)
        - wick/single-bar structure (momentum health)
        - regime-aware scoring (implicit through vol shape)

    Output:
        Adds:
            mfe_potential_score ∈ [0, 1]
    """

    if "mfe_potential_score" in df.columns:
        raise ValueError("mfe_potential_score already exists; refusing to overwrite.")

    w = df.copy()

    # -------------------------
    # Core Vol Inputs
    # -------------------------
    atr = _sanitize(w.get("ATR_14", pd.Series(index=w.index))).fillna(0)
    atr_slope = _sanitize(w.get("ATR_slope", pd.Series(index=w.index))).fillna(0)

    rv5 = _sanitize(w.get("realized_vol_5", pd.Series(index=w.index))).fillna(0)
    rv15 = _sanitize(w.get("realized_vol_15", pd.Series(index=w.index))).fillna(0)
    rv30 = _sanitize(w.get("realized_vol_30", pd.Series(index=w.index))).fillna(0)

    comp = _sanitize(w.get("volatility_compression_score", pd.Series(index=w.index))).fillna(0)
    spread = _sanitize(w.get("spread", pd.Series(index=w.index))).fillna(0)
    wick = _sanitize(w.get("wick_ratio", pd.Series(index=w.index))).fillna(0.5)
    volume_exp = _sanitize(w.get("volume_expansion_5", pd.Series(index=w.index))).fillna(1.0)

    # -------------------------
    # Volatility Shape & Curvature
    # -------------------------
    slope_1 = rv5 - rv15          # short-term vol acceleration
    slope_2 = rv15 - rv30         # mid-term vol momentum
    curvature = slope_1 - slope_2 # convexity = "ignition" signal

    # -------------------------
    # Volatility acceleration signals
    # -------------------------
    atr_normalised = _robust_norm(atr)
    atr_acc = 0.5 * (np.tanh(atr_slope) + 1)

    comp_inv = 1 - _robust_norm(comp)
    comp_acc = comp_inv.diff().fillna(0)
    comp_acc_norm = 0.5 * (np.tanh(comp_acc) + 1)

    # volume-volatility synergy
    vov_synergy = _robust_norm(volume_exp * rv5)

    # wick health (momentum)
    wick_health = (1 - np.abs(wick - 0.5) / 0.5).clip(0, 1)

    # spread as activity proxy
    spread_norm = _robust_norm(spread)

    # normalize slopes and curvature
    slope_norm = _robust_norm(slope_1 + 0.5 * slope_2)
    curve_norm = _robust_norm(curvature)

    # -------------------------
    # Softmax large-move classifier
    # -------------------------
    # 0 = low extension
    # 1 = medium extension
    # 2 = high extension (true MFE potential)

    logits = np.vstack([
        # LOW MFE
        -(slope_norm + curve_norm + vov_synergy)
        + comp
        + (1 - spread_norm),

        # MEDIUM MFE
        0.6 * atr_normalised
        + 0.5 * slope_norm
        + 0.5 * vov_synergy
        + 0.4 * wick_health,

        # HIGH MFE (extension)
        1.3 * slope_norm
        + 1.4 * curve_norm
        + 1.2 * vov_synergy
        + 0.7 * atr_acc
        + 0.6 * comp_acc_norm
        + 0.4 * spread_norm
    ]).T

    probs = _softmax(logits, axis=1)
    high_prob = probs[:, 2]

    w["mfe_potential_score"] = np.clip(high_prob, 0, 1)
    return w