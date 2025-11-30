import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Utility scalers
# ---------------------------------------------------------

def _sanitize(series: pd.Series) -> pd.Series:
    """Coerce to float and remove infs."""
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _robust_norm(series: pd.Series) -> pd.Series:
    """
    Robust 0–1 scaling using median/IQR.
    Much more stable than min–max for fat-tailed market data.
    """
    s = series.replace([np.inf, -np.inf], np.nan)
    med = s.median(skipna=True)
    iqr = s.quantile(0.75) - s.quantile(0.25)

    if iqr is None or pd.isna(iqr) or iqr == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)

    z = (s - med) / (iqr * 1.349)  # Gaussian consistency factor
    return 1 / (1 + np.exp(-z))     # logistic squash to (0,1)


def _bounded_log_norm(series: pd.Series) -> pd.Series:
    """
    For ATR/log-vol features: stabilises regime shifts.
    """
    s = series.clip(lower=1e-12).replace([np.inf, -np.inf], np.nan)
    log_s = np.log(s)
    return _robust_norm(log_s)


# ---------------------------------------------------------
# Main computation
# ---------------------------------------------------------

def compute_adverse_move(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elite-level adverse move risk estimator.

    Components:
        - Sell pressure (imbalance, delta)
        - Volume stress (volume_expansion_5)
        - Liquidity vulnerability (distance_to_low)
        - Wick asymmetry (upper wick dominance)
        - Regime volatility (ATR_14)

    Output:
        mae_risk_score ∈ [0, 1], stationary, logistic-squashed,
        robust to outliers, and stable across datasets.
    """
    if "mae_risk_score" in df.columns:
        raise ValueError("mae_risk_score already exists; refusing to overwrite.")

    w = df.copy()

    # ------------------------
    # Extract & sanitize
    # ------------------------
    imbalance = _sanitize(w.get("imbalance", pd.Series(index=w.index)))
    delta = _sanitize(w.get("delta", pd.Series(index=w.index)))
    vol_exp = _sanitize(w.get("volume_expansion_5", pd.Series(index=w.index)))
    dist_low = _sanitize(w.get("distance_to_low", pd.Series(index=w.index)))
    wick_imb = _sanitize(w.get("wick_imbalance_score", pd.Series(index=w.index)))
    atr = _sanitize(w.get("ATR_14", pd.Series(index=w.index)))

    # ------------------------
    # Normalisations
    # ------------------------

    # Sell pressure
    risk_imbalance = _robust_norm(-imbalance)   # more negative = risk
    risk_delta = _robust_norm(-delta)

    # Volume stress
    risk_volume = _robust_norm(vol_exp)

    # Liquidity vulnerability: log proximity
    risk_distance = 1 - _robust_norm(np.log1p(dist_low.clip(lower=0)))

    # Wick imbalance: map [-1,1] → [0,1]
    wick_norm = (wick_imb.clip(-1, 1) + 1) / 2
    risk_wick = _robust_norm(wick_norm)

    # Volatility stress
    risk_atr = _bounded_log_norm(atr)

    # ------------------------
    # Weighting (softmax normalised)
    # ------------------------
    raw_weights = np.array([0.20, 0.20, 0.20, 0.20, 0.10, 0.10])
    softmax_weights = np.exp(raw_weights) / np.exp(raw_weights).sum()

    # Unpack weights
    w_imb, w_del, w_vol, w_dist, w_wick, w_atr = softmax_weights

    # ------------------------
    # Final risk score
    # ------------------------
    score = (
        w_imb * risk_imbalance
        + w_del * risk_delta
        + w_vol * risk_volume
        + w_dist * risk_distance
        + w_wick * risk_wick
        + w_atr * risk_atr
    )

    w["mae_risk_score"] = score.clip(0, 1).fillna(0.0)
    return w