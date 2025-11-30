import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Utility scalers
# ---------------------------------------------------------

def _sanitize(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _robust_norm(series: pd.Series) -> pd.Series:
    """Robust logistic normalisation using median/IQR."""
    s = series.replace([np.inf, -np.inf], np.nan)
    med = s.median(skipna=True)
    iqr = s.quantile(0.75) - s.quantile(0.25)
    if iqr is None or pd.isna(iqr) or iqr == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    z = (s - med) / (iqr * 1.349)
    return 1 / (1 + np.exp(-z))


def _bell_curve_preference(series: pd.Series) -> pd.Series:
    """
    Preference for mid-range volatility (not too high or low).
    Gaussian kernel centered on the median.
    """
    s = series.replace([np.inf, -np.inf], np.nan)
    med = s.median(skipna=True)
    iqr = (s.quantile(0.75) - s.quantile(0.25)) or 1.0
    z = (s - med) / (iqr * 1.349)
    return np.exp(-(z ** 2))  # peak=1 at median, decays outward


def _rsi_midrange_kernel(rsi: pd.Series) -> pd.Series:
    """
    RSI-based continuation strength:
    - Peak continuation at RSI 40–55
    - Penalize oversold (<30) and overbought (>70)
    Uses bell-shaped kernel centered ~47.5.
    """
    target = 47.5
    width = 20  # bandwidth
    z = (rsi - target) / width
    return np.exp(-(z ** 2))  # maps to (0,1)


# ---------------------------------------------------------
# Main continuation engine
# ---------------------------------------------------------

def compute_favourable_move(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elite-level trend continuation estimator.

    Components:
        - Momentum (robust normalised)
        - Trend strength (robust normalised)
        - Structure score (robust normalised)
        - Volatility moderation (bell curve around median vol)
        - RSI midrange continuation kernel (realistic shape)
        - Momentum–Trend–Structure synergy term

    Output:
        trend_continuation_score ∈ [0,1]
    """
    if "trend_continuation_score" in df.columns:
        raise ValueError("trend_continuation_score already exists; refusing to overwrite.")

    w = df.copy()

    # ------------------------
    # Extract & sanitize
    # ------------------------
    momentum = _robust_norm(_sanitize(w.get("momentum_score", pd.Series(index=w.index))))
    trend = _robust_norm(_sanitize(w.get("trend_score", pd.Series(index=w.index))))
    structure = _robust_norm(_sanitize(w.get("structure_score", pd.Series(index=w.index))))
    vol_raw = _sanitize(w.get("volatility_score", pd.Series(index=w.index)))
    rsi = _sanitize(w.get("short_term_exhaustion", pd.Series(index=w.index)))

    # ------------------------
    # Derived kernels
    # ------------------------
    vol_balance = _bell_curve_preference(vol_raw)
    rsi_kernel = _rsi_midrange_kernel(rsi)

    # Synergy: momentum × trend × structure
    synergy = (momentum * trend * structure) ** (1/3)  # geometric mean

    # ------------------------
    # Softmax-normalised weights
    # ------------------------
    raw_w = np.array([0.22, 0.22, 0.18, 0.14, 0.14, 0.10])
    weights = np.exp(raw_w) / np.exp(raw_w).sum()

    w_mom, w_tr, w_str, w_vol, w_rsi, w_syn = weights

    # ------------------------
    # Final Score
    # ------------------------
    score = (
        w_mom * momentum +
        w_tr * trend +
        w_str * structure +
        w_vol * vol_balance +
        w_rsi * rsi_kernel +
        w_syn * synergy
    )

    w["trend_continuation_score"] = score.clip(0, 1).fillna(0.0)
    return w