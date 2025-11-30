import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def _sanitize(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _robust_norm(series: pd.Series) -> pd.Series:
    """Robust logistic scaling using median/IQR (quant standard)."""
    s = series.replace([np.inf, -np.inf], np.nan)
    med = s.median(skipna=True)
    iqr = s.quantile(0.75) - s.quantile(0.25)
    if iqr is None or pd.isna(iqr) or iqr == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    z = (s - med) / (iqr * 1.349)
    return 1 / (1 + np.exp(-z))


def _bell_curve(series: pd.Series) -> pd.Series:
    """
    Gaussian preference: favours values near the median,
    penalises extremes (either too low or too high).
    """
    s = series.replace([np.inf, -np.inf], np.nan)
    med = s.median(skipna=True)
    iqr = (s.quantile(0.75) - s.quantile(0.25)) or 1.0
    z = (s - med) / (iqr * 1.349)
    return np.exp(-(z ** 2))


# ---------------------------------------------------------
# Main engine
# ---------------------------------------------------------

def compute_path_shape(df: pd.DataFrame) -> pd.DataFrame:
    """
    Smoothness vs choppiness quant estimator.

    Inputs:
        - micro_trend_entropy  (market randomness)
        - bar_clustering_score (consecutive small-body bars)
        - realized_vol_5       (local volatility turbulence)

    Logic:
        - Smooth paths = low entropy, high clustering, low realized vol
        - Chaotic chop = high entropy OR high realized vol
        - Pullback regimes = moderate entropy + moderate volatility
          → handled by Gaussian bell kernels

    Output:
        path_entropy_score ∈ [0, 1] where:
            1 = very smooth / directional
            0 = very chaotic / noisy / turbulent
    """

    if "path_entropy_score" in df.columns:
        raise ValueError("path_entropy_score already exists; refusing to overwrite.")

    w = df.copy()

    # Sanitize inputs
    entropy_raw = _sanitize(w.get("micro_trend_entropy", pd.Series(index=w.index)))
    cluster_raw = _sanitize(w.get("bar_clustering_score", pd.Series(index=w.index)))
    vol_raw = _sanitize(w.get("realized_vol_5", pd.Series(index=w.index)))

    # Robust normalisations
    entropy_norm = _robust_norm(entropy_raw)
    cluster_norm = _robust_norm(cluster_raw)
    vol_norm = _robust_norm(vol_raw)

    # Higher entropy = worse → apply inverted bell to reward smoothness
    entropy_smoothness = 1 - _bell_curve(entropy_raw)

    # Cluster bell curve = clusters near trend get boosted
    cluster_strength = _bell_curve(cluster_raw)

    # Volatility: smooth paths prefer mid-low vol
    volatility_smoothness = 1 - _bell_curve(vol_raw)

    # Synergy between entropy and volatility:
    # chaotic only if BOTH are high simultaneously
    chaos_penalty = np.sqrt(entropy_norm * vol_norm)

    # Softmax weights for stability
    raw_w = np.array([0.28, 0.28, 0.22, 0.22])
    weights = np.exp(raw_w) / np.exp(raw_w).sum()
    w_ent, w_clu, w_vol, w_syn = weights

    # Final score (smoothness)
    score = (
        w_ent * entropy_smoothness +
        w_clu * cluster_strength +
        w_vol * volatility_smoothness +
        w_syn * (1 - chaos_penalty)
    )

    w["path_entropy_score"] = score.clip(0, 1).fillna(0.0)
    return w