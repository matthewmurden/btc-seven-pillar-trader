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


def compute_path_shape(df: pd.DataFrame) -> pd.DataFrame:
    """
    Smoothness vs. choppiness estimator for future path shape.

    Inputs:
        micro_trend_entropy, bar_clustering_score, realized_vol_5.
    Logic:
        - Higher entropy and higher realized volatility lower the score.
        - Higher clustering raises the score.
        - Score = 0.4*(1 - entropy_norm) + 0.35*cluster_norm + 0.25*(1 - vol_norm).
    Output:
        Adds column `path_entropy_score` in [0, 1], NaN-safe and clipped.
    """
    if "path_entropy_score" in df.columns:
        raise ValueError("path_entropy_score already exists; refusing to overwrite.")

    working = df.copy()

    entropy_norm = _min_max_norm(_sanitize(working.get("micro_trend_entropy", pd.Series(index=working.index))))
    cluster_norm = _min_max_norm(_sanitize(working.get("bar_clustering_score", pd.Series(index=working.index))))
    vol_norm = _min_max_norm(_sanitize(working.get("realized_vol_5", pd.Series(index=working.index))))

    score = 0.40 * (1 - entropy_norm) + 0.35 * cluster_norm + 0.25 * (1 - vol_norm)
    working["path_entropy_score"] = score.clip(0, 1).fillna(0.0)
    return working
