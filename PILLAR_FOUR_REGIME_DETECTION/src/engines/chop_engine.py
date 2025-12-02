import pandas as pd

CHOP_THRESHOLD = 0.6  # deterministic cut-off


def _minmax(series: pd.Series) -> pd.Series:
    smin, smax = series.min(), series.max()
    if smax - smin == 0:
        return pd.Series(0.0, index=series.index)
    return (series - smin) / (smax - smin)


def compute_chop_engine(df: pd.DataFrame) -> pd.DataFrame:
    chop_idx = df.get("chop_index", pd.Series(0, index=df.index))
    entropy = df.get("entropy_feat", df.get("entropy", pd.Series(0, index=df.index)))

    entropy_norm = (entropy / (entropy.max() if entropy.max() != 0 else 1.0)).fillna(0)
    chop_score = _minmax(chop_idx).fillna(0)
    chop_flag = (chop_score > CHOP_THRESHOLD).astype(int)

    return pd.DataFrame(
        {
            "chop_flag": chop_flag,
            "chop_score": chop_score,
            "entropy_normalised": entropy_norm,
        },
        index=df.index,
    )
