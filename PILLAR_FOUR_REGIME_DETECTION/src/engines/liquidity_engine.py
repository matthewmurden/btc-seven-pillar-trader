import numpy as np
import pandas as pd

LOW_TH = 0.3
HIGH_TH = 0.7
VOLUME_DROUGHT_PENALTY = 0.5


def _minmax(series: pd.Series) -> pd.Series:
    smin, smax = series.min(), series.max()
    if smax - smin == 0:
        return pd.Series(0.0, index=series.index)
    return (series - smin) / (smax - smin)


def compute_liquidity_engine(df: pd.DataFrame) -> pd.DataFrame:
    delta_persistence = df.get("delta_persistence_feat", df.get("delta_persistence", pd.Series(0, index=df.index)))
    volume_drought = df.get("volume_drought", pd.Series(0, index=df.index))
    order_proxy = df.get("order_book_proxy", df.get("order_imbalance", pd.Series(0, index=df.index)))

    raw_score = delta_persistence - volume_drought * VOLUME_DROUGHT_PENALTY + order_proxy * 0.1
    liquidity_score = _minmax(raw_score).fillna(0)

    regime = pd.Series("normal", index=df.index)
    regime = regime.mask(liquidity_score < LOW_TH, "drought")
    regime = regime.mask(liquidity_score > HIGH_TH, "expansion")

    return pd.DataFrame(
        {
            "liquidity_regime": regime,
            "liquidity_score": liquidity_score,
        },
        index=df.index,
    )
