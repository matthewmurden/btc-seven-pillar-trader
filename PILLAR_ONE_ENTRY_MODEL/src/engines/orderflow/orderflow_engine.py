import numpy as np
import pandas as pd

from .orderflow_engine_config import DEFAULT_ORDERFLOW_CONFIG, OrderflowConfig


def _robust_z(series: pd.Series, clip: float) -> pd.Series:
    s = series.astype(float)
    med = s.median()
    iqr = s.quantile(0.75) - s.quantile(0.25)
    scale = iqr * 0.7413
    if not np.isfinite(scale) or scale < 1e-9:
        scale = s.std()
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    z = (s - med) / scale
    return z.clip(-clip, clip).fillna(0.0)


def compute_orderflow_engine_score(
    df: pd.DataFrame, config: OrderflowConfig = DEFAULT_ORDERFLOW_CONFIG
) -> pd.DataFrame:
    """
    Build an order flow score from thrust, imbalance, sweeps, absorption, and
    cumulative volume imbalance with volatility weighting.
    """
    out = df.copy()

    delta = _robust_z(out["delta"], config.clip_value)
    raw_imb = _robust_z(out["of_raw_imbalance"], config.clip_value)
    norm_imb = _robust_z(out["of_imbalance"], config.clip_value)

    sweep_up = out.get("sweep_up", pd.Series(0.0, index=out.index)).astype(float)
    sweep_down = out.get("sweep_down", pd.Series(0.0, index=out.index)).astype(float)
    sweep_score = _robust_z(sweep_up - sweep_down, config.clip_value)

    tbv = out.get("taker_buy_vol", pd.Series(0.0, index=out.index)).astype(float)
    tsv = out.get("taker_sell_qty", pd.Series(0.0, index=out.index)).astype(float)
    cum_imbalance = _robust_z(tbv - tsv, config.clip_value)

    # Absorption: low delta while price stable (small candle_range)
    candle_range = out.get("candle_range", pd.Series(1.0, index=out.index)).astype(float)
    stable_price = (candle_range / (candle_range.median() + 1e-6)).clip(0.0, 5.0)
    absorption = _robust_z(-delta.abs() * (1.0 / (stable_price + 1e-6)), config.clip_value)

    vol = out.get("volatility", out.get("atr", pd.Series(0.0, index=out.index))).astype(float)
    vol_floor = max(config.vol_floor, vol.median() * 0.25)
    vol_norm = (1.0 / (vol + vol_floor)).clip(upper=10.0)

    weights = config
    raw_score = (
        weights.thrust_weight * delta +
        weights.imbalance_weight * norm_imb +
        weights.sweep_weight * sweep_score +
        weights.absorption_weight * absorption +
        weights.cum_imbalance_weight * cum_imbalance +
        weights.vol_weight * _robust_z(vol, config.clip_value)
    )

    raw_score = raw_score * vol_norm

    out["orderflow_score"] = pd.Series(np.tanh(raw_score)).ewm(
        alpha=config.smoothing_alpha, adjust=False
    ).mean()
    return out
