import numpy as np
import pandas as pd
from typing import Optional

from .signal_decay_config import DEFAULT_SIGNAL_DECAY_CONFIG, SignalDecayConfig


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


def _compute_event_state(values: np.ndarray, threshold: float) -> tuple:
    """
    Compute age since last meaningful event and the last event value.
    An event is defined as |value| > threshold.
    """
    n = len(values)
    idx = np.arange(n, dtype=float)
    mask = np.abs(values) > threshold
    last_idx = np.where(mask, idx, -1)
    last_idx = np.maximum.accumulate(last_idx)
    age = idx - last_idx
    age[last_idx < 0] = np.inf  # no event seen yet
    event_values = np.where(mask, values, np.nan)
    last_val = pd.Series(event_values).ffill().fillna(0.0).to_numpy()
    return age, last_val


def _decay_series(
    series: pd.Series,
    decay_type: str,
    decay_rate: float,
    valid_bars: int,
    activation_threshold: float,
    eps: float,
    rate_scale: Optional[np.ndarray] = None,
    valid_scale: Optional[np.ndarray] = None,
) -> pd.Series:
    values = series.astype(float).to_numpy()
    age, last_val = _compute_event_state(values, activation_threshold)

    eff_rate = decay_rate * (rate_scale if rate_scale is not None else 1.0)
    eff_valid = valid_bars * (valid_scale if valid_scale is not None else 1.0)
    eff_valid = np.maximum(eff_valid, 1.0)

    if decay_type == "exp":
        factors = np.exp(-eff_rate * age)
    elif decay_type == "linear":
        factors = np.maximum(0.0, 1.0 - (age / (eff_valid + eps)))
    else:
        raise ValueError(f"Unsupported decay type: {decay_type}")

    factors = np.where(age <= eff_valid, factors, 0.0)
    factors = np.where(np.isfinite(age), factors, 0.0)

    decayed = last_val * factors
    return pd.Series(decayed, index=series.index)


def apply_signal_decay(df: pd.DataFrame, config: SignalDecayConfig = DEFAULT_SIGNAL_DECAY_CONFIG) -> pd.DataFrame:
    """
    Applies decay curves and validity windows to all engine & meta scores.
    Returns a new DataFrame with *_decayed versions of each score.
    """
    out = df.copy()
    eps = config.epsilon

    # Support columns for adaptive decay
    sweep_up = out.get("sweep_up", pd.Series(0.0, index=out.index)).astype(float)
    sweep_down = out.get("sweep_down", pd.Series(0.0, index=out.index)).astype(float)
    sweep_intensity = _robust_z(sweep_up - sweep_down, config.sweep_clip)
    sweep_scale = (1.0 - config.sweep_decay_reducer * sweep_intensity.clip(lower=0.0)).clip(0.2, 2.0)

    structure_signal = out.get("structure_signal", pd.Series(0.0, index=out.index)).astype(float)
    structure_stability = _robust_z(structure_signal, config.structure_clip)
    structure_scale = (1.0 - config.structure_stability_boost * structure_stability.clip(lower=0.0)).clip(0.2, 2.0)

    # Engine decays
    out["momentum_score_decayed"] = _decay_series(
        out.get("momentum_score", pd.Series(0.0, index=out.index)),
        config.momentum_decay_type,
        config.momentum_decay_rate,
        config.momentum_valid_bars,
        config.momentum_activation,
        eps,
    ).clip(-config.max_clip, config.max_clip)

    out["trend_score_decayed"] = _decay_series(
        out.get("trend_score", pd.Series(0.0, index=out.index)),
        config.trend_decay_type,
        config.trend_decay_rate,
        config.trend_valid_bars,
        config.trend_activation,
        eps,
    ).clip(-config.max_clip, config.max_clip)

    out["volatility_score_decayed"] = out.get("volatility_score", pd.Series(0.0, index=out.index)).astype(float)

    out["reversion_score_decayed"] = _decay_series(
        out.get("reversion_score", pd.Series(0.0, index=out.index)),
        config.reversion_decay_type,
        config.reversion_decay_rate,
        config.reversion_valid_bars,
        config.reversion_activation,
        eps,
    ).clip(-config.max_clip, config.max_clip)

    out["orderflow_score_decayed"] = _decay_series(
        out.get("orderflow_score", pd.Series(0.0, index=out.index)),
        config.orderflow_decay_type,
        config.orderflow_decay_rate,
        config.orderflow_valid_bars,
        config.orderflow_activation,
        eps,
        rate_scale=sweep_scale.to_numpy(),
    ).clip(-config.max_clip, config.max_clip)

    out["structure_score_decayed"] = _decay_series(
        out.get("structure_score", pd.Series(0.0, index=out.index)),
        config.structure_decay_type,
        config.structure_decay_rate,
        config.structure_valid_bars,
        config.structure_activation,
        eps,
        rate_scale=structure_scale.to_numpy(),
    ).clip(-config.max_clip, config.max_clip)

    out["pattern_score_decayed"] = _decay_series(
        out.get("pattern_score", pd.Series(0.0, index=out.index)),
        config.pattern_decay_type,
        config.pattern_decay_rate,
        config.pattern_valid_bars,
        config.pattern_activation,
        eps,
    ).clip(-config.max_clip, config.max_clip)

    # Meta features
    out["meta_agreement_score_decayed"] = _decay_series(
        out.get("agreement_score", pd.Series(0.0, index=out.index)),
        config.meta_agreement_type,
        config.meta_agreement_rate,
        config.meta_agreement_valid,
        config.meta_activation,
        eps,
    ).clip(-config.max_clip, config.max_clip)

    out["meta_conflict_score_decayed"] = _decay_series(
        out.get("conflict_score", pd.Series(0.0, index=out.index)),
        config.meta_conflict_type,
        config.meta_conflict_rate,
        config.meta_conflict_valid,
        config.meta_activation,
        eps,
    ).clip(-config.max_clip, config.max_clip)

    out["meta_volatility_weighted_momentum_decayed"] = _decay_series(
        out.get("volatility_weighted_momentum", pd.Series(0.0, index=out.index)),
        config.meta_vol_mom_type,
        config.meta_vol_mom_rate,
        config.meta_vol_mom_valid,
        config.meta_activation,
        eps,
    ).clip(-config.max_clip, config.max_clip)

    out["meta_trend_plus_orderflow_confirmation_decayed"] = _decay_series(
        out.get("trend_plus_orderflow_confirmation", pd.Series(0.0, index=out.index)),
        config.meta_trend_of_type,
        config.meta_trend_of_rate,
        config.meta_trend_of_valid,
        config.meta_activation,
        eps,
    ).clip(-config.max_clip, config.max_clip)

    out["meta_structure_aligned_momentum_decayed"] = _decay_series(
        out.get("structure_aligned_momentum", pd.Series(0.0, index=out.index)),
        config.meta_structure_mom_type,
        config.meta_structure_mom_rate,
        config.meta_structure_mom_valid,
        config.meta_activation,
        eps,
    ).clip(-config.max_clip, config.max_clip)

    out["meta_regime_adjusted_signal_strength_decayed"] = _decay_series(
        out.get("regime_adjusted_signal_strength", pd.Series(0.0, index=out.index)),
        config.meta_regime_strength_type,
        config.meta_regime_strength_rate,
        config.meta_regime_strength_valid,
        config.meta_activation,
        eps,
    ).clip(-config.max_clip, config.max_clip)

    out["meta_horizon_consistency_score_decayed"] = _decay_series(
        out.get("horizon_consistency_score", pd.Series(0.0, index=out.index)),
        config.meta_horizon_type,
        config.meta_horizon_rate,
        config.meta_horizon_valid,
        config.meta_activation,
        eps,
    ).clip(-config.max_clip, config.max_clip)

    # Regime confidence smoothing only (state unchanged)
    regime_conf = out.get("regime_confidence", pd.Series(0.5, index=out.index)).astype(float)
    out["regime_confidence_decayed"] = regime_conf.ewm(alpha=config.regime_conf_alpha, adjust=False).mean()

    out = out.fillna(method="ffill").fillna(method="bfill")
    return out
