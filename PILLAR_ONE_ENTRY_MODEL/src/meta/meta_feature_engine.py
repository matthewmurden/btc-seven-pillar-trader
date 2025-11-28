import numpy as np
import pandas as pd

from .meta_feature_engine_config import DEFAULT_META_FEATURE_CONFIG, MetaFeatureConfig


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


def _ema(data, alpha: float, index) -> pd.Series:
    return pd.Series(data, index=index).ewm(alpha=alpha, adjust=False).mean()


def _soft_regime_weights(regime_state: pd.Series, regime_conf: pd.Series, eps: float) -> pd.DataFrame:
    state = regime_state.round().clip(0, 2).astype(int)
    conf = regime_conf.clip(0.0, 1.0).fillna(0.5)
    base = np.stack(
        [
            (state == 0).astype(float),
            (state == 1).astype(float),
            (state == 2).astype(float),
        ],
        axis=1,
    )
    weights = base * conf.values[:, None]
    weights += ((1.0 - conf.values) / 3.0)[:, None]  # soften when confidence is low
    weights = weights / (weights.sum(axis=1, keepdims=True) + eps)
    return pd.DataFrame(weights, columns=["range", "trend", "breakout"], index=regime_state.index)


def compute_meta_features(df: pd.DataFrame, config: MetaFeatureConfig = DEFAULT_META_FEATURE_CONFIG) -> pd.DataFrame:
    """
    Deterministic meta-feature engine combining engine scores and model signals.
    Returns a DataFrame with all meta scores.
    """
    out = df.copy()

    # Pull base signals
    momentum = out.get("momentum_score", pd.Series(0.0, index=out.index)).astype(float)
    trend = out.get("trend_score", pd.Series(0.0, index=out.index)).astype(float)
    volatility = out.get("volatility_score", pd.Series(0.0, index=out.index)).astype(float)
    reversion = out.get("reversion_score", pd.Series(0.0, index=out.index)).astype(float)
    orderflow = out.get("orderflow_score", pd.Series(0.0, index=out.index)).astype(float)
    structure = out.get("structure_score", pd.Series(0.0, index=out.index)).astype(float)
    pattern = out.get("pattern_score", pd.Series(0.0, index=out.index)).astype(float)

    regime_state = out.get("regime_state", pd.Series(1, index=out.index)).astype(float)
    regime_conf = out.get("regime_confidence", pd.Series(0.5, index=out.index)).astype(float)

    p1 = out.get("p_up_1m", pd.Series(0.5, index=out.index)).astype(float).clip(0.0, 1.0)
    p5 = out.get("p_up_5m", pd.Series(0.5, index=out.index)).astype(float).clip(0.0, 1.0)
    p15 = out.get("p_up_15m", pd.Series(0.5, index=out.index)).astype(float).clip(0.0, 1.0)
    p_breakout = out.get("p_breakout", pd.Series(0.5, index=out.index)).astype(float).clip(0.0, 1.0)
    p_reversion_pred = out.get("p_reversion", pd.Series(0.5, index=out.index)).astype(float).clip(0.0, 1.0)

    weights = _soft_regime_weights(regime_state, regime_conf, config.epsilon)

    # Signs for agreement/conflict logic
    sign_mom = np.sign(momentum)
    sign_trend = np.sign(trend)
    sign_vol = np.sign(volatility)
    sign_rev = np.sign(reversion)
    sign_of = np.sign(orderflow)
    sign_struct = np.sign(structure)
    sign_pat = np.sign(pattern)

    # 1) Agreement score (regime-weighted)
    agree_trend = weights["trend"].values * (
        config.agreement_trend_weight * sign_trend +
        config.agreement_momentum_weight * sign_mom
    )
    agree_range = weights["range"].values * (
        config.agreement_reversion_weight * sign_rev +
        config.agreement_structure_weight * sign_struct
    )
    agree_breakout = weights["breakout"].values * (
        config.agreement_orderflow_weight * sign_of +
        config.agreement_volatility_weight * sign_vol +
        config.agreement_pattern_weight * sign_pat
    )

    agreement_numer = agree_trend + agree_range + agree_breakout
    agreement_denom = (
        weights["trend"].values * (config.agreement_trend_weight + config.agreement_momentum_weight) +
        weights["range"].values * (config.agreement_reversion_weight + config.agreement_structure_weight) +
        weights["breakout"].values * (
            config.agreement_orderflow_weight +
            config.agreement_volatility_weight +
            config.agreement_pattern_weight
        )
    )
    agreement_raw = agreement_numer / (agreement_denom + config.epsilon)
    agreement_score = _ema(
        np.tanh(_robust_z(pd.Series(agreement_raw), config.agreement_clip)),
        config.base_smoothing_alpha,
        out.index,
    )

    # 2) Conflict score (normalized [0, 1])
    engine_signs = np.vstack([
        sign_mom,
        sign_trend,
        sign_vol,
        sign_rev,
        sign_of,
        sign_struct,
        sign_pat,
    ])
    engine_weights = np.array(config.conflict_engine_weights, dtype=float).reshape(-1, 1)
    weighted_mean = (engine_signs * engine_weights).sum(axis=0) / (engine_weights.sum() + config.epsilon)
    dispersion = ((engine_signs - weighted_mean) ** 2 * engine_weights).sum(axis=0)
    dispersion = np.sqrt(dispersion / (engine_weights.sum() + config.epsilon))
    disp_norm = np.clip(dispersion / np.sqrt(engine_signs.shape[0]), 0.0, 1.0)

    conflict_raw = (
        config.conflict_mean_weight * (1.0 - np.abs(weighted_mean)) +
        config.conflict_dispersion_weight * disp_norm +
        config.regime_uncertainty_weight * (1.0 - regime_conf.values)
    )
    conflict_raw = np.clip(conflict_raw, 0.0, 1.0)
    conflict_score = _ema(conflict_raw, config.base_smoothing_alpha, out.index)

    # 3) Volatility-weighted momentum
    vol_mom = momentum * (1.0 + volatility)
    vol_mom = np.clip(vol_mom, -config.vol_momentum_clip, config.vol_momentum_clip)
    vol_mom = _robust_z(pd.Series(vol_mom), config.vol_momentum_clip)
    vol_mom_score = _ema(np.tanh(vol_mom), config.fast_smoothing_alpha, out.index)

    # 4) Trend + orderflow confirmation
    confirmation_base = _robust_z(pd.Series(trend + orderflow), config.confirmation_clip)
    confirmation_raw = np.tanh(config.confirmation_scale * confirmation_base)
    trend_of_confirmation = _ema(confirmation_raw, config.base_smoothing_alpha, out.index)

    # 5) Structure-aligned momentum
    struct_align_raw = momentum * structure * np.sign(trend)
    struct_align_norm = _robust_z(pd.Series(struct_align_raw), config.structure_momentum_clip)
    struct_align_score = _ema(np.tanh(struct_align_norm), config.fast_smoothing_alpha, out.index)

    # 6) Regime-adjusted signal strength
    trend_component = config.regime_boost_trend * 0.5 * (trend + momentum)
    range_component = config.regime_boost_range * (reversion + (p_reversion_pred - 0.5))
    breakout_component = config.regime_boost_breakout * 0.5 * (volatility + orderflow + (p_breakout - 0.5) * 2.0)

    strength_raw = (
        weights["trend"].values * trend_component +
        weights["range"].values * range_component +
        weights["breakout"].values * breakout_component
    )
    regime_adjusted_strength = _ema(np.tanh(strength_raw), config.base_smoothing_alpha, out.index)

    # 7) Horizon consistency (multi-horizon ML outputs)
    prob_matrix = np.vstack([p1, p5, p15]).T
    centered = prob_matrix - 0.5
    norm = np.linalg.norm(centered, axis=1)
    direction = np.sign(centered.mean(axis=1))
    reference = np.stack([direction, direction, direction], axis=1)
    cos_sim = (centered * reference).sum(axis=1) / (norm * np.sqrt(3.0) + config.epsilon)
    variance_penalty = np.exp(-config.horizon_variance_penalty * centered.var(axis=1))
    horizon_raw = cos_sim * variance_penalty
    horizon_consistency = _ema(
        np.tanh(config.horizon_scale * horizon_raw),
        config.horizon_smoothing_alpha,
        out.index,
    )

    meta = pd.DataFrame(
        {
            "agreement_score": agreement_score,
            "conflict_score": conflict_score,
            "volatility_weighted_momentum": vol_mom_score,
            "trend_plus_orderflow_confirmation": trend_of_confirmation,
            "structure_aligned_momentum": struct_align_score,
            "regime_adjusted_signal_strength": regime_adjusted_strength,
            "horizon_consistency_score": horizon_consistency,
        },
        index=df.index,
    )

    meta = meta.fillna(method="ffill").fillna(method="bfill")
    return meta
