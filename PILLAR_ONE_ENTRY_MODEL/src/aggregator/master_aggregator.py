import numpy as np
import pandas as pd
from typing import List

from .master_aggregator_config import DEFAULT_MASTER_AGGREGATOR_CONFIG, MasterAggregatorConfig


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


def _validate_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        formatted = ", ".join(missing)
        raise ValueError(f"MasterAggregator missing required columns: {formatted}")


def _tanh_clip(x, clip_value: float):
    return np.tanh(np.clip(x, -clip_value, clip_value))


def compute_master_aggregator(df: pd.DataFrame, config: MasterAggregatorConfig = DEFAULT_MASTER_AGGREGATOR_CONFIG) -> pd.DataFrame:
    """
    Regime-weighted deterministic fusion of engines, meta features, predictions, and decay-adjusted signals.
    """
    out = df.copy()

    engine_cols = [
        "momentum_score", "trend_score", "volatility_score", "reversion_score",
        "orderflow_score", "structure_score", "pattern_score",
        "momentum_score_decayed", "trend_score_decayed", "volatility_score_decayed", "reversion_score_decayed",
        "orderflow_score_decayed", "structure_score_decayed", "pattern_score_decayed",
    ]

    meta_cols = [
        "agreement_score", "conflict_score", "volatility_weighted_momentum",
        "trend_plus_orderflow_confirmation", "structure_aligned_momentum",
        "regime_adjusted_signal_strength", "horizon_consistency_score",
        "meta_agreement_score_decayed", "meta_conflict_score_decayed",
        "meta_volatility_weighted_momentum_decayed", "meta_trend_plus_orderflow_confirmation_decayed",
        "meta_structure_aligned_momentum_decayed", "meta_regime_adjusted_signal_strength_decayed",
        "meta_horizon_consistency_score_decayed",
    ]

    pred_cols = [
        "pred_1m_up", "pred_5m_up", "pred_15m_up", "pred_breakout", "pred_reversion",
    ]

    regime_cols = ["regime_state", "regime_confidence", "regime_confidence_decayed"]

    required = engine_cols + meta_cols + pred_cols + regime_cols
    _validate_columns(out, required)

    # Core signals (use decayed variants where applicable)
    trend_decayed = out["trend_score_decayed"].astype(float)
    momentum_decayed = out["momentum_score_decayed"].astype(float)
    vol_decayed = out["volatility_score_decayed"].astype(float)
    rev_decayed = out["reversion_score_decayed"].astype(float)
    of_decayed = out["orderflow_score_decayed"].astype(float)
    struct_decayed = out["structure_score_decayed"].astype(float)
    pattern_decayed = out["pattern_score_decayed"].astype(float)

    confirm_decayed = out["meta_trend_plus_orderflow_confirmation_decayed"].astype(float)
    horizon_consistency = out["meta_horizon_consistency_score_decayed"].astype(float)
    meta_regime_strength = out["meta_regime_adjusted_signal_strength_decayed"].astype(float)
    meta_agreement = out["meta_agreement_score_decayed"].astype(float)
    meta_conflict = out["meta_conflict_score_decayed"].astype(float)

    pred1 = out["pred_1m_up"].astype(float)
    pred5 = out["pred_5m_up"].astype(float)
    pred15 = out["pred_15m_up"].astype(float)
    pred_break = out["pred_breakout"].astype(float)
    pred_rev = out["pred_reversion"].astype(float)

    # Robust z-normalization
    z = {
        "trend": _robust_z(trend_decayed, config.z_clip),
        "momentum": _robust_z(momentum_decayed, config.z_clip),
        "volatility": _robust_z(vol_decayed, config.z_clip),
        "reversion": _robust_z(rev_decayed, config.z_clip),
        "orderflow": _robust_z(of_decayed, config.z_clip),
        "structure": _robust_z(struct_decayed, config.z_clip),
        "pattern": _robust_z(pattern_decayed, config.z_clip),
        "confirmation": _robust_z(confirm_decayed, config.z_clip),
        "meta_regime": _robust_z(meta_regime_strength, config.z_clip),
        "meta_agreement": _robust_z(meta_agreement - meta_conflict, config.z_clip),
        "horizon_consistency": _robust_z(horizon_consistency, config.z_clip),
        "pred1": _robust_z(pred1 * 2.0 - 1.0, config.z_clip),
        "pred5": _robust_z(pred5 * 2.0 - 1.0, config.z_clip),
        "pred15": _robust_z(pred15 * 2.0 - 1.0, config.z_clip),
        "pred_break": _robust_z(pred_break * 2.0 - 1.0, config.z_clip),
        "pred_rev": _robust_z(pred_rev * 2.0 - 1.0, config.z_clip),
    }

    regime_state = out["regime_state"].round().clip(0, 2).astype(int)
    regime_conf = out["regime_confidence"].astype(float).clip(0.0, 1.0)

    # Adaptive multi-horizon confirmation (regime-weighted)
    d1 = pred1 - 0.5
    d5 = pred5 - 0.5
    d15 = pred15 - 0.5

    w1 = np.full(len(out), 1 / 3)
    w5 = np.full(len(out), 1 / 3)
    w15 = np.full(len(out), 1 / 3)

    # Trend regime weights
    w1 = np.where(regime_state == 1, 0.30, w1)
    w5 = np.where(regime_state == 1, 0.35, w5)
    w15 = np.where(regime_state == 1, 0.35, w15)

    # Range regime weights
    w1 = np.where(regime_state == 0, 0.70, w1)
    w5 = np.where(regime_state == 0, 0.20, w5)
    w15 = np.where(regime_state == 0, 0.10, w15)

    # Breakout regime weights
    w1 = np.where(regime_state == 2, 0.45, w1)
    w5 = np.where(regime_state == 2, 0.40, w5)
    w15 = np.where(regime_state == 2, 0.15, w15)

    direction_score = d1 * w1 + d5 * w5 + d15 * w15

    # Regime-dependent multipliers
    trend_mult = 1.0 + config.regime_trend_boost * regime_conf * (regime_state == 1)
    range_mult = 1.0 + config.regime_range_boost * regime_conf * (regime_state == 0)
    breakout_mult = 1.0 + config.regime_breakout_boost * regime_conf * (regime_state == 2)

    # Bundles -----------------------------------------------------------
    trend_bundle_raw = (
        config.trend_weight_trend * z["trend"] * trend_mult +
        config.trend_weight_momentum * z["momentum"] * trend_mult +
        config.trend_weight_confirmation * z["confirmation"] * trend_mult +
        config.trend_weight_pred15 * z["pred15"] * trend_mult
    )
    trend_bundle = _tanh_clip(_robust_z(trend_bundle_raw, config.z_clip), config.tanh_clip)

    reversion_bundle_raw = (
        config.reversion_weight_reversion * z["reversion"] * range_mult +
        config.reversion_weight_structure * z["structure"] * range_mult +
        config.reversion_weight_pattern * z["pattern"] * range_mult +
        config.reversion_weight_pred * z["pred_rev"] * range_mult
    )
    reversion_bundle = _tanh_clip(_robust_z(reversion_bundle_raw, config.z_clip), config.tanh_clip)

    breakout_bundle_raw = (
        config.breakout_weight_volatility * z["volatility"] * breakout_mult +
        config.breakout_weight_orderflow * z["orderflow"] * breakout_mult +
        config.breakout_weight_pred * z["pred_break"] * breakout_mult
    )
    breakout_bundle = _tanh_clip(_robust_z(breakout_bundle_raw, config.z_clip), config.tanh_clip)

    horizon_weight = (horizon_consistency.clip(-1.0, 1.0) + 1.0) * 0.5
    horizon_bundle_raw = (
        config.horizon_weight_pred1 * z["pred1"] * horizon_weight +
        config.horizon_weight_pred5 * z["pred5"] * (1.0 - horizon_weight)
    )
    horizon_bundle = _tanh_clip(_robust_z(horizon_bundle_raw, config.z_clip), config.tanh_clip)

    meta_bundle_raw = (
        config.meta_weight_agreement * z["meta_agreement"] +
        config.meta_weight_regime_strength * z["meta_regime"]
    )
    meta_bundle = _tanh_clip(_robust_z(meta_bundle_raw, config.z_clip), config.tanh_clip)

    # Conflict/agreement gating
    conflict_dampen = np.clip(
        1.0 - np.maximum(0.0, meta_conflict - config.minimum_conflict_threshold),
        config.conflict_floor,
        1.0,
    )
    agreement_boost = 1.0 + np.clip(meta_agreement - config.minimum_agreement_threshold, 0.0, config.agreement_ceiling)
    gating = conflict_dampen * agreement_boost

    # Final fusion
    fused = (
        config.bundle_trend * trend_bundle +
        config.bundle_reversion * reversion_bundle +
        config.bundle_breakout * breakout_bundle +
        config.bundle_horizon * horizon_bundle +
        config.bundle_meta * meta_bundle
    )

    fused = fused * gating
    fused = fused + direction_score * 2.0

    # Upgrade #4 — Conflict Penalty Layer
    if config.conflict_penalty_strength > 0:
        penalty = np.zeros(len(out))

        # Horizon disagreement
        p1 = pred1
        p5 = pred5
        p15 = pred15

        disagree_1_5 = np.abs(p1 - p5)
        disagree_5_15 = np.abs(p5 - p15)
        disagree_1_15 = np.abs(p1 - p15)

        horizon_penalty = (
            disagree_1_5 + disagree_5_15 + disagree_1_15
        ) * config.horizon_disagreement_weight
        penalty += horizon_penalty

        # Engine conflict
        trend = out["trend_score"].astype(float)
        revert = out["reversion_score"].astype(float)
        momentum = out["momentum_score"].astype(float)
        structure = out["structure_score"].astype(float)
        orderflow = out["orderflow_score"].astype(float)

        engine_disagree = (
            np.abs(trend + revert) +
            np.abs(momentum - structure) +
            np.abs(orderflow - trend)
        ) * config.engine_conflict_weight
        penalty += engine_disagree

        # Pattern structure conflict
        zero_series = pd.Series(0.0, index=out.index)
        bullish_pattern = out.get("engulfing_bull", zero_series).astype(float) + out.get("pin_bar_bull", zero_series).astype(float)
        bearish_pattern = out.get("engulfing_bear", zero_series).astype(float) + out.get("pin_bar_bear", zero_series).astype(float)

        pattern_conflict = (
            np.maximum(0, bullish_pattern * np.sign(-momentum)) +
            np.maximum(0, bearish_pattern * np.sign(momentum))
        ) * config.pattern_conflict_weight
        penalty += pattern_conflict

        shrink_factor = 1 - (config.conflict_penalty_strength * penalty)
        shrink_factor = np.clip(shrink_factor, 0.2, 1.0)

        fused = fused * shrink_factor

    fused = _tanh_clip(fused, config.tanh_clip)

    smoothed = pd.Series(fused, index=out.index).ewm(alpha=config.smoothing_alpha, adjust=False).mean()
    out["raw_entry_confidence"] = smoothed.clip(-1.0, 1.0)
    out["raw_entry_confidence"] = out["raw_entry_confidence"].ffill().bfill()
    return out
