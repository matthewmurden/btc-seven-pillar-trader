# Master Aggregator configuration

from dataclasses import dataclass


@dataclass
class MasterAggregatorConfig:
    # normalization and clipping
    z_clip: float = 4.0
    tanh_clip: float = 3.0

    # bundle component weights
    trend_weight_trend: float = 0.35
    trend_weight_momentum: float = 0.30
    trend_weight_confirmation: float = 0.20
    trend_weight_pred15: float = 0.15

    reversion_weight_reversion: float = 0.35
    reversion_weight_structure: float = 0.25
    reversion_weight_pattern: float = 0.20
    reversion_weight_pred: float = 0.20

    breakout_weight_volatility: float = 0.35
    breakout_weight_orderflow: float = 0.35
    breakout_weight_pred: float = 0.30

    horizon_weight_pred1: float = 0.55
    horizon_weight_pred5: float = 0.45

    meta_weight_agreement: float = 0.55
    meta_weight_regime_strength: float = 0.45

    # top-level bundle weights
    bundle_trend: float = 0.28
    bundle_reversion: float = 0.22
    bundle_breakout: float = 0.26
    bundle_horizon: float = 0.12
    bundle_meta: float = 0.12

    # regime boosts
    regime_trend_boost: float = 0.35
    regime_range_boost: float = 0.25
    regime_breakout_boost: float = 0.35

    # gating thresholds
    minimum_conflict_threshold: float = 0.40
    minimum_agreement_threshold: float = 0.20

    # smoothing
    smoothing_alpha: float = 0.12

    # fallback behavior
    conflict_floor: float = 0.15
    agreement_ceiling: float = 0.9

    # --- NEW: Conflict Penalty Layer ---
    conflict_penalty_strength: float = 0.15   # global strength
    horizon_disagreement_weight: float = 0.5
    engine_conflict_weight: float = 0.5
    pattern_conflict_weight: float = 0.5

DEFAULT_CONFLICT_PENALTY = {
    "enabled": True
}


DEFAULT_MASTER_AGGREGATOR_CONFIG = MasterAggregatorConfig()
