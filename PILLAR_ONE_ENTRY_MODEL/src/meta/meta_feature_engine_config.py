# Meta-Feature Engine Configuration

from dataclasses import dataclass


@dataclass
class MetaFeatureConfig:
    # --- agreement weighting (regime-aware) ---
    agreement_trend_weight: float = 1.0
    agreement_momentum_weight: float = 0.9
    agreement_reversion_weight: float = 0.9
    agreement_structure_weight: float = 0.75
    agreement_orderflow_weight: float = 1.0
    agreement_volatility_weight: float = 0.85
    agreement_pattern_weight: float = 0.65
    agreement_clip: float = 3.0

    # --- conflict weighting ---
    conflict_engine_weights: tuple = (1.0, 1.0, 0.9, 0.9, 1.0, 0.8, 0.7)
    conflict_dispersion_weight: float = 0.55
    conflict_mean_weight: float = 0.45
    regime_uncertainty_weight: float = 0.25

    # --- fused feature tuning ---
    vol_momentum_clip: float = 2.5
    structure_momentum_clip: float = 3.0
    confirmation_scale: float = 1.3
    confirmation_clip: float = 3.0
    horizon_scale: float = 1.4
    horizon_variance_penalty: float = 3.0

    # --- smoothing ---
    base_smoothing_alpha: float = 0.12
    fast_smoothing_alpha: float = 0.18
    horizon_smoothing_alpha: float = 0.10

    # --- regime boosts ---
    regime_boost_trend: float = 1.1
    regime_boost_range: float = 0.95
    regime_boost_breakout: float = 1.05

    # --- numerical stability ---
    epsilon: float = 1e-6


DEFAULT_META_FEATURE_CONFIG = MetaFeatureConfig()
