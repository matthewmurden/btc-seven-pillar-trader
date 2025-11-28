from dataclasses import dataclass


@dataclass(frozen=True)
class VolatilityConfig:
    atr_weight: float = 0.25
    atrx_weight: float = 0.15
    realized_weight: float = 0.25
    percentile_weight: float = 0.15
    dynamics_weight: float = 0.20
    smoothing_alpha: float = 0.12
    clip_value: float = 4.0


DEFAULT_VOLATILITY_CONFIG = VolatilityConfig()
