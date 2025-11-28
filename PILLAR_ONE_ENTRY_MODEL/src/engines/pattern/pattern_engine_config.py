from dataclasses import dataclass


@dataclass(frozen=True)
class PatternConfig:
    base_weight: float = 0.25
    pin_weight: float = 0.20
    momentum_bar_weight: float = 0.20
    engulfing_weight: float = 0.15
    imbalance_bar_weight: float = 0.10
    vol_bias_weight: float = 0.10
    smoothing_alpha: float = 0.10
    clip_value: float = 4.0


DEFAULT_PATTERN_CONFIG = PatternConfig()
