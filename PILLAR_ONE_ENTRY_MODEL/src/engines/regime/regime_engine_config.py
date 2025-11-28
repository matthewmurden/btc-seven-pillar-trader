from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeConfig:
    vol_threshold_trend: float = 0.35
    vol_threshold_breakout: float = 0.65
    trend_strength_threshold: float = 0.55  # trend_strength scaled /100
    bb_width_threshold: float = 0.6
    atr_expansion_threshold: float = 0.02

    vol_weight: float = 0.3
    trend_weight: float = 0.4
    bb_weight: float = 0.2
    atrx_weight: float = 0.1
    smoothing_alpha: float = 0.15
    clip_value: float = 4.0


DEFAULT_REGIME_CONFIG = RegimeConfig()
