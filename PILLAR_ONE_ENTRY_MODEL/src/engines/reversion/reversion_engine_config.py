from dataclasses import dataclass


@dataclass(frozen=True)
class ReversionConfig:
    rsi_weight: float = 0.25
    overshoot_weight: float = 0.20
    zscore_weight: float = 0.20
    signal_weight: float = 0.20
    accel_weight: float = 0.10
    smoothing_alpha: float = 0.12
    clip_value: float = 4.0
    vol_floor: float = 1e-6


DEFAULT_REVERSION_CONFIG = ReversionConfig()
