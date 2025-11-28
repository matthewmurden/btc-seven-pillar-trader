from dataclasses import dataclass


@dataclass(frozen=True)
class OrderflowConfig:
    thrust_weight: float = 0.30
    imbalance_weight: float = 0.25
    sweep_weight: float = 0.15
    absorption_weight: float = 0.10
    cum_imbalance_weight: float = 0.10
    vol_weight: float = 0.10
    smoothing_alpha: float = 0.10
    clip_value: float = 4.0
    vol_floor: float = 1e-6


DEFAULT_ORDERFLOW_CONFIG = OrderflowConfig()
