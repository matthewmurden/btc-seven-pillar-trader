from dataclasses import dataclass


@dataclass(frozen=True)
class StructureConfig:
    bos_weight: float = 0.30
    choch_weight: float = 0.25
    swing_weight: float = 0.20
    structure_signal_weight: float = 0.15
    sweep_weight: float = 0.10
    smoothing_alpha: float = 0.10
    clip_value: float = 4.0


DEFAULT_STRUCTURE_CONFIG = StructureConfig()
