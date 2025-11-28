# Calibration layer configuration

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CalibrationConfig:
    method: str = "platt"  # "platt" or "isotonic"
    clip_min: float = 0.0005
    clip_max: float = 0.9995
    model_path: str = "calibration_layer.pkl"

    def resolve_path(self) -> Path:
        return Path(self.model_path)


DEFAULT_CALIBRATION_CONFIG = CalibrationConfig()
