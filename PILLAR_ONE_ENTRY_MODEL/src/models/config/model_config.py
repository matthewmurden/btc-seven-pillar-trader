from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass(frozen=True)
class ModelConfig:
    """Shared LightGBM defaults to keep models deterministic and efficient."""

    params: Dict[str, Any] = field(
        default_factory=lambda: {
            "boosting_type": "gbdt",
            "n_estimators": 120,
            "learning_rate": 0.05,
            "num_leaves": 64,
            "max_depth": -1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "min_child_samples": 20,
            "random_state": 42,
            "objective": "binary",
            "n_jobs": -1,
        }
    )


DEFAULT_MODEL_CONFIG = ModelConfig()
