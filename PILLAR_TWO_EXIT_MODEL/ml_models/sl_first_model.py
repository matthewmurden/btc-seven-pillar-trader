from __future__ import annotations

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    from sklearn.ensemble import GradientBoostingClassifier as LGBMClassifier

from ml_models.base_model import BaseExitModel


class SLFirstModel(BaseExitModel):
    """Binary classifier for SL-first touch."""

    def __init__(self, feature_names: list[str] | None = None, random_state: int = 42):
        model = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
        )
        super().__init__(model, feature_names, random_state)
