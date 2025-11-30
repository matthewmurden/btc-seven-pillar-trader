from __future__ import annotations

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    from sklearn.ensemble import GradientBoostingRegressor as LGBMRegressor

from ml_models.base_model import BaseExitModel


class ExitTimeModel(BaseExitModel):
    """Regression model predicting exit_bar (time to exit)."""

    def __init__(self, feature_names: list[str] | None = None, random_state: int = 42):
        model = LGBMRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
        )
        super().__init__(model, feature_names, random_state)
