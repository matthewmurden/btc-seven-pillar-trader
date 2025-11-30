from __future__ import annotations

import numpy as np
import pandas as pd

from ml_models.base_model import BaseExitModel


class SurvivalExitModel(BaseExitModel):
    """
    Simple survival-style model using exponential baseline hazard approximation.

    Stores mean hazard rate from training as 1 / mean survival time.
    """

    def __init__(self, feature_names: list[str] | None = None, random_state: int = 42):
        # Placeholder model object; fit/predict implemented directly.
        super().__init__(model=self, feature_names=feature_names, random_state=random_state)
        self._lambda_: float | None = None

    def fit(self, X: pd.DataFrame, durations: pd.Series, events: pd.Series) -> "SurvivalExitModel":
        durations = pd.to_numeric(durations, errors="coerce")
        events = pd.to_numeric(events, errors="coerce").fillna(0)
        observed = durations[events == 1]
        mean_time = observed.mean() if len(observed) else durations.mean()
        self._lambda_ = 1.0 / max(mean_time, 1e-6)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._lambda_ is None:
            raise ValueError("Model not fitted.")
        Xv = self.validate_features(X)
        risk = np.full(len(Xv), self._lambda_, dtype=np.float32)
        return risk

    def save(self, path):
        import joblib

        joblib.dump({"lambda": self._lambda_, "feature_names": self.feature_names}, path)

    @classmethod
    def load(cls, path):
        import joblib

        data = joblib.load(path)
        obj = cls(feature_names=data.get("feature_names"))
        obj._lambda_ = data.get("lambda")
        return obj
