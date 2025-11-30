"""
Base model for exit tasks providing common fit/predict/save/load utilities.
"""

from __future__ import annotations

import joblib
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseExitModel:
    """Shared utilities for exit models with deterministic behavior."""

    model: Any
    feature_names: Optional[list[str]]
    random_state: int

    def __init__(self, model: Any, feature_names: Optional[list[str]] = None, random_state: int = 42):
        self.model = model
        self.feature_names = feature_names
        self.random_state = random_state

    def validate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if self.feature_names:
            missing = [c for c in self.feature_names if c not in X.columns]
            if missing:
                raise ValueError(f"Missing features: {missing}")
            X = X[self.feature_names]
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return X

    def fit(self, X: pd.DataFrame, y: Any) -> "BaseExitModel":
        Xv = self.validate_features(X)
        self.model.fit(Xv, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xv = self.validate_features(X)
        preds = self.model.predict(Xv)
        return np.asarray(preds, dtype=np.float32)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("Underlying model does not support predict_proba.")
        Xv = self.validate_features(X)
        probs = self.model.predict_proba(Xv)
        return np.asarray(probs, dtype=np.float32)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "feature_names": self.feature_names, "random_state": self.random_state},
            path,
        )
        logger.info("Saved model to %s", path)

    @classmethod
    def load(cls, path: Path) -> "BaseExitModel":
        data = joblib.load(path)
        obj = cls(model=data["model"], feature_names=data.get("feature_names"), random_state=data.get("random_state", 42))
        return obj
