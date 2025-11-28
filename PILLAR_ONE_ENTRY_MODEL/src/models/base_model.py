import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.models.config.model_config import DEFAULT_MODEL_CONFIG, ModelConfig


TARGET_COLUMNS = {
    "next_1m_up",
    "next_5m_up",
    "next_15m_trend",
    "breakout_target",
    "reversion_target",
}


class BaseModel:
    """
    Deterministic LightGBM wrapper with unified train/predict/save/load.
    Expects upstream code to have already run ffill/bfill.
    """

    def __init__(self, target_name: str, model_config: ModelConfig = DEFAULT_MODEL_CONFIG):
        self.target_name = target_name
        self.model_config = model_config
        self.model = LGBMClassifier(**self.model_config.params)
        self.feature_names = None

    @staticmethod
    def _select_features(df: pd.DataFrame) -> pd.DataFrame:
        # Drop known targets and non-feature columns
        exclude = TARGET_COLUMNS | {"timestamp"}
        feature_cols = [c for c in df.columns if c not in exclude]
        return df[feature_cols]

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = self._select_features(df)
        # Convert to numeric only; non-numeric columns are coerced to NaN then filled
        feats = feats.apply(pd.to_numeric, errors="coerce")
        feats = feats.fillna(method="ffill").fillna(method="bfill")
        return feats

    def train(self, df: pd.DataFrame) -> None:
        assert self.target_name in df.columns, f"Missing target {self.target_name}"
        X = self._prepare_features(df)
        y = df[self.target_name].fillna(0).astype(int)
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._prepare_features(df)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "target_name": self.target_name, "feature_names": self.feature_names}, path)

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        payload = joblib.load(path)
        obj = cls.__new__(cls)
        BaseModel.__init__(obj, target_name=payload["target_name"])
        obj.model = payload["model"]
        obj.feature_names = payload.get("feature_names")
        return obj
