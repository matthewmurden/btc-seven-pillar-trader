import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .calibration_config import CalibrationConfig, DEFAULT_CALIBRATION_CONFIG


def _robust_z(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    med = s.median()
    iqr = s.quantile(0.75) - s.quantile(0.25)
    scale = iqr * 0.7413
    if not np.isfinite(scale) or scale < 1e-9:
        scale = s.std()
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    z = (s - med) / scale
    return z.fillna(0.0)


class CalibrationLayer:
    def __init__(self, method: str = "platt"):
        self.method = method.lower()
        self.model: Optional[object] = None

    def _build_model(self):
        if self.method == "platt":
            return LogisticRegression(C=1e6, class_weight="balanced", solver="lbfgs", max_iter=200, random_state=42)
        if self.method == "isotonic":
            return IsotonicRegression(out_of_bounds="clip")
        raise ValueError(f"Unsupported calibration method: {self.method}")

    def fit(self, df: pd.DataFrame, config: CalibrationConfig = DEFAULT_CALIBRATION_CONFIG) -> None:
        if "raw_entry_confidence" not in df.columns:
            raise ValueError("CalibrationLayer.fit requires 'raw_entry_confidence' column")
        if "target" not in df.columns:
            raise ValueError("CalibrationLayer.fit requires 'target' column for calibration labels")

        logits = _robust_z(df["raw_entry_confidence"])
        y = df["target"].astype(int).to_numpy()

        model = self._build_model()
        if self.method == "platt":
            model.fit(logits.values.reshape(-1, 1), y)
        else:  # isotonic
            model.fit(logits.values, y)

        self.model = model
        self.method = config.method.lower()

    def predict(self, df: pd.DataFrame, config: CalibrationConfig = DEFAULT_CALIBRATION_CONFIG) -> pd.Series:
        if self.model is None:
            raise ValueError("Calibration model is not fitted. Call fit or load first.")
        if "raw_entry_confidence" not in df.columns:
            raise ValueError("CalibrationLayer.predict requires 'raw_entry_confidence' column")

        logits = _robust_z(df["raw_entry_confidence"])
        if self.method == "platt":
            probs = self.model.predict_proba(logits.values.reshape(-1, 1))[:, 1]
        else:
            probs = self.model.predict(logits.values)

        probs = np.clip(probs, config.clip_min, config.clip_max)
        return pd.Series(probs, index=df.index, name="entry_probability")

    def save(self, path: str) -> None:
        payload = {"method": self.method, "model": self.model}
        with Path(path).open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "CalibrationLayer":
        payload = pickle.load(Path(path).open("rb"))
        obj = cls(method=payload["method"])
        obj.model = payload["model"]
        return obj


def fit_calibrator(df: pd.DataFrame, config: CalibrationConfig = DEFAULT_CALIBRATION_CONFIG) -> CalibrationLayer:
    layer = CalibrationLayer(method=config.method)
    layer.fit(df, config)
    layer.save(config.resolve_path())
    return layer


def apply_calibrator(df: pd.DataFrame, config: CalibrationConfig = DEFAULT_CALIBRATION_CONFIG) -> pd.DataFrame:
    path = config.resolve_path()
    if not path.exists():
        raise FileNotFoundError(f"Calibration model not found at {path}")
    layer = CalibrationLayer.load(path)
    probs = layer.predict(df, config)
    out = df.copy()
    out["entry_probability"] = probs
    return out
