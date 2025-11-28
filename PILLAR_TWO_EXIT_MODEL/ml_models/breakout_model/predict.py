"""Prediction script for breakout model."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"


def _load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file missing at {MODEL_PATH}. Run train.py first.")
    return joblib.load(MODEL_PATH)


def _prepare_features(df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df.columns]
    for col in missing:
        df[col] = 0
    return df.reindex(columns=feature_cols).fillna(0)


def predict(df: pd.DataFrame) -> np.ndarray:
    bundle = _load_bundle()
    model = bundle["model"]
    feature_cols = bundle["features"]
    X = _prepare_features(df.copy(), feature_cols)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    raise AttributeError("Loaded model lacks predict_proba/decision_function.")


if __name__ == "__main__":
    print("Use this module via ModelPredictionMaster or import predict().")
