# ======================================================
# apply_regime_classifier.py
# ======================================================

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from .model_utils import load_model, load_feature_list
from .regime_dataset_builder import build_regime_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_FILE = PROJECT_ROOT / "PILLAR_FOUR_REGIME_DETECTION/data/processed/regime_ml_predictions.parquet"


def apply_models():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df, features, _ = build_regime_dataset(include_labels=False)
    X = df[features].astype(float)

    # Load models + feature list
    saved_features = load_feature_list()
    if saved_features != features:
        raise ValueError("Feature mismatch between training & inference!")

    lgb = load_model("lgbm_regime.pkl")
    xgb = load_model("xgboost_regime.pkl")

    # Predict
    prob_l = lgb.predict_proba(X)
    prob_x = xgb.predict_proba(X)

    # Average ensemble
    probs = (prob_l + prob_x) / 2
    labels = probs.argmax(axis=1)
    confidence = probs.max(axis=1)

    out = df[["timestamp"]].copy()
    out["regime_pred"] = labels
    out["regime_confidence"] = confidence
    for i in range(probs.shape[1]):
        out[f"prob_{i}"] = probs[:, i]

    out.to_parquet(OUT_FILE, index=False)
    logging.info("Saved ML regime predictions: %s", OUT_FILE)