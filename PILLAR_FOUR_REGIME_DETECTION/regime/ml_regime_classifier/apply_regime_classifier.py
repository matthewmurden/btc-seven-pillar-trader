"""
Apply trained regime classifier models to new data.
"""

from __future__ import annotations
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from .regime_dataset_builder import build_regime_dataset

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def apply_regime_classifier():
    df, features, _ = build_regime_dataset(include_labels=False)

    X = df[features].to_numpy(dtype=np.float32)

    with open(MODEL_DIR / "lgbm_regime_model.pkl", "rb") as f:
        lgbm = pickle.load(f)
    with open(MODEL_DIR / "xgboost_regime_model.pkl", "rb") as f:
        xgb = pickle.load(f)
    with open(MODEL_DIR / "feature_list.pkl", "rb") as f:
        saved_features = pickle.load(f)

    if features != saved_features:
        raise ValueError("Feature mismatch between training and inference!")

    p_lgbm = lgbm.predict_proba(X)
    p_xgb = xgb.predict_proba(X)

    final_probs = (p_lgbm + p_xgb) / 2
    preds = np.argmax(final_probs, axis=1)

    out = pd.DataFrame({
        "timestamp": df["timestamp"],
        "regime_pred": preds,
        "confidence": final_probs.max(axis=1),
    })

    # Save
    out_path = MODEL_DIR.parents[2] / "data/processed/regime_ml_predictions.parquet"
    out.to_parquet(out_path, index=False)

    return out