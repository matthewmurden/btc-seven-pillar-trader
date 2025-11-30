"""
Train LightGBM + XGBoost regime classifiers.
Ensures consistent feature ordering and strict reproducibility.
"""

from __future__ import annotations
import logging
import pickle
from pathlib import Path
import numpy as np

from .regime_dataset_builder import build_regime_dataset
from .model_utils import train_lgbm, train_xgb, eval_model

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_and_save():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df, features, labels = build_regime_dataset(include_labels=True)

    X = df[features].to_numpy(dtype=np.float32)
    y = labels.astype(np.int8)

    # Train LGBM
    lgbm = train_lgbm(X, y)
    lgbm_metrics = eval_model(lgbm, X, y)

    # Train XGB
    xgb = train_xgb(X, y)
    xgb_metrics = eval_model(xgb, X, y)

    # Save models
    with open(MODEL_DIR / "lgbm_regime_model.pkl", "wb") as f:
        pickle.dump(lgbm, f)
    with open(MODEL_DIR / "xgboost_regime_model.pkl", "wb") as f:
        pickle.dump(xgb, f)
    with open(MODEL_DIR / "feature_list.pkl", "wb") as f:
        pickle.dump(features, f)

    # Save config
    with open(MODEL_DIR / "model_config.json", "w") as f:
        import json
        json.dump({
            "model": "regime_classifier",
            "features_sha256": "placeholder",
            "feature_count": len(features),
            "lgbm_metrics": lgbm_metrics,
            "xgb_metrics": xgb_metrics
        }, f, indent=4)

    logging.info("Saved regime classifier models + config.")