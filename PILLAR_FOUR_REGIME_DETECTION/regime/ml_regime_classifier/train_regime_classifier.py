# ======================================================
# train_regime_classifier.py (Pillar 4, Part 3)
# ======================================================

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, log_loss, confusion_matrix
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from .regime_dataset_builder import build_regime_dataset
from .model_utils import save_model, save_feature_list, save_metadata

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_PATH = PROJECT_ROOT / "PILLAR_FOUR_REGIME_DETECTION" / "data" / "processed"


def train_and_save():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df, features, labels = build_regime_dataset(include_labels=True)

    X = df[features].astype(float)
    y = labels
    num_classes = int(y.max()) + 1

    # ======================================================
    # LightGBM
    # ======================================================
    lgb = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=96,
        objective="multiclass",
        num_class=num_classes
    )
    lgb.fit(X, y)
    save_model(lgb, "lgbm_regime.pkl")

    pred_l = lgb.predict(X)
    prob_l = lgb.predict_proba(X)

    logging.info("LightGBM:")
    logging.info("Accuracy %.5f", accuracy_score(y, pred_l))
    logging.info("LogLoss %.5f", log_loss(y, prob_l))
    logging.info("Confusion: %s", confusion_matrix(y, pred_l).tolist())
    logging.info("Report: %s", classification_report(y, pred_l))

    # ======================================================
    # XGBoost
    # ======================================================
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        tree_method="hist",
        learning_rate=0.03,
        n_estimators=400,
        max_depth=8
    )
    xgb.fit(X, y)
    save_model(xgb, "xgboost_regime.pkl")

    pred_x = xgb.predict(X)
    prob_x = xgb.predict_proba(X)

    logging.info("XGBoost:")
    logging.info("Accuracy %.5f", accuracy_score(y, pred_x))
    logging.info("LogLoss %.5f", log_loss(y, prob_x))
    logging.info("Confusion: %s", confusion_matrix(y, pred_x).tolist())

    # ======================================================
    # Metadata
    # ======================================================
    save_feature_list(features)
    save_metadata({
        "num_classes": num_classes,
        "feature_count": len(features)
    })

    logging.info("Saved both models + metadata.")


if __name__ == "__main__":
    train_and_save()