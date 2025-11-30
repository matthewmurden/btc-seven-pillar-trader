import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, classification_report


def train_lgbm(X, y):
    model = lgb.LGBMClassifier(
        n_estimators=300,
        num_leaves=64,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multiclass",
        num_class=8,
        learning_rate=0.05,
        min_child_samples=30,
    )
    model.fit(X, y)
    return model


def train_xgb(X, y):
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=8,
        eval_metric="mlogloss",
    )
    model.fit(X, y)
    return model


def eval_model(model, X, y):
    probs = model.predict_proba(X)
    preds = np.argmax(probs, axis=1)

    return {
        "logloss": log_loss(y, probs),
        "accuracy": accuracy_score(y, preds),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
        "classification_report": classification_report(y, preds, output_dict=True),
    }