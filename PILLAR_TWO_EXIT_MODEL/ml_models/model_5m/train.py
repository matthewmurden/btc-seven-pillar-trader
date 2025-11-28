"""Training script for 5m horizon model."""

from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
import pandas as pd
from pandas.api.types import is_numeric_dtype


def _get_model():
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            boosting_type="gbdt",
            objective="binary",
            num_leaves=63,
            max_depth=-1,
            learning_rate=0.03,
            n_estimators=300,
        )
    except ImportError:
        try:
            from xgboost import XGBClassifier

            return XGBClassifier(
                objective="binary:logistic",
                max_depth=6,
                learning_rate=0.05,
                n_estimators=300,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier

            return GradientBoostingClassifier()


BASE_DIR = Path(__file__).resolve().parents[2]
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features_part3_with_labels.parquet"
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
LABEL_COL = "forward_returns_5m"
LABEL_COLUMNS = {
    "forward_returns_1m",
    "forward_returns_5m",
    "forward_returns_15m",
    "breakout_label",
    "reversion_label",
}


def _load_training_data() -> tuple[pd.DataFrame, pd.Series, List[str]]:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Features parquet not found at {FEATURES_PATH}")

    df = pd.read_parquet(FEATURES_PATH)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' missing from features file.")

    y_raw = df[LABEL_COL]
    y = (y_raw > 0).astype(int)

    feature_cols = [
        c
        for c in df.columns
        if is_numeric_dtype(df[c]) and c not in LABEL_COLUMNS
    ]
    X = df[feature_cols].fillna(0)
    return X, y, feature_cols


def train() -> Path:
    X, y, feature_cols = _load_training_data()
    model = _get_model()
    model.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feature_cols}, MODEL_PATH)
    print(f"Trained 5m model; saved to {MODEL_PATH}")
    return MODEL_PATH


if __name__ == "__main__":
    train()
