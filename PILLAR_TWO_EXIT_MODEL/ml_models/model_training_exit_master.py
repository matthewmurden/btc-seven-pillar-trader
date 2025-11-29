"""Master trainer for exit regression/classification models (Part 4)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor, LGBMClassifier
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
TRAINING_PATH = BASE_DIR / "data" / "processed" / "exit_training.parquet"

REG_MODELS = {
    "Model_MAE": {"label": "MAE_pct"},
    "Model_MFE": {"label": "MFE_pct"},
    "Model_ExitTime": {"label": "exit_bar"},
    "Model_OptimalTP": {"label": "first_touch_tp_distance"},
    "Model_OptimalSL": {"label": "first_touch_sl_distance"},
}

CLS_MODELS = {
    "Model_TP_First": {"label": "tp_first"},
    "Model_SL_First": {"label": "sl_first"},
}


def _get_features(df: pd.DataFrame, label_cols) -> pd.DataFrame:
    return df[[c for c in df.columns if is_numeric_dtype(df[c]) and c not in label_cols]]


def _time_split(df: pd.DataFrame, test_size: float = 0.2):
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    return train_df, val_df


def _train_regressor(X_train, y_train):
    return LGBMRegressor(
        boosting_type="gbdt",
        learning_rate=0.05,
        n_estimators=200,
        num_leaves=63,
        max_depth=-1,
    ).fit(X_train, y_train)


def _train_classifier(X_train, y_train):
    return LGBMClassifier(
        boosting_type="gbdt",
        learning_rate=0.05,
        n_estimators=200,
        num_leaves=63,
        max_depth=-1,
        objective="binary",
    ).fit(X_train, y_train)


def train_all_exit_models() -> None:
    if not TRAINING_PATH.exists():
        raise FileNotFoundError(f"Training dataset missing at {TRAINING_PATH}")
    df = pd.read_parquet(TRAINING_PATH)

    label_cols = set(v["label"] for v in {**REG_MODELS, **CLS_MODELS}.values())
    feature_df = _get_features(df, label_cols)

    train_df, _ = _time_split(df)
    X_train = feature_df.loc[train_df.index]

    # Regression models
    for name, cfg in REG_MODELS.items():
        y = train_df[cfg["label"]]
        model = _train_regressor(X_train, y)
        model_dir = BASE_DIR / "ml_models" / name
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "features": list(X_train.columns)}, model_dir / "model.pkl")
        print(f"Trained {name}")

    # Classification models
    for name, cfg in CLS_MODELS.items():
        y = train_df[cfg["label"]]
        model = _train_classifier(X_train, y)
        model_dir = BASE_DIR / "ml_models" / name
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "features": list(X_train.columns)}, model_dir / "model.pkl")
        print(f"Trained {name}")


if __name__ == "__main__":
    train_all_exit_models()
