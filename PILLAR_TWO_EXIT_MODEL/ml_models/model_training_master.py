"""Master trainer to fit Pillar Two exit models with time-based split."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from pandas.api.types import is_numeric_dtype
from lightgbm import LGBMRegressor, LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATA_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_CANDIDATES = [
    DATA_DIR / "exit_training_aligned.parquet",
    DATA_DIR / "features_with_labels_and_pillar1.parquet",
    DATA_DIR / "exit_training.parquet",
]
TRAIN_PATH = DATA_DIR / "exit_train.parquet"
TEST_PATH = DATA_DIR / "exit_test.parquet"


REG_MODELS: Dict[str, str] = {
    "Model_MAE": "MAE_pct",
    "Model_MFE": "MFE_pct",
    "Model_ExitTime": "exit_bar",
    "Model_OptimalTP": "first_touch_tp_distance",
    "Model_OptimalSL": "first_touch_sl_distance",
}

CLS_MODELS: Dict[str, str] = {
    "Model_TP_First": "tp_first",
    "Model_SL_First": "sl_first",
}

LABEL_COLS = set(REG_MODELS.values()) | set(CLS_MODELS.values())


def _load_features() -> pd.DataFrame:
    for path in FEATURES_CANDIDATES:
        if path.exists():
            df = pd.read_parquet(path)
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp")
            return df.reset_index(drop=True)
    raise FileNotFoundError("No features dataset found for training.")


def _split_train_test(df: pd.DataFrame, train_frac: float = 0.8):
    split_idx = int(len(df) * train_frac)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(TRAIN_PATH)
    test_df.to_parquet(TEST_PATH)
    return train_df, test_df


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if is_numeric_dtype(df[c]) and c not in LABEL_COLS]


def _train_reg(X: pd.DataFrame, y: pd.Series):
    model = LGBMRegressor(
        boosting_type="gbdt",
        learning_rate=0.05,
        n_estimators=200,
        num_leaves=63,
        max_depth=-1,
    )
    model.fit(X, y)
    return model


def _train_cls(X: pd.DataFrame, y: pd.Series):
    model = LGBMClassifier(
        boosting_type="gbdt",
        learning_rate=0.05,
        n_estimators=200,
        num_leaves=63,
        max_depth=-1,
        objective="binary",
    )
    model.fit(X, y)
    return model


class ModelTrainingMaster:
    """Runs training with time-based split and saves train/test sets."""

    def train_all(self) -> None:
        df = _load_features()
        if df.isna().any().any():
            raise ValueError("Training data contains NaNs")
        if len(df) < 5000:
            print("Warning: training data has fewer than 5000 samples; results may be unstable.")
        train_df, _ = _split_train_test(df)
        feat_cols = _get_feature_columns(train_df)
        X_train = train_df[feat_cols].fillna(0)

        # Regression models
        for name, label_col in REG_MODELS.items():
            y = train_df[label_col]
            model = _train_reg(X_train, y)
            model_dir = PROJECT_ROOT / "ml_models" / name
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": model, "features": feat_cols}, model_dir / "model.pkl")
            print(f"Trained {name}")

        # Classification models
        for name, label_col in CLS_MODELS.items():
            y = train_df[label_col]
            model = _train_cls(X_train, y)
            model_dir = PROJECT_ROOT / "ml_models" / name
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": model, "features": feat_cols}, model_dir / "model.pkl")
            print(f"Trained {name}")

        print(f"Saved train split to {TRAIN_PATH} and test split to {TEST_PATH}")
        print("Completed training for all models.")


if __name__ == "__main__":
    ModelTrainingMaster().train_all()
