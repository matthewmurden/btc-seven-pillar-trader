"""Validation for Pillar Two Part 4 multi-horizon predictors."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_engine.feature_engine_master import FeatureEngineMaster  # noqa: E402
from mini_engines.mini_engines_master import MiniEnginesMaster  # noqa: E402
from label_builder.label_builder import LabelBuilder, build_and_save_labels  # noqa: E402
from data_pipeline.merge_features_and_labels import merge_features_and_labels  # noqa: E402
from ml_models.model_prediction_master import ModelPredictionMaster  # noqa: E402
from ml_models.model_training_master import ModelTrainingMaster  # noqa: E402


RAW_PATH = PROJECT_ROOT / "data" / "raw" / "exit_raw.parquet"
FEATURES_PART3_PATH = PROJECT_ROOT / "data" / "processed" / "features_part3.parquet"
LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "labels.parquet"
MERGED_PATH = PROJECT_ROOT / "data" / "processed" / "features_part3_with_labels.parquet"

MODEL_PATHS = [
    PROJECT_ROOT / "ml_models" / "model_1m" / "model.pkl",
    PROJECT_ROOT / "ml_models" / "model_5m" / "model.pkl",
    PROJECT_ROOT / "ml_models" / "model_15m" / "model.pkl",
    PROJECT_ROOT / "ml_models" / "breakout_model" / "model.pkl",
    PROJECT_ROOT / "ml_models" / "reversion_model" / "model.pkl",
]

PRED_COLS = [
    "pred_1m_up",
    "pred_5m_up",
    "pred_15m_up",
    "pred_breakout",
    "pred_reversion",
]

LABEL_COLS = [
    "forward_returns_1m",
    "forward_returns_5m",
    "forward_returns_15m",
    "breakout_label",
    "reversion_label",
]


def ensure_features_part3() -> pd.DataFrame:
    if FEATURES_PART3_PATH.exists():
        return pd.read_parquet(FEATURES_PART3_PATH)
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data missing at {RAW_PATH}")
    raw_df = pd.read_parquet(RAW_PATH)
    feats = FeatureEngineMaster().run_all(raw_df, verbose=False)
    feats = MiniEnginesMaster().run_all(feats, verbose=False)
    FEATURES_PART3_PATH.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(FEATURES_PART3_PATH)
    return feats


def ensure_labels() -> pd.DataFrame:
    if LABELS_PATH.exists():
        existing = pd.read_parquet(LABELS_PATH)
        if existing.get("reversion_label", pd.Series(dtype=int)).sum() > 0:
            return existing
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data missing at {RAW_PATH}")
    raw_df = pd.read_parquet(RAW_PATH)
    build_and_save_labels(raw_df, LABELS_PATH, LabelBuilder())
    return pd.read_parquet(LABELS_PATH)


def ensure_merged() -> pd.DataFrame:
    if MERGED_PATH.exists():
        return pd.read_parquet(MERGED_PATH)
    ensure_features_part3()
    ensure_labels()
    merge_features_and_labels()
    return pd.read_parquet(MERGED_PATH)


def ensure_models(features_df: pd.DataFrame):
    if all(path.exists() for path in MODEL_PATHS):
        return
    missing = [c for c in LABEL_COLS if c not in features_df.columns]
    if missing:
        raise ValueError(f"Cannot train models; missing required label columns: {missing}")
    trainer = ModelTrainingMaster()
    trainer.train_all()


def assert_predictions(df: pd.DataFrame, cols) -> None:
    for col in cols:
        if col not in df.columns:
            raise AssertionError(f"Missing prediction column: {col}")
        series = df[col]
        if not is_numeric_dtype(series):
            raise TypeError(f"Prediction column {col} is not numeric (dtype={series.dtype})")
        values = series.dropna()
        if not np.isfinite(values).all():
            raise ValueError(f"Prediction column {col} contains non-finite values")
        if not ((values >= 0).all() and (values <= 1).all()):
            raise AssertionError(f"Prediction column {col} outside [0, 1]")


def main() -> None:
    df_part3 = ensure_merged()
    ensure_models(df_part3)

    predictor = ModelPredictionMaster()
    df_out = predictor.run_all(df_part3, save=False)

    if df_out.shape[0] != df_part3.shape[0]:
        raise AssertionError("Output dataframe length changed after predictions.")

    assert_predictions(df_out, PRED_COLS)

    print("Part 4 ML Predictors passed")
    print(df_out[PRED_COLS].head())


if __name__ == "__main__":
    main()
