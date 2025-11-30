"""
Run inference for exit models on a provided dataset.

Usage:
    python -m ml_models.run_exit_predictions --model-path ml_models/models/mfe_model.pkl \
        --input data/processed/exit_test.parquet \
        --output data/processed/exit_predictions.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml_models.model_utils import get_feature_schema

logger = logging.getLogger(__name__)


def load_model(model_path: Path) -> Any:
    obj = joblib.load(model_path)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"], obj.get("feature_names")
    return obj, getattr(obj, "feature_names", None)


def prepare_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    X = df[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


def predict(model: Any, X: pd.DataFrame, want_proba: bool) -> pd.DataFrame:
    if want_proba and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if isinstance(proba, list):
            proba = np.array(proba)
        proba_df = pd.DataFrame(proba, index=X.index)
        proba_df.columns = [f"proba_{i}" for i in range(proba_df.shape[1])]
        return proba_df
    preds = model.predict(X)
    preds = np.asarray(preds)
    return pd.DataFrame({"prediction": preds}, index=X.index)


def run(model_path: Path, input_path: Path, output_path: Path, want_proba: bool) -> None:
    feature_schema = get_feature_schema()
    df = pd.read_parquet(input_path)
    model, saved_features = load_model(model_path)
    features = saved_features or feature_schema
    X = prepare_features(df, features)
    preds_df = predict(model, X, want_proba)
    result = pd.concat([df.reset_index(drop=True), preds_df.reset_index(drop=True)], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    logger.info("Saved predictions to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Run exit model predictions on a dataset.")
    parser.add_argument("--model-path", required=True, type=Path, help="Path to saved model (joblib).")
    parser.add_argument("--input", required=True, type=Path, help="Input parquet file.")
    parser.add_argument("--output", required=True, type=Path, help="Output parquet file for predictions.")
    parser.add_argument("--proba", action="store_true", help="If set, use predict_proba when available.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.model_path, args.input, args.output, args.proba)


if __name__ == "__main__":
    main()
