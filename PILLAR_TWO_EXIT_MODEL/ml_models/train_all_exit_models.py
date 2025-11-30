"""
Train all exit models sequentially on the prepared training dataset.

This script:
- Builds the training dataset if missing (features + labels).
- Trains each model with a fixed random seed.
- Saves models under ml_models/models/.
"""

from __future__ import annotations

import logging
from pathlib import Path

from data.build_exit_training_set import build_training_dataset
from ml_models.model_utils import get_feature_schema, load_training_data
from ml_models.mae_model import MAEModel
from ml_models.mfe_model import MFEModel
from ml_models.exit_time_model import ExitTimeModel
from ml_models.tp_first_model import TPFirstModel
from ml_models.sl_first_model import SLFirstModel
from ml_models.optimal_tp_model import OptimalTPModel
from ml_models.optimal_sl_model import OptimalSLModel
from ml_models.survival_model import SurvivalExitModel


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "processed" / "exit_training_dataset.parquet"
MODELS_DIR = BASE_DIR / "ml_models" / "models"


def _ensure_dataset():
    if DATASET_PATH.exists():
        df = load_training_data(DATASET_PATH)
    else:
        _, _ = build_training_dataset()
        df = load_training_data(DATASET_PATH)
    return df


def train_all_models():
    feature_schema = get_feature_schema()
    df = _ensure_dataset()

    # Features and labels
    X = df[feature_schema].copy()
    y_mae = df["mae"]
    y_mfe = df["mfe"]
    y_exit = df["exit_time"]
    y_tp = df["tp_first"]
    y_sl = df["sl_first"]
    y_opt_tp = df["optimal_tp"]
    y_opt_sl = df["optimal_sl"]
    y_survival_time = df["survival_time"]
    y_event_flag = df["event_flag"]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("mae_model.pkl", MAEModel(feature_names=feature_schema), y_mae),
        ("mfe_model.pkl", MFEModel(feature_names=feature_schema), y_mfe),
        ("exit_time_model.pkl", ExitTimeModel(feature_names=feature_schema), y_exit),
        ("tp_first_model.pkl", TPFirstModel(feature_names=feature_schema), y_tp),
        ("sl_first_model.pkl", SLFirstModel(feature_names=feature_schema), y_sl),
        ("optimal_tp_model.pkl", OptimalTPModel(feature_names=feature_schema), y_opt_tp),
        ("optimal_sl_model.pkl", OptimalSLModel(feature_names=feature_schema), y_opt_sl),
    ]

    for filename, model, target in tasks:
        logger.info("Training %s", filename)
        model.fit(X, target)
        model.save(MODELS_DIR / filename)

    # Survival model
    survival = SurvivalExitModel(feature_names=feature_schema)
    logger.info("Training survival model")
    survival.fit(X, y_survival_time, y_event_flag)
    survival.save(MODELS_DIR / "survival_model.pkl")

    logger.info("All exit models trained and saved to %s", MODELS_DIR)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train_all_models()


if __name__ == "__main__":
    main()
