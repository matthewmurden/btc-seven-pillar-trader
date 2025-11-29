"""Master predictor for exit models (Part 4)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
TRAINING_PATH = BASE_DIR / "data" / "processed" / "exit_training.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "exit_with_predictions.parquet"

MODEL_CONFIG: Dict[str, str] = {
    "mae_pred": "Model_MAE",
    "mfe_pred": "Model_MFE",
    "exit_time_pred": "Model_ExitTime",
    "optimal_tp_pred": "Model_OptimalTP",
    "optimal_sl_pred": "Model_OptimalSL",
    "tp_first_pred": "Model_TP_First",
    "sl_first_pred": "Model_SL_First",
}


def _load_model(model_dir: Path):
    bundle = joblib.load(model_dir / "model.pkl")
    return bundle["model"], bundle["features"]


class ExitModelPredictionMaster:
    """Loads all exit models and applies predictions to a dataframe."""

    def __init__(self) -> None:
        self.model_dirs = {name: BASE_DIR / "ml_models" / dirname for name, dirname in MODEL_CONFIG.items()}
        self.models = {
            name: _load_model(path) for name, path in self.model_dirs.items()
        }

    def run_all(self, df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
        output = df.copy()
        for pred_col, _dir in MODEL_CONFIG.items():
            model, feat_cols = self.models[pred_col]
            X = output.reindex(columns=feat_cols).fillna(0)
            if hasattr(model, "predict_proba"):
                preds = model.predict_proba(X)[:, 1]
            else:
                preds = model.predict(X)
            output[pred_col] = preds

        if save:
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            output.to_parquet(OUTPUT_PATH)
        return output


if __name__ == "__main__":
    if not TRAINING_PATH.exists():
        raise FileNotFoundError(f"Training dataset missing at {TRAINING_PATH}")
    df = pd.read_parquet(TRAINING_PATH)
    ExitModelPredictionMaster().run_all(df)
