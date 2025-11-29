"""Generate predictions on the test split using trained exit models."""

from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TEST_PATH = DATA_DIR / "exit_test.parquet"
OUTPUT_PATH = DATA_DIR / "exit_test_with_preds.parquet"

MODEL_MAP = {
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


def generate_test_predictions() -> Path:
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing test split at {TEST_PATH}")
    df = pd.read_parquet(TEST_PATH)
    for pred_col, model_name in MODEL_MAP.items():
        model_dir = PROJECT_ROOT / "ml_models" / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory missing: {model_dir}")
        model, feat_cols = _load_model(model_dir)
        X = df.reindex(columns=feat_cols).fillna(0)
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(X)[:, 1]
        else:
            preds = model.predict(X)
        df[pred_col] = preds

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH)
    return OUTPUT_PATH


if __name__ == "__main__":
    generate_test_predictions()
