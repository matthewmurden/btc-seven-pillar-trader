"""
Meta Exit Feature Engine — Ultra-Optimised Version
--------------------------------------------------
Produces stable, drift-resistant meta-signals derived from all exit models.
Includes:
- asymmetry ratio
- reward/risk prediction
- survival confidence
- volatility-after-entry delta
- exit-direction consistency
- expected exit time
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from ml_models.model_utils import get_feature_schema


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "processed" / "exit_features_with_mini_engines.parquet"
MODELS_DIR = BASE_DIR / "ml_models" / "models"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "meta_exit_features.parquet"

REQUIRED_MINI_COLS = [
    "mae_risk_score",
    "trend_continuation_score",
    "path_entropy_score",
    "exit_regime",
    "mfe_potential_score",
]

MODEL_FILES = {
    "mae": "mae_model.pkl",
    "mfe": "mfe_model.pkl",
    "opt_tp": "optimal_tp_model.pkl",
    "opt_sl": "optimal_sl_model.pkl",
    "exit_time": "exit_time_model.pkl",
    "tp_first": "tp_first_model.pkl",
    "sl_first": "sl_first_model.pkl",
    "survival": "survival_model.pkl",
}

# Avoid floating point blowups.
EPS = 1e-8


# -------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------
def _start_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _require_columns(df: pd.DataFrame, cols: list[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {context}: {missing}")


def _load_model(path: Path):
    """Load any model format: raw booster, sklearn model, or dict-wrapped model."""
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")

    obj = joblib.load(path)

    # Case 1 — dict wrapper {"model": ..., "feature_names": [...]}
    if isinstance(obj, dict):
        model = obj.get("model", obj)   # fallback: entire dict IS the model
        feats = obj.get("feature_names")
        return model, feats

    # Case 2 — sklearn/lightgbm model with attribute .feature_names
    feats = getattr(obj, "feature_names", None)

    # Case 3 — raw LightGBM booster (no .predict_proba, no .feature_names)
    return obj, feats


def _prepare_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    _require_columns(df, feature_names, "model input feature schema")

    X = df[feature_names].astype(np.float64)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


def _predict(model, X: pd.DataFrame) -> np.ndarray:
    """Uniform prediction for LightGBM Booster and sklearn models."""
    if hasattr(model, "predict"):
        pred = model.predict(X)
    else:
        # LightGBM Booster via raw booster interface
        pred = model.predict(X.to_numpy())
    return np.asarray(pred, dtype=np.float64)


def _predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return np.asarray(proba, dtype=np.float64)

    # fallback for boosters → convert predicted scores to pseudo-proba
    raw = _predict(model, X)
    prob = 1.0 / (1.0 + np.exp(-raw))
    return np.column_stack([1 - prob, prob])


# -------------------------------------------------------------------------
# Meta Feature Construction
# -------------------------------------------------------------------------
def _compute_survival_confidence(exit_time_pred: np.ndarray) -> np.ndarray:
    """
    Adaptive exponential decay based on typical exit time distribution.
    survival_conf = exp(-t / (median_t * 4))
    """
    arr = np.asarray(exit_time_pred, dtype=np.float64)
    arr = np.clip(arr, 0, None)

    med = np.median(arr) if np.isfinite(arr).all() else 2.0
    scale = max(med * 4.0, 1.0)

    return np.exp(-arr / scale)


def _clip_and_validate(meta: pd.DataFrame) -> pd.DataFrame:
    meta["asymmetry_ratio"] = meta["asymmetry_ratio"].clip(-1e6, 1e6)
    meta["reward_risk_prediction"] = meta["reward_risk_prediction"].clip(-1e6, 1e6)
    meta["survival_confidence"] = meta["survival_confidence"].clip(0, 1)
    meta["exit_direction_consistency"] = meta["exit_direction_consistency"].clip(-1, 1)
    meta["expected_exit_time"] = meta["expected_exit_time"].clip(lower=0)

    if meta.isna().any().any():
        raise ValueError("NaNs detected in meta exit features")

    if not np.isfinite(meta.to_numpy()).all():
        raise ValueError("Non-finite meta exit feature detected")

    return meta


# -------------------------------------------------------------------------
# Main Engine
# -------------------------------------------------------------------------
def apply_meta_features(
    input_path: Path = INPUT_PATH,
    models_dir: Path = MODELS_DIR,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    _start_logging()
    t0 = time.time()

    logger.info("Loading base features from %s", input_path)
    df = pd.read_parquet(input_path)
    _require_columns(df, REQUIRED_MINI_COLS, "mini engine outputs")

    feature_schema = get_feature_schema()
    _require_columns(df, feature_schema, "feature schema")

    # Load every model once
    models: Dict[str, Tuple[object, list[str] | None]] = {
        key: _load_model(models_dir / file)
        for key, file in MODEL_FILES.items()
    }

    logger.info("Predicting model outputs…")

    def _pred(key: str):
        model, feats = models[key]
        feats = feats or feature_schema
        return _predict(model, _prepare_features(df, feats))

    def _proba(key: str):
        model, feats = models[key]
        feats = feats or feature_schema
        return _predict_proba(model, _prepare_features(df, feats))

    # Regression outputs
    mae_pred = _pred("mae")
    mfe_pred = _pred("mfe")
    opt_tp = _pred("opt_tp")
    opt_sl = _pred("opt_sl")
    exit_time_pred = _pred("exit_time")

    # Classification probabilities
    tp_proba = _proba("tp_first")[:, 1]
    sl_proba = _proba("sl_first")[:, 1]

    # Survival confidence
    survival_conf = _compute_survival_confidence(exit_time_pred)

    # Meta computations
    asymmetry_ratio = mfe_pred / np.maximum(np.abs(mae_pred), EPS)
    reward_risk_prediction = opt_tp / np.maximum(np.abs(opt_sl), EPS)
    volatility_after_entry_pred = mfe_pred - np.abs(mae_pred)
    exit_direction_consistency = tp_proba - sl_proba
    expected_exit_time = np.maximum(exit_time_pred, 0)

    def _zscore(x: np.ndarray) -> np.ndarray:
        mean = np.mean(x)
        std = np.std(x) + 1e-12
        return (x - mean) / std

    meta = pd.DataFrame(
        {
            "asymmetry_ratio": asymmetry_ratio,
            "reward_risk_prediction": reward_risk_prediction,
            "survival_confidence": survival_conf,
            "volatility_after_entry_pred": volatility_after_entry_pred,
            "exit_direction_consistency": exit_direction_consistency,
            "expected_exit_time": expected_exit_time,
            "asymmetry_z": _zscore(asymmetry_ratio),
            "reward_risk_z": _zscore(reward_risk_prediction),
            "exit_direction_z": _zscore(exit_direction_consistency),
            "volatility_after_entry_z": _zscore(volatility_after_entry_pred),
        },
        index=df.index,
    )

    # Validate + save
    meta = _clip_and_validate(meta).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(output_path, index=False)

    logger.info("Saved meta exit features to %s", output_path)
    logger.info("Completed in %.2f seconds", time.time() - t0)

    return meta


if __name__ == "__main__":
    apply_meta_features()
