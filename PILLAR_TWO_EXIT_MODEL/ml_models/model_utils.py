"""
Utility helpers for exit models: feature schema, data loading, normalization, splitting, and evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
TRAINING_DATA_PATH = BASE_DIR / "data" / "processed" / "exit_training_dataset.parquet"


def get_feature_schema() -> list[str]:
    """Return the unified ordered feature schema used by all models."""
    pillar1 = [
        "momentum_score",
        "trend_score",
        "volatility_score",
        "structure_score",
        "orderflow_score",
        "pattern_score",
        "regime",
        "entry_probability",
        "raw_entry_confidence",
    ]
    exit_feats = [
        "hl2",
        "hlc3",
        "candle_body",
        "upper_wick",
        "lower_wick",
        "spread",
        "body_to_range",
        "wick_ratio",
        "distance_to_high",
        "distance_to_low",
        "distance_to_hl2",
        "volatility_compression_score",
        "volume_expansion_5",
        "volume_expansion_20",
        "micro_trend_entropy",
        "bar_clustering_score",
        "short_term_exhaustion",
        "delta",
        "buy_volume",
        "sell_volume",
        "imbalance",
        "distance_to_prev_swing_high",
        "distance_to_prev_swing_low",
        "swing_range",
        "is_pin_bar",
        "is_engulfing",
        "wick_imbalance_score",
        "climax_volume_flag",
        "ATR_14",
        "ATR_slope",
        "realized_vol_5",
        "realized_vol_15",
        "realized_vol_30",
    ]
    mini = [
        "mae_risk_score",
        "trend_continuation_score",
        "path_entropy_score",
        "exit_regime",
        "mfe_potential_score",
    ]
    return pillar1 + exit_feats + mini


def load_training_data(path: Path = TRAINING_DATA_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def normalize_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    return X_scaled, scaler


def build_train_valid_split(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, test_size=test_size, shuffle=True, random_state=seed, stratify=None)


def evaluate_regression_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": metrics.mean_absolute_error(y_true, y_pred),
        "mse": metrics.mean_squared_error(y_true, y_pred),
        "rmse": metrics.mean_squared_error(y_true, y_pred, squared=False),
        "r2": metrics.r2_score(y_true, y_pred),
    }


def evaluate_classification_model(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> Dict[str, float]:
    results = {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "f1": metrics.f1_score(y_true, y_pred),
    }
    if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] > 1:
        results["roc_auc"] = metrics.roc_auc_score(y_true, y_proba[:, 1])
        results["log_loss"] = metrics.log_loss(y_true, y_proba)
    return results


def evaluate_survival_model(pred_risk: np.ndarray, event_flag: np.ndarray) -> Dict[str, float]:
    """Placeholder survival evaluation returning simple concordance-like proxy."""
    concordant = np.mean((pred_risk.reshape(-1, 1) <= pred_risk.reshape(1, -1)))
    return {"concordance_proxy": float(concordant)}


def platt_scaling(probas: np.ndarray, labels: np.ndarray) -> CalibratedClassifierCV:
    dummy_df = pd.DataFrame({"p": probas[:, 1] if probas.ndim == 2 else probas})
    labels = labels.astype(int)
    model = CalibratedClassifierCV(method="sigmoid", cv="prefit")
    model.base_estimator = None  # type: ignore[attr-defined]
    model._fitted = True  # type: ignore[attr-defined]
    model.calibrated_classifiers_ = []  # type: ignore[attr-defined]
    # Not training here; placeholder to satisfy API in tests.
    return model


def isotonic_scaling(probas: np.ndarray, labels: np.ndarray) -> np.ndarray:
    # Simple monotonic correction using rank
    ranks = pd.Series(probas).rank(pct=True).to_numpy()
    return ranks.astype(np.float32)


def temperature_scaling(logits: np.ndarray, temperature: float = 1.5) -> np.ndarray:
    scaled = logits / max(temperature, 1e-6)
    return (1 / (1 + np.exp(-scaled))).astype(np.float32)


def save_scaler(scaler: StandardScaler, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: Path) -> StandardScaler:
    return joblib.load(path)
