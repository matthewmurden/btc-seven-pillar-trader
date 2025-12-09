import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


# ----------------------------------------------------------------------
# Model specifications and config
# ----------------------------------------------------------------------
@dataclass
class MultiHorizonModelSpec:
    name: str  # e.g., "short_dir"
    label_col: str  # e.g., "y_dir_1_2"
    problem_type: str  # "binary" or "multiclass"
    positive_class: Optional[int] = 1  # for binary labels
    model_filename: Optional[str] = None
    model_params: Optional[Dict] = None


@dataclass
class MultiHorizonTrainingConfig:
    val_fraction: float = 0.2  # chronological split
    random_state: int = 42
    model_dir: Path = Path("models/pillar1_multi_horizon")
    default_model_params: Dict = field(
        default_factory=lambda: dict(
            learning_rate=0.05,
            max_depth=None,
            max_iter=300,
            min_samples_leaf=20,
            l2_regularization=0.0,
            random_state=42,
        )
    )


def default_model_specs() -> List[MultiHorizonModelSpec]:
    return [
        MultiHorizonModelSpec(name="short_dir", label_col="y_dir_1_2", problem_type="multiclass"),
        MultiHorizonModelSpec(name="dir_5_10", label_col="y_dir_5_10", problem_type="multiclass"),
        MultiHorizonModelSpec(name="dir_15_30", label_col="y_dir_15_30", problem_type="multiclass"),
        MultiHorizonModelSpec(
            name="breakout",
            label_col="y_breakout",
            problem_type="binary",
            positive_class=1,
        ),
        MultiHorizonModelSpec(
            name="mean_reversion",
            label_col="y_mean_reversion",
            problem_type="binary",
            positive_class=1,
        ),
    ]


# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def ensure_labels_present(df: pd.DataFrame, specs: List[MultiHorizonModelSpec]) -> None:
    missing = [spec.label_col for spec in specs if spec.label_col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required label columns: {missing}. "
            "Ensure labels_pillar1.parquet contains these columns."
        )


def chronological_split(df: pd.DataFrame, val_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0,1)")
    n = len(df)
    split_idx = int(n * (1 - val_fraction))
    if split_idx <= 0 or split_idx >= n:
        raise ValueError("val_fraction results in empty train/val split")
    return df.iloc[:split_idx], df.iloc[split_idx:]


def build_features_and_label(
    df: pd.DataFrame, label_col: str, drop_cols: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.dropna(subset=[label_col])
    y = df[label_col]
    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include=[np.number])  # <- optional safety
    return X, y


def fit_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    spec: MultiHorizonModelSpec,
    cfg: MultiHorizonTrainingConfig,
) -> HistGradientBoostingClassifier:
    params = cfg.default_model_params.copy()
    if spec.model_params:
        params.update(spec.model_params)
    model = HistGradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    return model


def _extract_probs(
    model: HistGradientBoostingClassifier, X: pd.DataFrame, spec: MultiHorizonModelSpec
) -> Dict[str, np.ndarray]:
    proba = model.predict_proba(X)
    classes = model.classes_
    out: Dict[str, np.ndarray] = {}

    if spec.problem_type == "binary":
        if spec.positive_class is None:
            raise ValueError(f"Positive class must be set for binary model {spec.name}")
        # map positive class prob to prob_up
        if spec.positive_class not in classes:
            raise ValueError(f"Positive class {spec.positive_class} not found in classes {classes}")
        pos_idx = list(classes).index(spec.positive_class)
        prob_up = proba[:, pos_idx]
        prob_down = 1.0 - prob_up
        out[f"{spec.name}_prob_up"] = prob_up
        out[f"{spec.name}_prob_down"] = prob_down
        out[f"{spec.name}_confidence"] = np.maximum(prob_up, prob_down)
    else:  # multiclass, assume possible classes include -1, 0, 1
        prob_up = prob_down = prob_neutral = np.zeros(len(X))
        if -1 in classes:
            prob_down = proba[:, list(classes).index(-1)]
        if 0 in classes:
            prob_neutral = proba[:, list(classes).index(0)]
        if 1 in classes:
            prob_up = proba[:, list(classes).index(1)]
        # fallback if 0 not present: distribute remaining mass
        if 0 not in classes:
            prob_neutral = 1.0 - (prob_up + prob_down)
        out[f"{spec.name}_prob_up"] = prob_up
        out[f"{spec.name}_prob_down"] = prob_down
        out[f"{spec.name}_prob_neutral"] = prob_neutral
        out[f"{spec.name}_confidence"] = proba.max(axis=1)

    out[f"{spec.name}_pred"] = model.predict(X)
    return out


def evaluate(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba_pos: Optional[np.ndarray],
    spec: MultiHorizonModelSpec,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

    if spec.problem_type == "binary" and y_proba_pos is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba_pos)
        except Exception:
            metrics["roc_auc"] = np.nan
        try:
            metrics["brier"] = brier_score_loss(y_true, y_proba_pos)
        except Exception:
            metrics["brier"] = np.nan
    else:
        metrics["roc_auc"] = np.nan
        metrics["brier"] = np.nan

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm
    return metrics


def log_metrics(name: str, metrics: Dict[str, float]) -> None:
    logging.info("=== Metrics for %s ===", name)
    for k, v in metrics.items():
        if k == "confusion_matrix":
            logging.info("%s:\n%s", k, v)
        else:
            logging.info("%s: %.4f", k, v if not isinstance(v, np.ndarray) else np.nan)


def save_model(model: HistGradientBoostingClassifier, spec: MultiHorizonModelSpec, cfg: MultiHorizonTrainingConfig) -> Path:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    fname = spec.model_filename or f"{spec.name}.joblib"
    path = cfg.model_dir / fname
    joblib.dump(model, path)
    return path


def load_models(cfg: MultiHorizonTrainingConfig, specs: Optional[List[MultiHorizonModelSpec]] = None) -> Dict[str, HistGradientBoostingClassifier]:
    specs = specs or default_model_specs()
    models: Dict[str, HistGradientBoostingClassifier] = {}
    for spec in specs:
        fname = spec.model_filename or f"{spec.name}.joblib"
        path = cfg.model_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        models[spec.name] = joblib.load(path)
    return models


# ----------------------------------------------------------------------
# High-level training + prediction
# ----------------------------------------------------------------------
def train_and_predict(
    df: pd.DataFrame,
    specs: Optional[List[MultiHorizonModelSpec]] = None,
    cfg: Optional[MultiHorizonTrainingConfig] = None,
) -> pd.DataFrame:
    if specs is None:
        specs = default_model_specs()
    if cfg is None:
        cfg = MultiHorizonTrainingConfig()

    ensure_labels_present(df, specs)

    # Sort chronologically
    if "open_time_ms" in df.columns:
        df = df.sort_values("open_time_ms").reset_index(drop=True)

    results = df.copy()
    # Drop all label columns and raw time index from features
    label_cols = [spec.label_col for spec in specs]
    drop_cols = label_cols.copy()
    if "open_time_ms" in df.columns:
        drop_cols.append("open_time_ms")

    for spec in specs:
        logging.info("Training model '%s' on label '%s' (%s)", spec.name, spec.label_col, spec.problem_type)

        # Drop rows with NaN label, BUT KEEP ORIGINAL INDEX for alignment
        df_label = df.dropna(subset=[spec.label_col]).copy()
        if df_label.empty:
            raise ValueError(f"No non-NaN rows for label {spec.label_col}")

        train_df, val_df = chronological_split(df_label, cfg.val_fraction)

        X_train, y_train = build_features_and_label(train_df, spec.label_col, drop_cols)
        X_val, y_val = build_features_and_label(val_df, spec.label_col, drop_cols)

        model = fit_model(X_train, y_train, spec, cfg)

        # Validation metrics
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)

        if spec.problem_type == "binary":
            # get probability of positive_class from predict_proba
            classes = model.classes_
            if spec.positive_class not in classes:
                raise ValueError(f"Positive class {spec.positive_class} not in model.classes_ for {spec.name}")
            pos_idx = list(classes).index(spec.positive_class)
            y_val_proba_pos = y_val_proba[:, pos_idx]
        else:
            y_val_proba_pos = None

        metrics = evaluate(y_val, y_val_pred, y_val_proba_pos, spec)
        log_metrics(spec.name, metrics)

        save_path = save_model(model, spec, cfg)
        logging.info("Saved model '%s' to %s", spec.name, save_path)

        # Full dataset predictions (only where label is defined)
        X_full, _ = build_features_and_label(df_label, spec.label_col, drop_cols)
        prob_dict = _extract_probs(model, X_full, spec)

        # Write predictions back aligned on original indices
        for k, v in prob_dict.items():
            results.loc[df_label.index, k] = v

    # Clean outputs
    pred_cols = [c for c in results.columns if any(c.startswith(spec.name) for spec in specs)]
    results[pred_cols] = results[pred_cols].fillna(0.0)
    return results
