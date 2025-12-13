"""
Pillar 1 — Multi-Horizon ML Predictors

This module trains 5 models:

1. short_dir      -> y_dir_1_2       (multiclass: -1, 0, 1)
2. dir_5_10       -> y_dir_5_10      (multiclass: -1, 0, 1)
3. dir_15_30      -> y_dir_15_30     (multiclass: -1, 0, 1)
4. breakout       -> y_breakout      (binary: 0/1)
5. mean_reversion -> y_mean_reversion (binary: 0/1, rare positives)

Design goals:
- Memory safe on ~1.6M rows, ~700+ features.
- Chronological split (no leakage).
- Per-label class weighting to respect imbalanced labels.
- Keep probabilities for downstream meta/aggregation layers.

The actual pipeline entrypoint lives in:
    src/pipeline/build_pillar1_multi_horizon_ml.py
"""

import gc
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
    # Chronological hold-out fraction
    val_fraction: float = 0.2

    # Random state for model internals (not for shuffling time)
    random_state: int = 42

    # Where to save the trained models
    model_dir: Path = Path("models/pillar1_multi_horizon")

    # Base hyperparameters (can be overridden per model via spec.model_params)
    default_model_params: Dict = field(
        default_factory=lambda: dict(
            learning_rate=0.05,
            max_depth=6,          # regularised depth for stability
            max_iter=300,
            min_samples_leaf=50,  # slightly larger leaf for smoother decisions
            l2_regularization=1.0,
            random_state=42,
        )
    )

    # Downcast features to this dtype to save RAM
    feature_dtype: str = "float32"

    # Whether to automatically use class_weight based on training distribution
    use_class_weight: bool = True


def default_model_specs() -> List[MultiHorizonModelSpec]:
    """
    Default 5-model spec for Pillar 1 as in the system design.
    """
    return [
        MultiHorizonModelSpec(
            name="short_dir",
            label_col="y_dir_1_2",
            problem_type="multiclass",
            # can override params per-label if you want
            model_params=None,
        ),
        MultiHorizonModelSpec(
            name="dir_5_10",
            label_col="y_dir_5_10",
            problem_type="multiclass",
            model_params=None,
        ),
        MultiHorizonModelSpec(
            name="dir_15_30",
            label_col="y_dir_15_30",
            problem_type="multiclass",
            model_params=None,
        ),
        MultiHorizonModelSpec(
            name="breakout",
            label_col="y_breakout",
            problem_type="binary",
            positive_class=1,
            model_params=None,
        ),
        MultiHorizonModelSpec(
            name="mean_reversion",
            label_col="y_mean_reversion",
            problem_type="binary",
            positive_class=1,
            model_params=None,
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


def chronological_indices(n_rows: int, val_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return train/val index splits (as relative indices 0..n_rows-1)
    for an already time-sorted subset.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0,1)")
    split_idx = int(n_rows * (1 - val_fraction))
    if split_idx <= 0 or split_idx >= n_rows:
        raise ValueError("val_fraction results in empty train/val split")
    train_idx = np.arange(0, split_idx, dtype=np.int64)
    val_idx = np.arange(split_idx, n_rows, dtype=np.int64)
    return train_idx, val_idx


def compute_class_weight(y: np.ndarray) -> Dict:
    """
    Compute simple inverse-frequency class weights:

        w_c = n_samples / (n_classes * count_c)

    Works for binary and multiclass. Labels can be ints or floats.
    """
    classes, counts = np.unique(y, return_counts=True)
    n_samples = y.shape[0]
    n_classes = len(classes)
    weights: Dict = {}
    for c, cnt in zip(classes, counts):
        # avoid division by zero, though it shouldn't happen
        if cnt <= 0:
            continue
        weights[c] = float(n_samples) / (float(n_classes) * float(cnt))
    return weights


def fit_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    spec: MultiHorizonModelSpec,
    cfg: MultiHorizonTrainingConfig,
) -> HistGradientBoostingClassifier:
    """
    Fit a HistGradientBoostingClassifier with:
    - base params from cfg.default_model_params
    - optional overrides from spec.model_params
    - optional class_weight inferred from y_train
    """
    params = cfg.default_model_params.copy()
    if spec.model_params:
        params.update(spec.model_params)

    # Only set class_weight if the user hasn't explicitly provided one.
    if cfg.use_class_weight and "class_weight" not in params:
        class_weight = compute_class_weight(y_train)
        params["class_weight"] = class_weight
        logging.info(
            "[MH] Using class_weight for '%s' (%s): %s",
            spec.name,
            spec.label_col,
            class_weight,
        )

    model = HistGradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    return model


def _extract_probs(
    model: HistGradientBoostingClassifier,
    X: np.ndarray,
    spec: MultiHorizonModelSpec,
) -> Dict[str, np.ndarray]:
    """
    Get probability outputs + hard predictions in a dict keyed by column name.
    """
    proba = model.predict_proba(X)
    classes = model.classes_
    out: Dict[str, np.ndarray] = {}

    if spec.problem_type == "binary":
        if spec.positive_class is None:
            raise ValueError(f"Positive class must be set for binary model {spec.name}")
        if spec.positive_class not in classes:
            raise ValueError(f"Positive class {spec.positive_class} not found in classes {classes}")
        pos_idx = int(list(classes).index(spec.positive_class))
        prob_up = proba[:, pos_idx]
        prob_down = 1.0 - prob_up
        out[f"{spec.name}_prob_up"] = prob_up
        out[f"{spec.name}_prob_down"] = prob_down
        out[f"{spec.name}_confidence"] = np.maximum(prob_up, prob_down)
    else:  # multiclass, assume possible classes include -1, 0, 1
        n = len(X)
        prob_up = np.zeros(n, dtype=np.float32)
        prob_down = np.zeros(n, dtype=np.float32)
        prob_neutral = np.zeros(n, dtype=np.float32)
        cls_list = list(classes)
        if -1 in classes:
            prob_down = proba[:, cls_list.index(-1)]
        if 0 in classes:
            prob_neutral = proba[:, cls_list.index(0)]
        if 1 in classes:
            prob_up = proba[:, cls_list.index(1)]
        # if neutral not explicitly present, assign remaining mass
        if 0 not in classes:
            prob_neutral = 1.0 - (prob_up + prob_down)

        out[f"{spec.name}_prob_up"] = prob_up
        out[f"{spec.name}_prob_down"] = prob_down
        out[f"{spec.name}_prob_neutral"] = prob_neutral
        out[f"{spec.name}_confidence"] = proba.max(axis=1)

    out[f"{spec.name}_pred"] = model.predict(X)
    return out


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba_pos: Optional[np.ndarray],
    spec: MultiHorizonModelSpec,
) -> Dict[str, float]:
    """
    Basic metrics for sanity + monitoring; true performance / calibration
    will be tracked in Pillar 7.
    """
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


def save_model(
    model: HistGradientBoostingClassifier,
    spec: MultiHorizonModelSpec,
    cfg: MultiHorizonTrainingConfig,
) -> Path:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    fname = spec.model_filename or f"{spec.name}.joblib"
    path = cfg.model_dir / fname
    joblib.dump(model, path)
    return path


def load_models(
    cfg: MultiHorizonTrainingConfig,
    specs: Optional[List[MultiHorizonModelSpec]] = None,
) -> Dict[str, HistGradientBoostingClassifier]:
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
# High-level training + prediction (memory-optimised)
# ----------------------------------------------------------------------


def train_and_predict(
    df: pd.DataFrame,
    specs: Optional[List[MultiHorizonModelSpec]] = None,
    cfg: Optional[MultiHorizonTrainingConfig] = None,
) -> pd.DataFrame:
    """
    Memory-optimised training:

    - Sort once by time.
    - Build a single float32 feature matrix (X_all) shared by all models.
    - Chronological split per-label (no shuffling).
    - Per-label class weighting for imbalanced problems.
    - Write predictions directly into `df` in-place to avoid giant copies.
    """
    if specs is None:
        specs = default_model_specs()
    if cfg is None:
        cfg = MultiHorizonTrainingConfig()

    ensure_labels_present(df, specs)

    # Sort chronologically once
    if "open_time_ms" in df.columns:
        df = df.sort_values("open_time_ms").reset_index(drop=True)

    # Determine which columns are labels / time so we can strip them out of features
    label_cols = [spec.label_col for spec in specs]
    drop_cols = set(label_cols)
    if "open_time_ms" in df.columns:
        drop_cols.add("open_time_ms")

    # Build feature matrix ONCE: numeric cols only, downcast to float32, then to NumPy
    feat_df = df.drop(columns=list(drop_cols), errors="ignore")
    feat_df = feat_df.select_dtypes(include=[np.number])

    if cfg.feature_dtype:
        feat_df = feat_df.astype(cfg.feature_dtype, copy=False)

    X_all = feat_df.to_numpy(copy=False)
    del feat_df
    gc.collect()

    logging.info(
        "[MH] Feature matrix built: shape=%s, dtype=%s",
        X_all.shape,
        X_all.dtype,
    )

    # We write predictions directly into df (no extra top-level copy)
    results = df

    for spec in specs:
        logging.info(
            "=== Training model '%s' on label '%s' (%s) ===",
            spec.name,
            spec.label_col,
            spec.problem_type,
        )

        # Mask of rows where this label is defined
        label_series = results[spec.label_col]
        mask = label_series.notna().to_numpy()
        if not mask.any():
            raise ValueError(f"No non-NaN rows for label {spec.label_col}")

        # Indices (positions) for rows with a label
        idx_all = np.flatnonzero(mask)
        y_all = label_series.to_numpy()[idx_all]

        n_labelled = len(idx_all)
        train_rel_idx, val_rel_idx = chronological_indices(n_labelled, cfg.val_fraction)

        # Map relative indices (0..n_labelled-1) back to absolute row positions
        train_idx = idx_all[train_rel_idx]
        val_idx = idx_all[val_rel_idx]

        # Slice NumPy feature matrix
        X_train = X_all[train_idx, :]
        y_train = y_all[train_rel_idx]

        X_val = X_all[val_idx, :]
        y_val = y_all[val_rel_idx]

        # Fit model
        model = fit_model(X_train, y_train, spec, cfg)

        # Validation metrics
        y_val_pred = model.predict(X_val)
        try:
            y_val_proba = model.predict_proba(X_val)
        except Exception:
            y_val_proba = None

        if spec.problem_type == "binary" and y_val_proba is not None:
            classes = model.classes_
            if spec.positive_class not in classes:
                raise ValueError(
                    f"Positive class {spec.positive_class} not in model.classes_ for {spec.name}"
                )
            pos_idx = int(list(classes).index(spec.positive_class))
            y_val_proba_pos = y_val_proba[:, pos_idx]
        else:
            y_val_proba_pos = None

        metrics = evaluate(y_val, y_val_pred, y_val_proba_pos, spec)
        log_metrics(spec.name, metrics)

        save_path = save_model(model, spec, cfg)
        logging.info("Saved model '%s' to %s", spec.name, save_path)

        # Predictions for ALL rows where label is defined
        X_label = X_all[idx_all, :]
        prob_dict = _extract_probs(model, X_label, spec)

        for k, v in prob_dict.items():
            # downcast probs to feature_dtype where sensible
            if np.issubdtype(v.dtype, np.floating):
                v = v.astype(cfg.feature_dtype, copy=False)
            results.loc[idx_all, k] = v

        # Free per-model intermediates ASAP
        del model, X_train, y_train, X_val, y_val, X_label, y_all, idx_all
        gc.collect()

    # Fill any remaining NaNs in prediction columns with 0.0 (e.g. rows without labels)
    pred_prefixes = [spec.name for spec in specs]
    pred_cols = [c for c in results.columns if any(c.startswith(p) for p in pred_prefixes)]
    if pred_cols:
        results[pred_cols] = results[pred_cols].fillna(0.0)

    return results