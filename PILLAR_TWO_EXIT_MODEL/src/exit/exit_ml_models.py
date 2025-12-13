# src/exit/exit_ml_models.py

"""
Pillar 2 — ML Exit Models (Core Predictors)

This module trains and applies the ML components of Pillar 2:

  - TP distance regressor
  - SL distance regressor
  - TP-first probability classifier
  - Exit time forecaster
  - Volatility expansion impact model

Inputs (from merged dataframe):
  - all raw exit features (ExitRaw)
  - mini-engine outputs (ExitMini)
  - deterministic exit profile (ExitDeterministic)
  - Pillar 1 meta + directional context (from features_pillar1_final)
  - Pillar 4 regime features

Targets (from exit_labels + exit_with_deterministic):
  - optimal_tp_distance
  - optimal_sl_distance
  - exit_outcome (→ binary "tp_first")
  - expected_exit_time
  - exit_vol_short_trend  (volatility expansion impact proxy)

Outputs (columns added to df):
  - predicted_tp_distance
  - predicted_sl_distance
  - predicted_TP_first_probability
  - predicted_exit_time
  - predicted_volatility_effect

All classification probabilities are calibrated with Platt (logistic)
or Isotonic regression, selected by Brier score on a chrono hold-out
validation set.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    brier_score_loss,
    roc_auc_score,
    accuracy_score,
)


# ---------------------------------------------------------------------
# Config + specs
# ---------------------------------------------------------------------


@dataclass
class ExitModelSpec:
    """
    Describes one ML exit model:
      - name: identifier & filename stem
      - target_col: column to predict
      - problem_type: 'regression' or 'binary'
      - pred_col: name of prediction column to add to df
    """

    name: str
    target_col: str
    problem_type: str  # 'regression' or 'binary'
    pred_col: str


@dataclass
class ExitMLConfig:
    """
    Training configuration for Pillar 2 exit ML models.
    """

    model_dir: Path

    # chrono split for train/validation (by open_time_ms)
    val_fraction: float = 0.2

    random_state: int = 42
    min_train_rows: int = 10_000

    # classification calibration
    calibrate_classifiers: bool = True
    calib_method: str = "auto"  # 'auto', 'platt', 'isotonic'

    # column for time axis
    col_time: str = "open_time_ms"


def default_exit_model_specs() -> List[ExitModelSpec]:
    """
    Default suite of exit models as per Pillar 2 design.
    """
    return [
        ExitModelSpec(
            name="tp_distance_reg",
            target_col="optimal_tp_distance",
            problem_type="regression",
            pred_col="predicted_tp_distance",
        ),
        ExitModelSpec(
            name="sl_distance_reg",
            target_col="optimal_sl_distance",
            problem_type="regression",
            pred_col="predicted_sl_distance",
        ),
        ExitModelSpec(
            name="tp_first_clf",
            target_col="exit_outcome",  # will be converted to binary
            problem_type="binary",
            pred_col="predicted_TP_first_probability",
        ),
        ExitModelSpec(
            name="exit_time_reg",
            target_col="expected_exit_time",
            problem_type="regression",
            pred_col="predicted_exit_time",
        ),
        ExitModelSpec(
            name="vol_impact_reg",
            target_col="exit_vol_short_trend",  # from mini-engines
            problem_type="regression",
            pred_col="predicted_volatility_effect",
        ),
    ]


# ---------------------------------------------------------------------
# Helpers: features, splits, calibration
# ---------------------------------------------------------------------


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024**2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def _build_feature_matrix(
    df: pd.DataFrame,
    cfg: ExitMLConfig,
    target_cols: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Builds the numeric feature matrix X from df by:

      - keeping only numeric columns (float, int, bool)
      - excluding:
          * target_cols
          * obvious label/leak columns (future MFE/MAE/etc)
          * time column

    Returns:
      X_df, feature_names
    """
    # Start from numeric columns only
    num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()

    # Columns that must NEVER be used as features (future labels)
    leak_cols = [
        "mfe_dir",
        "mae_dir",
        "expected_exit_time",
        "exit_vol_short_trend",
        "exit_vol_ratio_short_long",
        "optimal_tp_distance",
        "optimal_sl_distance",
        "optimal_risk_reward",
    ]

    # Time column is not a feature
    if cfg.col_time in num_cols:
        num_cols.remove(cfg.col_time)

    # Remove targets and leak columns (if present)
    forbidden = set(target_cols) | set(leak_cols)
    feature_cols = [c for c in num_cols if c not in forbidden]

    X_df = df[feature_cols].copy()
    _log_mem(X_df, "[ExitML] Feature matrix (numeric, leak-free)")
    logging.info("[ExitML] Using %d features", X_df.shape[1])
    return X_df, feature_cols


def _chrono_train_val_split(
    df: pd.DataFrame,
    cfg: ExitMLConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (train_idx, val_idx) as index arrays using chronological split
    on cfg.col_time with fraction cfg.val_fraction for validation.
    """
    if cfg.col_time not in df.columns:
        raise ValueError(f"[ExitML] Time column {cfg.col_time} not in df.")

    df_sorted = df.sort_values(cfg.col_time)
    idx_sorted = df_sorted.index.to_numpy()

    n = len(idx_sorted)
    n_val = int(max(1, n * cfg.val_fraction))
    split_at = n - n_val

    train_idx = idx_sorted[:split_at]
    val_idx = idx_sorted[split_at:]

    logging.info(
        "[ExitML] Chrono split on %s: n_train=%d, n_val=%d",
        cfg.col_time,
        len(train_idx),
        len(val_idx),
    )
    return train_idx, val_idx


def _fit_probability_calibrator(
    y_val: np.ndarray,
    p_raw: np.ndarray,
    method: str = "auto",
) -> Tuple[Dict, np.ndarray]:
    """
    Fit Platt (logistic) and/or Isotonic calibrators on validation data,
    returning:

      calibrator_info: dict with fields:
         - 'type': 'platt' or 'isotonic'
         - 'model': fitted sklearn model
      p_cal: calibrated probabilities on validation set

    y_val is expected to be in {0,1}.
    p_raw is the raw predicted probability for class 1.
    """
    mask = np.isfinite(y_val) & np.isfinite(p_raw)
    y = y_val[mask].astype(int)
    p = p_raw[mask].astype(float).clip(1e-6, 1 - 1e-6)

    if y.size < 1000:
        logging.warning(
            "[ExitML] Very few samples for calibration (n=%d). "
            "Returning identity calibration.",
            y.size,
        )
        return {"type": "identity", "model": None}, p_raw

    results = []

    # Platt (logistic on raw probability)
    if method in ("auto", "platt"):
        lr = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
        )
        lr.fit(p.reshape(-1, 1), y)
        p_platt = lr.predict_proba(p.reshape(-1, 1))[:, 1]
        brier_platt = brier_score_loss(y, p_platt)
        results.append(("platt", lr, brier_platt, p_platt))

    # Isotonic
    if method in ("auto", "isotonic"):
        iso = IsotonicRegression(
            out_of_bounds="clip",
        )
        p_iso = iso.fit_transform(p, y)
        brier_iso = brier_score_loss(y, p_iso)
        results.append(("isotonic", iso, brier_iso, p_iso))

    if not results:
        logging.warning(
            "[ExitML] No calibration methods selected; using identity."
        )
        return {"type": "identity", "model": None}, p_raw

    # Pick best by Brier
    best_type, best_model, best_brier, best_p = min(
        results, key=lambda x: x[2]
    )

    logging.info(
        "[ExitML] Calibration: selected %s (Brier=%.6f)",
        best_type,
        best_brier,
    )

    return {"type": best_type, "model": best_model}, _restore_full_probs(
        p_raw, best_p, mask
    )


def _restore_full_probs(
    p_raw: np.ndarray,
    p_cal_masked: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Utility: put calibrated probabilities back into an array of the same
    shape as p_raw, using mask to place calibrated values and leaving
    others as raw.
    """
    out = p_raw.copy()
    out[mask] = p_cal_masked
    return out


def _apply_calibrator(
    calibrator_info: Dict,
    p_raw: np.ndarray,
) -> np.ndarray:
    """
    Apply a previously fitted calibrator to raw probabilities.
    """
    ctype = calibrator_info.get("type")
    model = calibrator_info.get("model")

    if ctype is None or ctype == "identity" or model is None:
        return p_raw

    p = p_raw.astype(float).clip(1e-6, 1 - 1e-6)

    if ctype == "platt":
        return model.predict_proba(p.reshape(-1, 1))[:, 1]
    elif ctype == "isotonic":
        return model.transform(p)
    else:
        logging.warning("[ExitML] Unknown calibrator type: %s; using identity.", ctype)
        return p_raw


# ---------------------------------------------------------------------
# Training + prediction
# ---------------------------------------------------------------------


def _train_regressor(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: ExitMLConfig,
) -> HistGradientBoostingRegressor:
    """
    Train a regression model with HistGradientBoostingRegressor.
    Logs validation metrics.
    """
    reg = HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=8,
        learning_rate=0.05,
        max_iter=300,
        min_samples_leaf=100,
        random_state=cfg.random_state,
    )

    reg.fit(X_train, y_train)

    if X_val is not None and y_val is not None and len(y_val) > 0:
        y_pred = reg.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        rmse = mse**0.5
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        logging.info(
            "[ExitML][%s] Regression val metrics: RMSE=%.6f, MAE=%.6f, R2=%.4f",
            name,
            rmse,
            mae,
            r2,
        )

    return reg


def _train_classifier_with_calibration(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: ExitMLConfig,
) -> Dict:
    """
    Train a binary classifier (HistGradientBoostingClassifier) and
    calibrate its probabilities on a validation set.

    Returns a bundle dict:
      {
        "model": clf,
        "calibrator": calibrator_info (dict),
      }
    """
    clf = HistGradientBoostingClassifier(
        loss="log_loss",
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        min_samples_leaf=200,
        random_state=cfg.random_state,
    )

    clf.fit(X_train, y_train)

    # Raw validation probabilities
    p_val_raw = clf.predict_proba(X_val)[:, 1]

    # Fit calibrator (if enabled)
    if cfg.calibrate_classifiers:
        calibrator_info, p_val_cal = _fit_probability_calibrator(
            y_val=y_val,
            p_raw=p_val_raw,
            method=cfg.calib_method,
        )
    else:
        calibrator_info = {"type": "identity", "model": None}
        p_val_cal = p_val_raw

    # Validation metrics
    try:
        auc = roc_auc_score(y_val, p_val_cal)
    except ValueError:
        auc = float("nan")

    brier = brier_score_loss(y_val, p_val_cal)
    acc = accuracy_score(y_val, (p_val_cal >= 0.5).astype(int))

    logging.info(
        "[ExitML][%s] Classifier val metrics: AUC=%.4f, Brier=%.6f, Acc=%.4f",
        name,
        auc,
        brier,
        acc,
    )

    return {"model": clf, "calibrator": calibrator_info}


def train_and_predict_exit_models(
    df: pd.DataFrame,
    specs: List[ExitModelSpec],
    cfg: ExitMLConfig,
) -> pd.DataFrame:
    """
    Train all exit models and write their predictions back into `df`.

    Steps:
      1. Build numeric feature matrix X (leak-free).
      2. Chronological train/validation split.
      3. For each model spec:
           - build y (label)
           - train regressor or classifier
           - evaluate on validation set
           - save model bundle to disk
           - compute predictions for ALL rows in df
      4. Return df with new prediction columns attached.
    """
    if not len(df):
        raise ValueError("[ExitML] Empty dataframe; nothing to train.")

    # Build common feature matrix
    all_target_cols = list({spec.target_col for spec in specs})
    X_df, feature_cols = _build_feature_matrix(df, cfg, target_cols=all_target_cols)

    # Chrono split
    train_idx, val_idx = _chrono_train_val_split(df, cfg)

    # Prepare base outputs
    df_out = df.copy()

    for spec in specs:
        logging.info(
            "[ExitML] === Training model '%s' on target '%s' (%s) ===",
            spec.name,
            spec.target_col,
            spec.problem_type,
        )

        if spec.target_col not in df_out.columns:
            logging.warning(
                "[ExitML][%s] Target column %s missing; skipping model.",
                spec.name,
                spec.target_col,
            )
            df_out[spec.pred_col] = np.nan
            continue

        # ---------- FIXED LABEL HANDLING ----------
        if spec.problem_type == "binary" and spec.target_col == "exit_outcome":
            # exit_outcome is string; convert to {0,1}
            y_exit = df_out["exit_outcome"].astype("object")
            # mask: just require non-missing labels
            label_mask = y_exit.notna().to_numpy()
            y_all = (y_exit == "tp_first").to_numpy().astype(float)
        else:
            # numeric targets: cast to float then mask via isfinite
            y_all = df_out[spec.target_col].astype(float).to_numpy()
            label_mask = np.isfinite(y_all)
        # -------------------------------------------

        # Apply mask within train/val sets
        row_positions = np.arange(len(df_out))
        train_mask = label_mask & np.isin(row_positions, train_idx)
        val_mask = label_mask & np.isin(row_positions, val_idx)

        n_train = int(train_mask.sum())
        n_val = int(val_mask.sum())

        if n_train < cfg.min_train_rows:
            logging.warning(
                "[ExitML][%s] Not enough train rows after masking (n_train=%d). "
                "Filling predictions with NaN.",
                spec.name,
                n_train,
            )
            df_out[spec.pred_col] = np.nan
            continue

        if spec.problem_type == "binary":
            pos_count = int(y_all[train_mask].sum())
            neg_count = int((1 - y_all[train_mask]).sum())
            if pos_count < 100 or neg_count < 100:
                logging.warning(
                    "[ExitML][%s] Very imbalanced / small classes "
                    "(pos=%d, neg=%d); training anyway but watch metrics.",
                    spec.name,
                    pos_count,
                    neg_count,
                )

        # Subset matrices
        X_train = X_df.loc[train_mask].to_numpy(dtype=np.float32)
        y_train = y_all[train_mask]

        X_val = X_df.loc[val_mask].to_numpy(dtype=np.float32)
        y_val = y_all[val_mask]

        # Train appropriate model
        if spec.problem_type == "regression":
            model = _train_regressor(
                name=spec.name,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                cfg=cfg,
            )

            # Save model bundle
            bundle = {
                "type": "regression",
                "name": spec.name,
                "target_col": spec.target_col,
                "pred_col": spec.pred_col,
                "features": feature_cols,
                "model": model,
            }
            cfg.model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(bundle, cfg.model_dir / f"{spec.name}.joblib")
            logging.info(
                "[ExitML][%s] Saved regression model -> %s",
                spec.name,
                cfg.model_dir / f"{spec.name}.joblib",
            )

            # Predictions for all rows
            X_all = X_df.to_numpy(dtype=np.float32)
            y_pred_all = model.predict(X_all)
            df_out[spec.pred_col] = y_pred_all.astype("float32")

        elif spec.problem_type == "binary":
            bundle = _train_classifier_with_calibration(
                name=spec.name,
                X_train=X_train,
                y_train=y_train.astype(int),
                X_val=X_val,
                y_val=y_val.astype(int),
                cfg=cfg,
            )
            clf = bundle["model"]
            calibrator_info = bundle["calibrator"]

            # Persist bundle (model + calibrator)
            save_bundle = {
                "type": "binary",
                "name": spec.name,
                "target_col": spec.target_col,
                "pred_col": spec.pred_col,
                "features": feature_cols,
                "model": clf,
                "calibrator": calibrator_info,
            }
            cfg.model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(save_bundle, cfg.model_dir / f"{spec.name}.joblib")
            logging.info(
                "[ExitML][%s] Saved classifier bundle -> %s",
                spec.name,
                cfg.model_dir / f"{spec.name}.joblib",
            )

            # Predict calibrated probabilities for all rows
            X_all = X_df.to_numpy(dtype=np.float32)
            p_raw_all = clf.predict_proba(X_all)[:, 1]
            p_cal_all = _apply_calibrator(calibrator_info, p_raw_all)
            df_out[spec.pred_col] = p_cal_all.astype("float32")

        else:
            logging.error(
                "[ExitML][%s] Unknown problem_type=%s; skipping.",
                spec.name,
                spec.problem_type,
            )
            df_out[spec.pred_col] = np.nan

    _log_mem(df_out, "[ExitML] Final df with ML exit predictions")
    return df_out