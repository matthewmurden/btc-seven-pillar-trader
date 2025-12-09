"""
Pillar 1 — Calibration Layer

Takes the raw continuous direction signal from the master aggregator:
    raw_entry_confidence ∈ [-1, 1]

and fits a 1D calibration mapping:
    raw_entry_confidence  →  entry_prob_up ∈ [0, 1]

using:
  - Platt scaling (logistic regression),
  - Isotonic regression, or
  - AUTO: train both on a chronological split and choose the
    better one by Brier score on the validation window.

Calibration target (by default):
  - y_dir_5_10 ∈ {-1, 0, 1} from Pillar 1 labels
  - We only use rows where y_dir_5_10 ∈ {-1, 1}.
  - Binary target: 1 if label == 1 (up), 0 if label == -1 (down).

Outputs:
  - entry_prob_up
  - entry_prob_down
  - entry_calibration_method (string: "platt" or "isotonic")
  - entry_prob_source_label (e.g. "y_dir_5_10")
"""

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
# from sklearn.calibration import CalibrationDisplay  # no longer needed
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


CalibrationMethod = Literal["platt", "isotonic", "auto"]


# ----------------------------------------------------------------------
# Config and Calibrator wrapper
# ----------------------------------------------------------------------
@dataclass
class CalibrationConfig:
    target_label_col: str = "y_dir_5_10"
    method: CalibrationMethod = "auto"  # "platt", "isotonic", or "auto"
    val_fraction: float = 0.2  # chronological split for calibration eval
    min_abs_raw_for_train: float = 0.05  # ignore tiny scores near 0
    model_dir: Path = Path("models/pillar1_calibration")
    random_state: int = 42
    # Safety: require at least this many up and down examples in training
    min_class_count: int = 200


@dataclass
class Pillar1Calibrator:
    method: str  # "platt" or "isotonic"
    target_label_col: str
    platt_model: Optional[LogisticRegression] = None
    iso_model: Optional[IsotonicRegression] = None

    def predict_prob_up(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        Predict P(up) given raw_entry_confidence.

        raw_scores: array-like of shape (n,) or (n,1)
        """
        x = np.asarray(raw_scores, dtype=float).reshape(-1, 1)

        if self.method == "platt":
            if self.platt_model is None:
                raise RuntimeError("Platt model is not set in calibrator.")
            # LogisticRegression returns (n,2) probs; we want P(y=1)
            proba = self.platt_model.predict_proba(x)[:, 1]
        elif self.method == "isotonic":
            if self.iso_model is None:
                raise RuntimeError("Isotonic model is not set in calibrator.")
            # IsotonicRegression takes 1D array
            proba = self.iso_model.predict(x.ravel())
        else:
            raise ValueError(f"Unknown calibrator method: {self.method}")

        # Clip numerically into [0,1] for safety
        return np.clip(proba, 0.0, 1.0)


# ----------------------------------------------------------------------
# Core helpers
# ----------------------------------------------------------------------
def _build_calibration_dataset(
    df: pd.DataFrame,
    cfg: CalibrationConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract (raw_entry_confidence, binary_target) pairs for calibration.

    - Use only rows where target_label_col ∈ {-1, 1}.
    - target 1 corresponds to label == 1 (up), 0 to label == -1 (down).
    - Optionally filter out cases where |raw_entry_confidence| is too small.
    """
    if "raw_entry_confidence" not in df.columns:
        raise ValueError(
            "raw_entry_confidence not found in DataFrame. "
            "Run the master aggregator step before calibration."
        )

    if cfg.target_label_col not in df.columns:
        raise ValueError(
            f"Target label column '{cfg.target_label_col}' not found in DataFrame. "
            "Ensure labels_pillar1.parquet has been merged."
        )

    raw = df["raw_entry_confidence"].astype(float)
    label = df[cfg.target_label_col]

    mask = label.isin([-1, 1]) & raw.notna()
    if cfg.min_abs_raw_for_train > 0:
        mask &= (raw.abs() >= cfg.min_abs_raw_for_train)

    x = raw[mask].to_numpy().reshape(-1, 1)
    y = (label[mask].to_numpy().astype(int) == 1).astype(int)

    n_total = len(df)
    n_used = len(x)
    logging.info(
        "[Calib] Using %d/%d rows for calibration after filtering "
        "(abs(raw) >= %.3f, label ∈ {-1,1})",
        n_used,
        n_total,
        cfg.min_abs_raw_for_train,
    )

    if n_used < cfg.min_class_count * 2:
        raise ValueError(
            f"[Calib] Not enough calibration samples after filtering: {n_used}. "
            f"Increase history or reduce min_abs_raw_for_train={cfg.min_abs_raw_for_train}."
        )

    # Check class balance
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    logging.info("[Calib] Class counts (up=1 / down=0): pos=%d, neg=%d", n_pos, n_neg)
    if n_pos < cfg.min_class_count or n_neg < cfg.min_class_count:
        raise ValueError(
            f"[Calib] Insufficient class balance for calibration: pos={n_pos}, neg={n_neg}. "
            f"Need at least {cfg.min_class_count} of each."
        )

    return x, y


def _chronological_split(
    x: np.ndarray,
    y: np.ndarray,
    val_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0 < val_fraction < 1):
        raise ValueError("val_fraction must be in (0,1).")
    n = len(x)
    split_idx = int(n * (1 - val_fraction))
    if split_idx <= 0 or split_idx >= n:
        raise ValueError("val_fraction produces empty train/val set.")
    return x[:split_idx], x[split_idx:], y[:split_idx], y[split_idx:]


def _fit_platt(
    x_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
) -> LogisticRegression:
    """
    1D logistic regression (Platt scaling).
    """
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        penalty="l2",
        random_state=random_state,
        max_iter=1000,
    )
    model.fit(x_train, y_train)
    return model


def _fit_isotonic(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> IsotonicRegression:
    """
    Isotonic regression with clipping to [0, 1].
    """
    iso = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )
    # IsotonicRegression expects 1D x
    iso.fit(x_train.ravel(), y_train)
    return iso


def _evaluate_brier(
    y_true: np.ndarray,
    prob_up: np.ndarray,
    label: str,
) -> float:
    score = brier_score_loss(y_true, prob_up)
    logging.info("[Calib] Brier score (%s): %.6f", label, score)
    return score


def _log_calibration_curve(
    y_true: np.ndarray,
    prob_up: np.ndarray,
    label: str,
    n_bins: int = 10,
) -> None:
    """
    Log a coarse calibration curve: predicted vs empirical probability in bins.

    We avoid using sklearn.CalibrationDisplay here so this stays
    headless-safe and doesn't depend on matplotlib plotting kwargs.
    """
    y_true = np.asarray(y_true).astype(int)
    prob_up = np.asarray(prob_up, dtype=float)

    # Guard: if something degenerate happens, bail gracefully
    if y_true.size == 0:
        logging.warning("[Calib] Empty y_true in calibration curve for '%s'; skipping.", label)
        return

    # Define uniform probability bins in [0, 1]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(prob_up, bins) - 1  # bin_indices in [0, n_bins-1]

    logging.info("[Calib] Calibration curve (%s):", label)
    for b in range(n_bins):
        mask = bin_indices == b
        if not np.any(mask):
            continue

        bin_pred = prob_up[mask].mean()
        bin_obs = y_true[mask].mean()  # fraction of positives
        logging.info("  bin %2d [%.2f, %.2f): pred=%.3f, obs=%.3f, n=%d",
                     b, bins[b], bins[b + 1], bin_pred, bin_obs, mask.sum())

# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def fit_calibrator(
    df: pd.DataFrame,
    cfg: Optional[CalibrationConfig] = None,
) -> Pillar1Calibrator:
    """
    Fit a Pillar 1 calibrator from a DataFrame that has both:
      - raw_entry_confidence
      - direction label column (e.g. y_dir_5_10)
    """
    if cfg is None:
        cfg = CalibrationConfig()

    logging.info("[Calib] Fitting calibrator with config: %s", cfg)

    x, y = _build_calibration_dataset(df, cfg)
    x_train, x_val, y_train, y_val = _chronological_split(x, y, cfg.val_fraction)

    methods_to_try = [cfg.method] if cfg.method in ("platt", "isotonic") else ["platt", "isotonic"]
    results: Dict[str, Dict] = {}

    for method in methods_to_try:
        logging.info("[Calib] Training %s calibrator...", method.upper())
        if method == "platt":
            platt = _fit_platt(x_train, y_train, random_state=cfg.random_state)
            prob_up_val = platt.predict_proba(x_val)[:, 1]
            brier = _evaluate_brier(y_val, prob_up_val, label="platt")
            _log_calibration_curve(y_val, prob_up_val, label="platt")
            results["platt"] = dict(model=platt, brier=brier)
        elif method == "isotonic":
            iso = _fit_isotonic(x_train, y_train)
            prob_up_val = iso.predict(x_val.ravel())
            brier = _evaluate_brier(y_val, prob_up_val, label="isotonic")
            _log_calibration_curve(y_val, prob_up_val, label="isotonic")
            results["isotonic"] = dict(model=iso, brier=brier)
        else:
            raise ValueError(f"Unknown calibration method '{method}'")

    # Choose best method by Brier score
    if cfg.method == "auto":
        best_method = min(results.keys(), key=lambda m: results[m]["brier"])
        logging.info(
            "[Calib] AUTO selected '%s' (Brier=%.6f) over %s",
            best_method,
            results[best_method]["brier"],
            ", ".join(f"{m}(Brier={results[m]['brier']:.6f})" for m in results if m != best_method),
        )
    else:
        best_method = cfg.method

    best_model = results[best_method]["model"]
    if best_method == "platt":
        calibrator = Pillar1Calibrator(
            method="platt",
            target_label_col=cfg.target_label_col,
            platt_model=best_model,
            iso_model=None,
        )
    else:
        calibrator = Pillar1Calibrator(
            method="isotonic",
            target_label_col=cfg.target_label_col,
            platt_model=None,
            iso_model=best_model,
        )

    return calibrator


def save_calibrator(
    calibrator: Pillar1Calibrator,
    cfg: CalibrationConfig,
) -> Path:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.model_dir / "pillar1_calibrator.joblib"
    payload = {
        "calibrator": calibrator,
        "config": asdict(cfg),
    }
    joblib.dump(payload, path)
    logging.info("[Calib] Saved calibrator -> %s", path)
    return path


def load_calibrator(
    cfg: CalibrationConfig,
) -> Pillar1Calibrator:
    path = cfg.model_dir / "pillar1_calibrator.joblib"
    if not path.exists():
        raise FileNotFoundError(f"[Calib] Calibrator file not found: {path}")
    payload = joblib.load(path)
    calibrator: Pillar1Calibrator = payload["calibrator"]
    return calibrator


def apply_calibration_to_df(
    df: pd.DataFrame,
    calibrator: Pillar1Calibrator,
    cfg: CalibrationConfig,
) -> pd.DataFrame:
    """
    Apply a fitted calibrator to a DataFrame that has raw_entry_confidence.

    Adds:
      - entry_prob_up
      - entry_prob_down
      - entry_calibration_method
      - entry_prob_source_label
    """
    if "raw_entry_confidence" not in df.columns:
        raise ValueError("raw_entry_confidence not found in DataFrame.")

    raw = df["raw_entry_confidence"].astype(float).to_numpy()
    prob_up = calibrator.predict_prob_up(raw)
    prob_down = 1.0 - prob_up

    out = df.copy()
    out["entry_prob_up"] = prob_up
    out["entry_prob_down"] = prob_down
    out["entry_calibration_method"] = calibrator.method
    out["entry_prob_source_label"] = calibrator.target_label_col

    return out