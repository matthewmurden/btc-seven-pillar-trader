"""
Calibration layer for exit probabilities and scores.

Supports isotonic regression, Platt scaling (logistic regression), and temperature scaling.
Selects the best method per target using Brier score, with logloss and ECE as tie-breakers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
CALIB_DIR = BASE_DIR / "data" / "calibration"
CALIB_FILES = {
    "tp": CALIB_DIR / "tp_calibrator.pkl",
    "sl": CALIB_DIR / "sl_calibrator.pkl",
    "exit_quality": CALIB_DIR / "exit_quality_calibrator.pkl",
}

MIN_PROB = 1e-6
MAX_PROB = 1 - 1e-6


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _clip_prob(p: np.ndarray) -> np.ndarray:
    return np.clip(p, MIN_PROB, MAX_PROB)


def compute_brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = _clip_prob(y_pred)
    return float(brier_score_loss(y_true, y_pred))


def compute_logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = _clip_prob(y_pred)
    return float(log_loss(y_true, y_pred))


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error for binary targets."""
    y_prob = _clip_prob(y_prob)
    y_true = np.asarray(y_true, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if not np.any(mask):
            continue
        acc = np.mean(y_true[mask])
        conf = np.mean(y_prob[mask])
        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


# ---------------------------------------------------------------------------
# Calibration methods
# ---------------------------------------------------------------------------
def _calibrate_isotonic(y_true: np.ndarray, y_prob: np.ndarray):
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(y_prob, y_true)
    calibrated = _clip_prob(model.predict(y_prob))
    return model, calibrated


def _calibrate_platt(y_true: np.ndarray, y_prob: np.ndarray):
    model = LogisticRegression(solver="lbfgs")
    model.fit(y_prob.reshape(-1, 1), y_true)
    calibrated = _clip_prob(model.predict_proba(y_prob.reshape(-1, 1))[:, 1])
    return model, calibrated


def _temperature_nll(y_true: np.ndarray, logits: np.ndarray, T: float) -> float:
    scaled = logits / T
    probs = 1.0 / (1.0 + np.exp(-scaled))
    probs = _clip_prob(probs)
    return float(log_loss(y_true, probs))


def _calibrate_temperature(y_true: np.ndarray, y_prob: np.ndarray, lr: float = 0.01, steps: int = 500) -> Tuple[float, np.ndarray]:
    probs = _clip_prob(y_prob)
    logits = np.log(probs / (1 - probs))
    T = 1.0
    for _ in range(steps):
        scaled = logits / T
        exp_term = np.exp(-scaled)
        prob_scaled = 1.0 / (1.0 + exp_term)
        prob_scaled = _clip_prob(prob_scaled)
        grad = np.mean((prob_scaled - y_true) * (-scaled / T))
        T -= lr * grad
        T = float(np.clip(T, 0.5, 5.0))
    scaled_logits = logits / T
    calibrated = _clip_prob(1.0 / (1.0 + np.exp(-scaled_logits)))
    return T, calibrated


def _evaluate_methods(y_true: np.ndarray, candidates: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    scores: Dict[str, Dict[str, float]] = {}
    for name, pred in candidates.items():
        scores[name] = {
            "brier": compute_brier(y_true, pred),
            "logloss": compute_logloss(y_true, pred),
            "ece": compute_ece(y_true, pred),
        }
    return scores


def _select_best(scores: Dict[str, Dict[str, float]]) -> str:
    # Lower is better
    methods = list(scores.keys())
    briers = np.array([scores[m]["brier"] for m in methods])
    best_brier = briers.min()
    brier_tol = 1e-6
    candidates = [m for m in methods if abs(scores[m]["brier"] - best_brier) <= brier_tol]
    if len(candidates) == 1:
        return candidates[0]
    # tie-breaker logloss
    loglosses = np.array([scores[m]["logloss"] for m in candidates])
    best_log = loglosses.min()
    log_tol = 1e-6
    candidates_log = [m for m in candidates if abs(scores[m]["logloss"] - best_log) <= log_tol]
    if len(candidates_log) == 1:
        return candidates_log[0]
    # tie-breaker ece
    eces = np.array([scores[m]["ece"] for m in candidates_log])
    idx = np.argmin(eces)
    return candidates_log[idx]


def _train_single(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[str, Dict[str, object], np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = _clip_prob(np.asarray(y_prob, dtype=np.float64))

    # Guard constant probabilities
    if np.all(y_prob == y_prob[0]):
        logger.warning("Constant probabilities detected; skipping calibration.")
        return "identity", {"method": "identity", "model": None, "temperature": 1.0}, y_prob

    candidates: Dict[str, np.ndarray] = {}
    models: Dict[str, Dict[str, object]] = {}

    # Isotonic
    try:
        iso_model, iso_pred = _calibrate_isotonic(y_true, y_prob)
        candidates["isotonic"] = iso_pred
        models["isotonic"] = {"method": "isotonic", "model": iso_model, "temperature": None}
    except Exception as exc:
        logger.warning("Isotonic calibration failed: %s", exc)

    # Platt
    try:
        platt_model, platt_pred = _calibrate_platt(y_true, y_prob)
        candidates["platt"] = platt_pred
        models["platt"] = {"method": "platt", "model": platt_model, "temperature": None}
    except Exception as exc:
        logger.warning("Platt calibration failed: %s", exc)

    # Temperature scaling
    try:
        temp, temp_pred = _calibrate_temperature(y_true, y_prob)
        candidates["temperature"] = temp_pred
        models["temperature"] = {"method": "temperature", "model": None, "temperature": temp}
        logger.info("Learned temperature: %.4f", temp)
    except Exception as exc:
        logger.warning("Temperature calibration failed: %s", exc)

    if not candidates:
        logger.warning("No calibration methods succeeded; returning identity.")
        return "identity", {"method": "identity", "model": None, "temperature": 1.0}, y_prob

    scores = _evaluate_methods(y_true, candidates)
    for name, sc in scores.items():
        logger.info("Calibration %s -> Brier=%.6f LogLoss=%.6f ECE=%.6f", name, sc["brier"], sc["logloss"], sc["ece"])

    best = _select_best(scores)
    logger.info("Selected calibration method: %s", best)
    return best, models[best], candidates[best]


# ---------------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------------
def train_calibration_layer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Train calibration for tp_first_proba, sl_first_proba, exit_quality_score.
    Requires columns: tp_first_proba, sl_first_proba, exit_quality_score, tp_first, sl_first, final_exit_label.
    Saves calibrators to disk.
    Returns df with calibrated columns.
    """
    required = [
        "tp_first_proba",
        "sl_first_proba",
        "exit_quality_score",
        "tp_first",
        "sl_first",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for calibration training: {missing}")

    if "final_exit_label" in df.columns:
        final_label = df["final_exit_label"].to_numpy()
    else:
        logger.warning("final_exit_label missing; using tp_first as proxy.")
        final_label = df["tp_first"].to_numpy()

    CALIB_DIR.mkdir(parents=True, exist_ok=True)

    # TP
    best_tp, tp_model, tp_calibrated = _train_single(df["tp_first"].to_numpy(), df["tp_first_proba"].to_numpy())
    joblib.dump(tp_model, CALIB_FILES["tp"])

    # SL
    best_sl, sl_model, sl_calibrated = _train_single(df["sl_first"].to_numpy(), df["sl_first_proba"].to_numpy())
    joblib.dump(sl_model, CALIB_FILES["sl"])

    # Exit quality (treat final_exit_label as binary)
    best_exit, exit_model, exit_calibrated = _train_single(final_label, df["exit_quality_score"].to_numpy())
    joblib.dump(exit_model, CALIB_FILES["exit_quality"])

    logger.info("Saved calibrators: tp=%s sl=%s exit=%s", best_tp, best_sl, best_exit)

    out = df.copy()
    out["tp_first_calibrated"] = tp_calibrated.astype(np.float32)
    out["sl_first_calibrated"] = sl_calibrated.astype(np.float32)
    out["exit_quality_calibrated"] = exit_calibrated.astype(np.float32)
    return out


def _apply_single(prob: np.ndarray, calibrator: Dict[str, object]) -> np.ndarray:
    prob = _clip_prob(prob)
    method = calibrator.get("method")
    model = calibrator.get("model")
    temp = calibrator.get("temperature")

    if method == "isotonic":
        return _clip_prob(model.predict(prob))
    if method == "platt":
        return _clip_prob(model.predict_proba(prob.reshape(-1, 1))[:, 1])
    if method == "temperature":
        logits = np.log(prob / (1 - prob))
        scaled = logits / float(temp if temp is not None else 1.0)
        return _clip_prob(1.0 / (1.0 + np.exp(-scaled)))
    return prob


def apply_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply stored calibrators to probabilities/scores.
    Requires columns: tp_first_proba, sl_first_proba, exit_quality_score.
    """
    for key, path in CALIB_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Calibrator missing: {path}. Train calibration first.")

    out = df.copy()
    tp_cal = joblib.load(CALIB_FILES["tp"])
    sl_cal = joblib.load(CALIB_FILES["sl"])
    exit_cal = joblib.load(CALIB_FILES["exit_quality"])

    out["tp_first_calibrated"] = _apply_single(out["tp_first_proba"].to_numpy(dtype=np.float64), tp_cal).astype(np.float32)
    out["sl_first_calibrated"] = _apply_single(out["sl_first_proba"].to_numpy(dtype=np.float64), sl_cal).astype(np.float32)
    out["exit_quality_calibrated"] = _apply_single(out["exit_quality_score"].to_numpy(dtype=np.float64), exit_cal).astype(np.float32)

    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sample_path = BASE_DIR / "data" / "processed" / "exit_master.parquet"
    df = pd.read_parquet(sample_path)
    train_calibration_layer(df)
