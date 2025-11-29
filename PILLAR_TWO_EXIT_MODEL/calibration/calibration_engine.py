"""Calibration layer for exit signals (Part 9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ExitCalibrator:
    """Calibrates TP/SL probabilities and exit quality via temperature scaling."""

    def __init__(self) -> None:
        self.iso_tp: IsotonicRegression | None = None
        self.platt_sl: LogisticRegression | None = None
        self.temperature_: float | None = None

    def fit(self, df: pd.DataFrame) -> None:
        required = [
            "tp_first_pred",
            "sl_first_pred",
            "exit_quality_score",
            "tp_first_label",
            "sl_first_label",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for calibration: {missing}")

        x_tp = df["tp_first_pred"].to_numpy()
        y_tp = df["tp_first_label"].to_numpy()
        self.iso_tp = IsotonicRegression(out_of_bounds="clip")
        self.iso_tp.fit(x_tp, y_tp)

        x_sl = df["sl_first_pred"].to_numpy().reshape(-1, 1)
        y_sl = df["sl_first_label"].to_numpy()
        self.platt_sl = LogisticRegression(max_iter=200)
        self.platt_sl.fit(x_sl, y_sl)

        # temperature scaling: grid search to minimise squared error to itself (self-calibration, deterministic)
        raw = df["exit_quality_score"].to_numpy()
        temps = np.linspace(0.5, 3.0, 11)
        best_t = 1.0
        best_loss = np.inf
        for T in temps:
            scaled = self._temperature_softmax(raw, T)
            loss = np.mean((scaled - raw) ** 2)
            if loss < best_loss:
                best_loss = loss
                best_t = T
        self.temperature_ = best_t

    @staticmethod
    def _temperature_softmax(raw: np.ndarray, T: float) -> np.ndarray:
        logits = raw / T
        other = (1 - raw) / T
        exp_a = np.exp(logits)
        exp_b = np.exp(other)
        return exp_a / (exp_a + exp_b)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.iso_tp is None or self.platt_sl is None or self.temperature_ is None:
            raise RuntimeError("Calibrator not fitted.")

        missing_pred = ["tp_first_pred", "sl_first_pred", "exit_quality_score"]
        missing = [c for c in missing_pred if c not in df.columns]
        if missing:
            raise ValueError(f"Missing prediction columns for calibration: {missing}")

        out = df.copy()

        out["tp_first_calibrated"] = self.iso_tp.predict(out["tp_first_pred"])
        out["sl_first_calibrated"] = self.platt_sl.predict_proba(out["sl_first_pred"].to_numpy().reshape(-1, 1))[:, 1]
        out["exit_quality_calibrated"] = self._temperature_softmax(
            out["exit_quality_score"].to_numpy(), self.temperature_
        )

        for col in ["tp_first_calibrated", "sl_first_calibrated", "exit_quality_calibrated"]:
            vals = out[col]
            if vals.isna().any() or not ((vals >= 0).all() and (vals <= 1).all()):
                raise ValueError(f"Calibrated column {col} out of bounds or NaN")

        return out
