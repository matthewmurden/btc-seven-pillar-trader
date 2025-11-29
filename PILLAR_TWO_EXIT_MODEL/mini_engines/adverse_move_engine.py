"""Adverse-Move deterministic engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


class AdverseMoveEngine:
    """Scores MAE risk based on orderflow and liquidity signals."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        delta_pressure = np.tanh(out.get("feature_exit_delta_into_entry", pd.Series(0, index=out.index)))
        absorption = out.get("feature_exit_absorption_score", pd.Series(0, index=out.index))
        liquidity_below = out.get("feature_exit_distance_to_liquidity", pd.Series(0, index=out.index))
        wick_dominance = out.get("feature_exit_wick_imbalance", pd.Series(0, index=out.index))

        score = (
            0.4 * delta_pressure.abs()
            + 0.2 * absorption.abs()
            + 0.2 * np.tanh(liquidity_below.fillna(0))
            + 0.2 * wick_dominance.fillna(0)
        )
        out["mae_risk_score"] = np.clip(score, 0, 1)
        return out
