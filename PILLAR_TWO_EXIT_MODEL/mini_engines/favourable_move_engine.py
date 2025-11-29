"""Favourable-Move deterministic engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


class FavourableMoveEngine:
    """Scores MFE potential using trend/impulse/volume/volatility signals."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        trend_cont = np.tanh(out.get("feature_exit_trend_stretch", pd.Series(0, index=out.index)))
        impulse = np.tanh(out.get("feature_reversion_impulse", pd.Series(0, index=out.index)))
        volume_burst = np.tanh(out.get("feature_exit_volume_expansion", pd.Series(0, index=out.index)))
        vol_state = 1 - out.get("feature_exit_vol_compression", pd.Series(0, index=out.index)).astype(float)

        score = (
            0.35 * trend_cont.fillna(0)
            + 0.25 * impulse.fillna(0)
            + 0.25 * volume_burst.fillna(0)
            + 0.15 * vol_state.fillna(0)
        )
        out["mfe_potential_score"] = np.clip(score, 0, 1)
        return out
