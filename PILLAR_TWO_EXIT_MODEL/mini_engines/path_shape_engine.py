"""Path-Shape deterministic engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


class PathShapeEngine:
    """Scores path entropy using clustering and entropy measures."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        clustering = out.get("feature_exit_bar_clustering", pd.Series(0, index=out.index))
        entropy = out.get("feature_exit_micro_entropy", pd.Series(0, index=out.index))
        smoothness = 1 - clustering.fillna(0).abs()

        norm_entropy = entropy / (entropy.rolling(200, min_periods=10).max() + 1e-9)

        score = (
            0.5 * (1 - norm_entropy.fillna(0))
            + 0.3 * clustering.fillna(0)
            + 0.2 * smoothness.fillna(0)
        )
        out["path_entropy_score"] = np.clip(score, 0, 1)
        return out
