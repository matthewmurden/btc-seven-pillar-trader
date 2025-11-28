"""Structure mini engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


class StructureEngine:
    """Compute structure_score from market structure primitives."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        swing_high = out.get("feature_structure_swing_high", pd.Series(0, index=out.index))
        swing_low = out.get("feature_structure_swing_low", pd.Series(0, index=out.index))
        bos_up = out.get("feature_structure_bos_up", pd.Series(0, index=out.index))
        bos_down = out.get("feature_structure_bos_down", pd.Series(0, index=out.index))
        choch_up = out.get("feature_structure_choch_up", pd.Series(0, index=out.index))
        choch_down = out.get("feature_structure_choch_down", pd.Series(0, index=out.index))

        structure_alignment = (
            bos_up.astype(float)
            - bos_down.astype(float)
            + (choch_up.astype(float) * 2)
            - (choch_down.astype(float) * 2)
        )

        liquidity_bias = swing_low.astype(float) - swing_high.astype(float)

        components = [
            structure_alignment,
            liquidity_bias,
            out.get("feature_structure_signal", pd.Series(0, index=out.index)),
        ]

        combined = sum(comp.fillna(0) for comp in components) / len(components)
        out["structure_score"] = np.tanh(combined)
        return out
