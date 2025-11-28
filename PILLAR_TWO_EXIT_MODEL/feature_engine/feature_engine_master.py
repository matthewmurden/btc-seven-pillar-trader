"""Master orchestrator for all feature groups."""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from .momentum_features import MomentumFeatures
from .trend_features import TrendFeatures
from .volatility_features import VolatilityFeatures
from .reversion_features import ReversionFeatures
from .orderflow_features import OrderflowFeatures
from .structure_features import StructureFeatures
from .pattern_features import PatternFeatures
from .regime_features import RegimeFeatures


class FeatureEngineMaster:
    """Run all Pillar Two feature groups sequentially."""

    def __init__(self) -> None:
        self.engines: List[Tuple[str, object]] = [
            ("momentum", MomentumFeatures()),
            ("trend", TrendFeatures()),
            ("volatility", VolatilityFeatures()),
            ("reversion", ReversionFeatures()),
            ("orderflow", OrderflowFeatures()),
            ("structure", StructureFeatures()),
            ("pattern", PatternFeatures()),
            ("regime", RegimeFeatures()),
        ]
        self.group_columns: Dict[str, List[str]] = {}

    def run_all(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        output = df.copy()
        self.group_columns = {}

        for name, engine in self.engines:
            before_cols = set(output.columns)
            output = engine.transform(output)
            after_cols = set(output.columns)
            added = sorted(col for col in after_cols - before_cols if col.startswith("feature_"))
            self.group_columns[name] = added
            if verbose:
                print(f"Applied feature group '{name}': {len(added)} features")

        return output
