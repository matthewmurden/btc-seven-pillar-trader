"""Master orchestrator for all mini engines."""

from __future__ import annotations

import pandas as pd

from .momentum_engine import MomentumEngine
from .trend_engine import TrendEngine
from .volatility_engine import VolatilityEngine
from .reversion_engine import ReversionEngine
from .orderflow_engine import OrderflowEngine
from .structure_engine import StructureEngine
from .pattern_engine import PatternEngine
from .regime_engine import RegimeEngine


class MiniEnginesMaster:
    """Run all deterministic mini engines sequentially."""

    def __init__(self) -> None:
        self.engines = [
            MomentumEngine(),
            TrendEngine(),
            VolatilityEngine(),
            ReversionEngine(),
            OrderflowEngine(),
            StructureEngine(),
            PatternEngine(),
            RegimeEngine(),
        ]

    def run_all(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        output = df.copy()
        for engine in self.engines:
            output = engine.transform(output)
            if verbose:
                print(f"Applied mini engine: {engine.__class__.__name__}")
        return output
