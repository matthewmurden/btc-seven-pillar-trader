"""Master orchestrator for exit mini engines (Part 3)."""

from __future__ import annotations

import pandas as pd

from .volatility_excursion_engine import VolatilityExcursionEngine
from .adverse_move_engine import AdverseMoveEngine
from .favourable_move_engine import FavourableMoveEngine
from .path_shape_engine import PathShapeEngine
from .regime_exit_engine import RegimeExitEngine


class ExitMiniEnginesMaster:
    """Run all exit-specific deterministic engines sequentially."""

    def __init__(self) -> None:
        self.engines = [
            VolatilityExcursionEngine(),
            AdverseMoveEngine(),
            FavourableMoveEngine(),
            PathShapeEngine(),
            RegimeExitEngine(),
        ]

    def run_all(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        output = df.copy()
        for engine in self.engines:
            output = engine.transform(output)
            if verbose:
                print(f"Applied exit mini engine: {engine.__class__.__name__}")
        return output
