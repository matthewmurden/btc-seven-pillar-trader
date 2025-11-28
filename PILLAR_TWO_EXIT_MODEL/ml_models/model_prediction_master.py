"""Master predictor to run all Pillar Two models."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .model_1m import predict as pred_1m  # noqa: E402
from .model_5m import predict as pred_5m  # noqa: E402
from .model_15m import predict as pred_15m  # noqa: E402
from .breakout_model import predict as pred_breakout  # noqa: E402
from .reversion_model import predict as pred_reversion  # noqa: E402

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "features_with_predictions.parquet"


class ModelPredictionMaster:
    """Run all trained models and append predictions to dataframe."""

    def __init__(self) -> None:
        self.predictors = {
            "pred_1m_up": pred_1m.predict,
            "pred_5m_up": pred_5m.predict,
            "pred_15m_up": pred_15m.predict,
            "pred_breakout": pred_breakout.predict,
            "pred_reversion": pred_reversion.predict,
        }

    def run_all(self, df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
        output = df.copy()
        for col, fn in self.predictors.items():
            output[col] = fn(output)

        if save:
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            output.to_parquet(OUTPUT_PATH)
        return output


if __name__ == "__main__":
    print("Use ModelPredictionMaster.run_all(df) to generate predictions.")
