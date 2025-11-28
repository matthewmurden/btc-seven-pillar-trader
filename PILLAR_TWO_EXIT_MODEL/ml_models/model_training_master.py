"""Master trainer to fit all Pillar Two models."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ModelTrainingMaster:
    """Runs training for all multi-horizon and pattern models."""

    def train_all(self) -> None:
        from .model_1m import train as train_1m
        from .model_5m import train as train_5m
        from .model_15m import train as train_15m
        from .breakout_model import train as train_breakout
        from .reversion_model import train as train_reversion

        train_1m.train()
        train_5m.train()
        train_15m.train()
        train_breakout.train()
        train_reversion.train()

        print("Completed training for all models.")


if __name__ == "__main__":
    ModelTrainingMaster().train_all()
