"""
Pipeline entrypoint for Pillar 1 mini-engines (deterministic signal layers).

- Loads primitive feature matrix from:
    data/processed/features_pillar1_primitives.parquet
- Computes mini-engine scores (momentum/trend/vol/reversion/orderflow/structure/pattern).
- Writes:
    data/processed/features_pillar1_with_engines.parquet
- Logs shapes, memory, and diagnostics.
"""

from pathlib import Path
import logging

import pandas as pd

from src.engines.pillar1_mini_engines import (
    Pillar1MiniEnginesConfig,
    compute_pillar1_mini_engine_scores,
    summarize_mini_engine_scores,
)

BASE_DIR = Path(__file__).resolve().parents[3]
INPUT = BASE_DIR / "PILLAR_ONE_ENTRY_MODEL" / "data" / "processed" / "features_pillar1_primitives.parquet"
OUTPUT = BASE_DIR / "PILLAR_ONE_ENTRY_MODEL" / "data" / "processed" / "features_pillar1_with_engines.parquet"


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    logging.info("Loading primitives from %s", INPUT)
    df = pd.read_parquet(INPUT)
    _log_mem(df, "[MiniEngines] Loaded primitives")

    cfg = Pillar1MiniEnginesConfig()
    logging.info("Computing mini-engine scores...")
    df = compute_pillar1_mini_engine_scores(df, cfg)
    _log_mem(df, "[MiniEngines] After computing scores")

    summarize_mini_engine_scores(df)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)
    logging.info("Saved Pillar 1 features with mini-engine scores -> %s", OUTPUT)


if __name__ == "__main__":
    main()
