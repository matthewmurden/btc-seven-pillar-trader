from __future__ import annotations

"""
Live helper to compute Pillar 1 deterministic mini-engine scores.

Mirrors PILLAR_ONE_ENTRY_MODEL/src/pipeline/build_pillar1_mini_engines.py but
uses the live inputs under RUN_LAYER/data/processed and writes outputs to the
same live processed tree.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Paths
THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parents[2]
RUN_PROCESSED = BASE_DIR / "RUN_LAYER" / "data" / "processed"

P1_PROCESSED = RUN_PROCESSED / "PILLAR_ONE_ENTRY_MODEL"
INPUT = P1_PROCESSED / "features_pillar1_primitives.parquet"
OUTPUT = P1_PROCESSED / "features_pillar1_with_engines.parquet"

# Pillar 1 src
P1_BASE = BASE_DIR / "PILLAR_ONE_ENTRY_MODEL"
P1_SRC = P1_BASE / "src"
if str(P1_SRC) not in sys.path:
    sys.path.append(str(P1_SRC))

from engines.pillar1_mini_engines import (  # type: ignore
    Pillar1MiniEnginesConfig,
    compute_pillar1_mini_engine_scores,
    summarize_mini_engine_scores,
)


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not INPUT.exists():
        raise FileNotFoundError(f"Primitives parquet missing: {INPUT}")

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
