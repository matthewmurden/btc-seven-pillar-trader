from __future__ import annotations

"""
Live helper to apply the Pillar 1 decision layer using calibrated probabilities.

Inputs:
  RUN_LAYER/data/processed/PILLAR_ONE_ENTRY_MODEL/features_pillar1_with_calib.parquet

Outputs:
  RUN_LAYER/data/processed/PILLAR_ONE_ENTRY_MODEL/features_pillar1_final.parquet
"""

import logging
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parents[2]
RUN_PROCESSED = BASE_DIR / "RUN_LAYER" / "data" / "processed"

P1_PROCESSED = RUN_PROCESSED / "PILLAR_ONE_ENTRY_MODEL"
INPUT_PATH = P1_PROCESSED / "features_pillar1_with_calib.parquet"
OUTPUT_PATH = P1_PROCESSED / "features_pillar1_final.parquet"

# Pillar 1 src
P1_BASE = BASE_DIR / "PILLAR_ONE_ENTRY_MODEL"
P1_SRC = P1_BASE / "src"
for p in (P1_BASE, P1_SRC):
    ps = str(p)
    if ps not in sys.path:
        sys.path.append(ps)

# Pillar 6 shared decision config (optional)
P6_SRC = BASE_DIR / "PILLAR_SIX_SAFETY_LAYER" / "src"
if str(P6_SRC) not in sys.path:
    sys.path.append(str(P6_SRC))

try:  # noqa: SIM105
    from safety.p1_decision_config import P1_DECISION_CONFIG  # type: ignore
except Exception:  # noqa: BLE001
    P1_DECISION_CONFIG = None
    logging.warning("P1_DECISION_CONFIG not found; using defaults.")

from models.pillar1_decision_layer import (  # type: ignore
    DecisionConfig,
    apply_decision_layer,
)


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"DecisionLayer: input parquet not found: {INPUT_PATH}")

    logging.info("Loading calibrated features (live): %s", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)
    _log_mem(df, "[Decision] Loaded features_with_calib")

    if "open_time_ms" in df.columns:
        df = df.sort_values("open_time_ms").reset_index(drop=True)

    if P1_DECISION_CONFIG is not None:
        cfg = DecisionConfig(**P1_DECISION_CONFIG)
        logging.info("Applying Pillar 1 decision layer (live) with shared config: %s", cfg)
    else:
        cfg = DecisionConfig(
            long_threshold=0.55,
            short_threshold=0.45,
            min_confidence_for_entry=0.10,
        )
        logging.info("Applying Pillar 1 decision layer (live) with local defaults: %s", cfg)

    df_out = apply_decision_layer(df, cfg=cfg)
    _log_mem(df_out, "[Decision] After decision layer")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Saved Pillar 1 final features (live) -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
