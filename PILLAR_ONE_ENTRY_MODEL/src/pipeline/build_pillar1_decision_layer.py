"""
Pipeline entrypoint for Pillar 1 — Decision Layer.

Inputs:
  - data/processed/features_pillar1_with_calib.parquet
    (must contain raw_entry_confidence, entry_probability, regime features, etc.)

Outputs:
  - data/processed/features_pillar1_final.parquet

Adds:
  - entry_decision  ('long' / 'short' / 'none')
  - entry_direction (1 / -1 / 0)
  - entry_confidence ∈ [0, 1]
  - signal_metadata  (compact string summary per row)

These outputs feed:
  - Pillar 2 (Exit Model)
  - Pillar 5 (Execution Engine)

Run:
  python -m src.pipeline.build_pillar1_decision_layer
"""

import logging
from pathlib import Path

import pandas as pd

from src.models.pillar1_decision_layer import DecisionConfig, apply_decision_layer

BASE_DIR = Path(__file__).resolve().parents[3]
P1_BASE = BASE_DIR / "PILLAR_ONE_ENTRY_MODEL"

INPUT_PATH = P1_BASE / "data" / "processed" / "features_pillar1_with_calib.parquet"
OUTPUT_PATH = P1_BASE / "data" / "processed" / "features_pillar1_final.parquet"


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"DecisionLayer: input parquet not found: {INPUT_PATH}")

    logging.info("Loading calibrated features: %s", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)
    _log_mem(df, "[Decision] Loaded features_with_calib")

    # Ensure chronological order (good hygiene even though previous steps already did it)
    if "open_time_ms" in df.columns:
        df = df.sort_values("open_time_ms").reset_index(drop=True)

    # Default config; you can tweak thresholds later if needed
    cfg = DecisionConfig(
        long_threshold=0.55,
        short_threshold=0.45,
        min_confidence_for_entry=0.10,
    )

    logging.info("Applying Pillar 1 decision layer with config: %s", cfg)
    df_out = apply_decision_layer(df, cfg=cfg)
    _log_mem(df_out, "[Decision] After decision layer")

    # Quick sanity snapshot
    if "entry_decision" in df_out.columns and "entry_probability" in df_out.columns:
        dec_counts = df_out["entry_decision"].value_counts(dropna=False)
        logging.info("[Decision] Final decision counts:\n%s", dec_counts.to_string())

        prob_desc = df_out["entry_probability"].describe()
        logging.info("[Decision] entry_probability summary:\n%s", prob_desc.to_string())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Saved Pillar 1 final features (with decisions) -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()