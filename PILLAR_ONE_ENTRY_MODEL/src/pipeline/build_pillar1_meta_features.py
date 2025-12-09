
"""
Pipeline entrypoint for Pillar 1 — Meta-Feature Engine.

Loads:
  - data/processed/features_pillar1_with_ml.parquet

Computes meta-features:
  - meta_agreement_score
  - meta_conflict_score
  - meta_vol_weighted_mom
  - meta_structure_aligned_mom
  - meta_regime_trend_confirmation
  - meta_breakout_alignment
  - meta_horizon_consistency
  - meta_directional_persistence

Writes:
  - data/processed/features_pillar1_with_meta.parquet

Run:
  python -m src.pipeline.build_pillar1_meta_features
"""

import logging
from pathlib import Path

import pandas as pd

from src.features.pillar1_meta_features import (
    MetaFeatureConfig,
    build_pillar1_meta_features,
    summarize_meta_features,
)

BASE_DIR = Path(__file__).resolve().parents[3]
P1_BASE = BASE_DIR / "PILLAR_ONE_ENTRY_MODEL"

INPUT_PATH = P1_BASE / "data" / "processed" / "features_pillar1_with_ml.parquet"
OUTPUT_PATH = P1_BASE / "data" / "processed" / "features_pillar1_with_meta.parquet"


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input features parquet not found: {INPUT_PATH}")

    logging.info("Loading features + ML predictions: %s", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)
    _log_mem(df, "[Meta] Loaded features_with_ml")

    if "open_time_ms" in df.columns:
        df = df.sort_values("open_time_ms").reset_index(drop=True)

    cfg = MetaFeatureConfig()
    logging.info("Building Pillar 1 meta-features with config: %s", cfg)
    df_out = build_pillar1_meta_features(df, cfg)

    _log_mem(df_out, "[Meta] After meta-feature construction")
    summarize_meta_features(df_out)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Saved Pillar 1 features with meta-features -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()