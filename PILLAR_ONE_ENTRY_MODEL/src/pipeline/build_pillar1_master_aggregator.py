"""
Pipeline entrypoint for Pillar 1 — Regime-Weighted Master Aggregator.

Loads:
  - data/processed/features_pillar1_with_meta.parquet

Computes:
  - raw_entry_confidence ∈ [-1, 1]
  - diagnostic components: agg_trend_block, agg_reversion_block, agg_ml_block,
    agg_breakout_block, agg_meta_block, agg_global_gate

Writes:
  - data/processed/features_pillar1_with_agg.parquet

Run:
  python -m src.pipeline.build_pillar1_master_aggregator
"""

import logging
from pathlib import Path

import pandas as pd

from src.models.pillar1_master_aggregator import (
    MasterAggregatorConfig,
    compute_master_aggregator,
)

BASE_DIR = Path(__file__).resolve().parents[3]
P1_BASE = BASE_DIR / "PILLAR_ONE_ENTRY_MODEL"

INPUT_PATH = P1_BASE / "data" / "processed" / "features_pillar1_with_meta.parquet"
OUTPUT_PATH = P1_BASE / "data" / "processed" / "features_pillar1_with_agg.parquet"


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input parquet not found: {INPUT_PATH}")

    logging.info("Loading features + meta-features: %s", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)
    _log_mem(df, "[Agg] Loaded features_with_meta")

    cfg = MasterAggregatorConfig()
    logging.info("Computing regime-weighted master aggregation with config: %s", cfg)

    df_out = compute_master_aggregator(df, cfg=cfg)
    _log_mem(df_out, "[Agg] After master aggregation")

    # Quick sanity on raw_entry_confidence
    if "raw_entry_confidence" not in df_out.columns:
        raise RuntimeError("raw_entry_confidence not found after aggregation.")

    rec = df_out["raw_entry_confidence"]
    logging.info(
        "raw_entry_confidence summary:\n%s",
        rec.describe().to_string(),
    )

    # Basic correlation sanity with key pieces
    cols_for_corr = [
        c
        for c in [
            "raw_entry_confidence",
            "momentum_score",
            "trend_score",
            "reversion_score",
            "orderflow_score",
            "meta_agreement_score",
            "meta_conflict_score",
            "meta_directional_persistence",
        ]
        if c in df_out.columns
    ]
    if len(cols_for_corr) > 1:
        corr = df_out[cols_for_corr].corr()
        logging.info("Correlations (raw_entry_confidence vs key components):\n%s", corr.to_string())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Saved Pillar 1 features with master aggregator -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()